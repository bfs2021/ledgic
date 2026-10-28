"""12-testing-qa §3 integration scenarios against local Supabase (`uv run pytest -m integration`)."""

from __future__ import annotations

import io
import json
import uuid
from decimal import Decimal

import httpx
import psycopg
import pytest

from app.api.auth import sign
from app.config import get_settings
from app.db.pool import connection
from app.main import create_app
from app.repo import master
from app.repo.sql import all_, one, val

from .conftest import (
    ORG,
    USERS,
    confirm,
    decide,
    drive,
    events,
    hero_docs,
    invoice_of,
    release,
    reset_hero,
    run_row,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("db")]


async def _exception(run_id: str) -> dict:
    r = await run_row(run_id)
    async with connection() as conn:
        e = await one(conn, "select * from exceptions where id = %s", (r["wait_exception_id"],))
    assert e is not None
    return e


async def test_touchless_bd104882() -> None:
    await reset_hero("BD-104882")
    run_id = await release(hero_docs("BD-104882")[0])
    run = await run_row(run_id)
    assert run["status"] == "succeeded" and run["current_stage"] == "close"
    inv = await invoice_of(run_id)
    assert inv["touchless"] is True and inv["status"] == "posted" and inv["human_touches"] == 0
    assert inv["total"] == Decimal("1982.47") and inv["extraction_mode"] == "recorded"
    evs = await events(run_id)
    assert [e["stage"] for e in evs] == ["intake", "extraction", "enrichment", "screening", "match", "posting", "close"]
    assert evs[0]["summary"] == "Received — email PDF, 2 pages"
    assert evs[1]["summary"] == "Extracted — 14 line items, conf 0.97" and evs[1]["detail"]["gate_passed"] is True
    assert evs[4]["summary"] == "Matched — PO-BS-2041, GRN-BS-5518"
    assert evs[5]["summary"].startswith("Posted — QuickBooks, Bill ")
    assert "GL 5000 COGS-Food" in evs[5]["detail"]["sub"]
    async with connection() as conn:
        p = await one(conn, "select * from erp_postings where invoice_id = %s", (inv["id"],))
        audit = await val(conn, "select count(*) from audit_log where entity_id = %s", (str(inv["id"]),))
    assert p["system"] == "quickbooks" and p["status"] == "posted" and p["response"]["mock"] is True
    assert audit >= 4


async def test_posted_invoice_consumes_its_po_lines() -> None:
    """#32: once BD-104882 posts, its quantities count against the PO for every *other* invoice."""
    await reset_hero("BD-104882")
    run_id = await release(hero_docs("BD-104882")[0])
    inv = await invoice_of(run_id)
    assert inv["status"] == "posted"
    async with connection() as conn:
        lines = await all_(
            conn, "select po_line_id::text as po_line_id, qty from invoice_lines where invoice_id = %s", (inv["id"],)
        )
        for_others = await master.invoiced_qty_by_po_line(conn, str(inv["po_id"]), str(uuid.uuid4()))
        for_itself = await master.invoiced_qty_by_po_line(conn, str(inv["po_id"]), str(inv["id"]))
    expected: dict[str, Decimal] = {}
    for ln in lines:
        if ln["po_line_id"]:
            expected[ln["po_line_id"]] = expected.get(ln["po_line_id"], Decimal(0)) + ln["qty"]
    assert expected and all(for_others.get(k, Decimal(0)) >= v for k, v in expected.items())
    assert all(for_itself.get(k, Decimal(0)) == for_others[k] - v for k, v in expected.items())

    # A line short-paid on quantity consumes only its payable quantity, so the vendor can bill the balance.
    line = next(ln for ln in lines if ln["po_line_id"] and ln["qty"] > 1)
    async with connection() as conn, conn.transaction(force_rollback=True):
        await conn.execute(
            "update match_line_results set verdict = 'qty_over_receipt', payable_qty = %s "
            "where invoice_line_id = (select id from invoice_lines where invoice_id = %s and po_line_id = %s limit 1)",
            (line["qty"] - 1, inv["id"], line["po_line_id"]),
        )
        exc_id = await val(
            conn,
            "insert into exceptions (org_id, invoice_id, type, reason, sla_due_at, status) "
            "values (%s, %s, 'qty_mismatch', 'pytest', now(), 'resolved') returning id",
            (ORG, inv["id"]),
        )
        await conn.execute(
            "insert into approvals (org_id, exception_id, actor_id, action) values (%s, %s, %s, 'short_pay')",
            (ORG, exc_id, USERS["elena"]),
        )
        short_paid = await master.invoiced_qty_by_po_line(conn, str(inv["po_id"]), str(uuid.uuid4()))
    assert short_paid[line["po_line_id"]] == for_others[line["po_line_id"]] - 1


async def test_pr2214_smudge_confirm_resume_posts() -> None:
    await reset_hero("PR-2214")
    run_id = await release(hero_docs("PR-2214")[0])
    run = await run_row(run_id)
    assert run["status"] == "waiting" and run["current_stage"] == "review"
    exc = await _exception(run_id)
    assert exc["type"] == "extraction_review" and exc["assigned_role"] == "ap_clerk"
    inv = await invoice_of(run_id)
    assert inv["doc_confidence"] == Decimal("0.880") and inv["gate_passed"] is False
    async with connection() as conn:
        line = await one(conn, "select id::text as id, qty from invoice_lines where invoice_id = %s and line_no = 2", (inv["id"],))
    res = await confirm("clerk", str(inv["id"]), [{"line_id": line["id"], "key": "qty", "value": "4"}])
    assert res["run_id"] == run_id
    assert await drive(run_id) == "succeeded"
    inv = await invoice_of(run_id)
    assert inv["status"] == "posted" and inv["touchless"] is False and inv["human_touches"] == 1
    stages = [(e["stage"], e["status"]) for e in await events(run_id)]
    assert ("review", "resumed") in stages and ("review", "succeeded") in stages and ("posting", "succeeded") in stages


async def test_ps88214_short_pay_posts_expected_amounts() -> None:
    await reset_hero("PS-88214")
    run_id = await release(hero_docs("PS-88214")[0])
    exc = await _exception(run_id)
    assert exc["type"] == "price_variance" and exc["amount_at_risk"] == Decimal("88.00")
    assert exc["reason"] == "shrimp 12% over contract"
    async with connection() as conn:
        rec = await one(conn, "select * from recommendations where exception_id = %s", (exc["id"],))
    assert (rec["action"], rec["pay_amount"], rec["hold_amount"], rec["confidence"]) == ("short_pay", Decimal("2759.60"), Decimal("88.00"), Decimal("0.940"))
    texts = [r["text"] for r in rec["reasons"]]
    assert texts[1] == "The same variance appeared twice this quarter ($61.60 and $47.20). Both were short-paid and both were accepted."
    await decide("elena", str(exc["id"]), "short_pay")
    assert await drive(run_id) == "succeeded"
    inv = await invoice_of(run_id)
    assert inv["status"] == "posted"
    async with connection() as conn:
        p = await one(conn, "select request from erp_postings where invoice_id = %s", (inv["id"],))
        a = await one(conn, "select pay_amount, hold_amount from approvals where exception_id = %s", (exc["id"],))
    assert p["request"]["TotalAmt"] == "2759.60"
    assert p["request"]["Line"][0]["Amount"] == "736.00"
    assert (a["pay_amount"], a["hold_amount"]) == (Decimal("2759.60"), Decimal("88.00"))


async def test_pr2201_duplicate_merge_second_never_posts() -> None:
    await reset_hero("PR-2201")
    email, photo = hero_docs("PR-2201")
    first = await release(email)
    assert (await run_row(first))["status"] == "succeeded"
    second = await release(photo)
    run = await run_row(second)
    assert run["status"] == "waiting" and run["current_stage"] == "screening"
    exc = await _exception(second)
    assert exc["type"] == "duplicate" and exc["amount_at_risk"] == Decimal("612.40")
    inv2 = await invoice_of(second)
    async with connection() as conn:
        dc = await one(conn, "select similarity, evidence from duplicate_candidates where invoice_id = %s", (inv2["id"],))
    assert dc["similarity"] >= Decimal("0.90")
    assert dc["evidence"]["invoice_number"] == "Exact match" and dc["evidence"]["line_items"] == "5 of 5 overlap"
    await decide("elena", str(exc["id"]), "merge")
    assert await drive(second) == "succeeded"
    inv2 = await invoice_of(second)
    assert inv2["status"] == "duplicate" and inv2["posted_at"] is None
    async with connection() as conn:
        n = await val(conn, "select count(*) from erp_postings where invoice_id = %s", (inv2["id"],))
    assert n == 0


async def test_nv0001_release_needs_controller_and_sod() -> None:
    await reset_hero("NV-0001")
    run_id = await release(hero_docs("NV-0001")[0])
    exc = await _exception(run_id)
    assert exc["type"] == "new_vendor_screen" and exc["assigned_role"] == "controller"
    async with connection() as conn:
        score = await val(conn, "select least(1, sum(weight)) from fraud_signals where invoice_id = %s", (exc["invoice_id"],))
        reason = exc["reason"]
    assert score == Decimal("0.900") and "4 days old" in reason
    extra = {"verified_by_phone": True}
    with pytest.raises(psycopg.Error, match="LEDGIC_FORBIDDEN"):
        await decide("elena", str(exc["id"]), "release_hold", comment="Called vendor", extra=extra)
    with pytest.raises(psycopg.Error, match="LEDGIC_FORBIDDEN"):  # admin created the vendor record (SoD)
        await decide("admin", str(exc["id"]), "release_hold", comment="Called vendor", extra=extra)
    await decide("tom", str(exc["id"]), "release_hold", comment="Verified bank change by phone", extra=extra)
    await drive(run_id)
    run = await run_row(run_id)
    assert run["status"] == "waiting" and run["current_stage"] == "approval"  # no PO → next exception
    nxt = await _exception(run_id)
    assert nxt["type"] == "no_po"


async def test_posting_failure_then_retry() -> None:
    await reset_hero("BD-104882")
    run_id = await release(hero_docs("BD-104882")[0], inject={"posting_failures": 5})
    run = await run_row(run_id)
    assert run["status"] == "waiting" and run["current_stage"] == "posting"
    exc = await _exception(run_id)
    assert exc["type"] == "posting_failed"
    fails = [e for e in await events(run_id) if e["stage"] == "posting" and e["status"] == "failed"]
    assert len(fails) == 5
    await decide("elena", str(exc["id"]), "retry_posting")
    assert await drive(run_id) == "succeeded"
    inv = await invoice_of(run_id)
    assert inv["status"] == "posted" and inv["touchless"] is False
    async with connection() as conn:
        p = await one(conn, "select status, attempts from erp_postings where invoice_id = %s", (inv["id"],))
    assert p["status"] == "posted" and p["attempts"] == 6


async def test_redelivery_creates_no_second_run() -> None:
    from reportlab.pdfgen import canvas

    from app.repo.storage import upload_object

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, f"Ledgic redelivery test {uuid.uuid4()}")
    c.save()
    pdf = buf.getvalue()
    paths = [f"org/{ORG}/uploads/{uuid.uuid4()}.pdf" for _ in range(2)]
    for pth in paths:
        await upload_object(pth, pdf, "application/pdf")
    app = create_app(start_background=False)
    secret = get_settings().core_hmac_secret
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://core") as client:
        out = []
        for pth in paths:
            body = json.dumps({"org_id": ORG, "channel": "upload", "storage_path": pth, "original_filename": "test.pdf"}).encode()
            r = await client.post("/v1/documents/ingest", content=body,
                                  headers={"X-Ledgic-Signature": sign("POST", "/v1/documents/ingest", body, secret), "Content-Type": "application/json"})
            assert r.status_code == 201, r.text
            out.append(r.json()["data"])
    assert out[0]["redelivered"] is False and out[0]["run_id"]
    assert out[1]["redelivered"] is True and out[1]["run_id"] is None
    assert out[1]["document_id"] == out[0]["document_id"]
    async with connection() as conn:
        n = await val(conn, "select count(*) from pipeline_runs where document_id = %s", (out[0]["document_id"],))
        skipped = await all_(conn, "select summary from pipeline_events where run_id = %s and status = 'skipped'", (out[0]["run_id"],))
        # tidy up: the upload run is not part of the demo story
        await conn.execute("update pipeline_runs set wait_exception_id = null where document_id = %s", (out[0]["document_id"],))
        await conn.execute("delete from documents where id = %s", (out[0]["document_id"],))
        await conn.commit()
    assert n == 1 and skipped and skipped[0]["summary"] == "Re-delivery ignored — same SHA-256"
