"""04 Three-way match (06 §8): price vs contract/PO with tolerance, qty vs GRN/PO, recommendation,
exception with reason + SLA. Clean → posting; otherwise → approval (parks on the exception)."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from app.domain import MatchLineInput, PoCandidate, fmt_money, match_invoice, recommend
from app.repo import exceptions as exc_repo
from app.repo import invoices as inv_repo
from app.repo import journal, master

from ..context import RunRow, StageContext
from ..stages import Stage
from ..transitions import FatalStageError, Transition
from .common import ROLE_LABEL, TYPE_LABEL, dec
from .enrichment import pair_with_po


async def _po_candidate(ctx: StageContext, inv: dict[str, Any]) -> PoCandidate | None:
    if not inv["vendor_id"] or inv["total"] is None:
        return None
    cands = await master.po_candidates(
        ctx.conn,
        inv["org_id"],
        str(inv["vendor_id"]),
        str(inv["location_id"]) if inv["location_id"] else None,
        inv["total"],
        inv["received_at"],
    )
    if not cands:
        return None
    c = cands[0]
    score = (
        max(Decimal("0.5"), 1 - abs(c["total"] - inv["total"]) / inv["total"])
        if inv["total"]
        else Decimal("0.5")
    )
    return PoCandidate(c["po_number"], score.quantize(Decimal("0.001")), c["id"])


def _assigned_role(etype: str, at_risk: Decimal, pol: dict[str, Any]) -> str:
    return "controller" if at_risk > Decimal(pol["manager_limit"]) else "ap_manager"


async def handle(ctx: StageContext, run: RunRow) -> Transition:
    conn = ctx.conn
    if not run.invoice_id:
        raise FatalStageError("match without invoice")
    t0 = time.perf_counter()
    inv = await inv_repo.get_invoice(conn, run.invoice_id)
    assert inv is not None
    pol = ctx.policy
    now = ctx.now()
    po_id = str(inv["po_id"]) if inv["po_id"] else None
    lines = await inv_repo.lines(conn, run.invoice_id)

    # link_po (or a PO linked after enrichment): re-pair against the linked PO
    if po_id and (run.context.get("paired_po") != po_id) and any(ln["po_line_id"] is None for ln in lines):
        await pair_with_po(ctx, run.invoice_id, po_id)
        lines = await inv_repo.lines(conn, run.invoice_id)

    pls = {str(pl["id"]): pl for pl in (await master.po_lines(conn, po_id) if po_id else [])}
    billed = await master.invoiced_qty_by_po_line(conn, po_id, run.invoice_id) if po_id else {}
    vendor = await master.get_vendor(conn, str(inv["vendor_id"])) if inv["vendor_id"] else None
    item_ids = [str(ln["item_id"]) for ln in lines if ln["item_id"]]
    cps = await master.contract_prices(conn, str(inv["vendor_id"]), item_ids) if inv["vendor_id"] else {}
    inputs = []
    for ln in lines:
        pl = pls.get(str(ln["po_line_id"])) if ln["po_line_id"] else None
        inputs.append(
            MatchLineInput(
                line_no=ln["line_no"],
                description=ln["description"],
                qty=dec(ln["qty"]) or Decimal(0),
                unit_price=dec(ln["unit_price"]) or Decimal(0),
                amount=dec(ln["amount"]) or Decimal(0),
                uom=ln["uom"],
                po_unit_price=pl["unit_price"] if pl else None,
                po_qty=pl["qty"] if pl else None,
                grn_qty=pl["grn_qty"] if pl else None,
                prior_invoiced_qty=(dec(billed.get(str(pl["id"]))) or Decimal(0)) if pl else Decimal(0),
                contract_prices=tuple(cps.get(str(ln["item_id"]), ())) if ln["item_id"] else (),
                invoice_line_id=ln["sid"],
                po_line_id=str(pl["id"]) if pl else None,
            )
        )
    inv_date = inv["invoice_date"] or now.date()
    m = match_invoice(
        inputs,
        invoice_date=inv_date,
        has_po=po_id is not None,
        invoice_total=inv["total"],
        match_mode=(vendor or {}).get("match_mode", "three_way"),
        over_receipt_allowance=pol["over_receipt_allowance"],
        pct=pol["tolerance_pct"],
        abs_amount=pol["tolerance_abs"],
    )
    duration = int((time.perf_counter() - t0) * 1000)
    line_rows = []
    for lm in m.lines:
        tol = lm.tolerance
        line_rows.append(
            {
                "invoice_line_id": lm.invoice_line_id,
                "po_line_id": lm.po_line_id,
                "grn_qty": lm.grn_qty,
                "expected_unit_price": lm.expected_unit,
                "expected_source": lm.expected_source,
                "contract_price_id": lm.contract.contract_price_id if lm.contract else None,
                "expected_amount": tol.expected_ext if tol else None,
                "delta_amount": tol.delta if tol else None,
                "delta_pct": tol.delta_pct.quantize(Decimal("0.00001"))
                if tol and tol.delta_pct is not None
                else None,
                "allowed_amount": tol.allowed if tol else None,
                "verdict": lm.verdict,
                "note": lm.note,
                "payable_qty": lm.qty.payable_qty if lm.qty else None,
            }
        )
    await inv_repo.replace_match(
        conn,
        invoice_id=run.invoice_id,
        header={
            "verdict": m.verdict,
            "tolerance_pct": m.tolerance_pct,
            "tolerance_abs": m.tolerance_abs,
            "expected_total": m.expected_total,
            "total_delta": m.total_delta,
            "lines_verified": m.lines_verified,
            "lines_total": m.lines_total,
            "duration_ms": duration,
        },
        line_rows=line_rows,
        at=now,
    )
    grns = await master.grn_numbers(conn, po_id) if po_id else []
    ref = ", ".join([inv["po_number"] or "", *grns[-1:]]).strip(", ") if po_id else "no PO"
    detail = {
        "verdict": m.verdict,
        "lines_verified": m.lines_verified,
        "lines_total": m.lines_total,
        "amount_at_risk": m.amount_at_risk,
        "match_ms": duration,
    }
    ctx_patch = {"paired_po": po_id}
    if m.is_clean:
        await inv_repo.update_invoice(conn, run.invoice_id, status="matched")
        await journal.write_audit(
            conn,
            org_id=run.org_id,
            actor_type="core",
            actor_id="match",
            action="invoice.matched",
            entity_type="invoice",
            entity_id=run.invoice_id,
            after={"verdict": "clean", "lines_verified": m.lines_verified},
            at=now,
        )
        return Transition.advance(
            Stage.POSTING,
            f"Matched — {ref}",
            f"{m.lines_verified} of {m.lines_total} lines within tolerance",
            detail=detail,
            context=ctx_patch,
            duration_ms=duration,
        )

    # ---- exception + recommendation
    etype = m.exception_type or "no_po"
    prior: list[Decimal] = []
    lead = max(m.failing_lines, key=lambda ln: (ln.hold_amount, -ln.line_no)) if m.failing_lines else None
    if etype == "price_variance" and inv["vendor_id"]:
        lead_line = next((ln for ln in lines if lead and ln["sid"] == lead.invoice_line_id), None)
        prior = await master.prior_short_pays(
            conn,
            run.org_id,
            str(inv["vendor_id"]),
            now,
            run.invoice_id,
            item_id=str(lead_line["item_id"]) if lead_line and lead_line["item_id"] else None,
            location_id=str(inv["location_id"]) if inv["location_id"] else None,
        )
    cand = await _po_candidate(ctx, inv) if etype == "no_po" else None
    rec = recommend(
        m,
        invoice_total=inv["total"] or Decimal(0),
        prior_short_pays=prior,
        invoice_date=inv_date,
        po_candidate=cand,
    )
    at_risk = m.amount_at_risk
    reason = m.reason or TYPE_LABEL.get(etype, etype)
    role = _assigned_role(etype, at_risk, pol)
    exc_id = await exc_repo.create_exception(
        conn,
        org_id=run.org_id,
        invoice_id=run.invoice_id,
        run_id=run.id,
        type=etype,
        reason=reason,
        amount_at_risk=at_risk,
        assigned_role=role,
        opened_at=now,
        sla_hours=int(pol["sla_hours"]),
    )
    if rec is not None:
        reasons = [b.as_json() for b in rec.reasons]
        if rec.po_candidate is not None:  # carry the candidate PO id for link_po (UI + replay actor)
            for r in reasons:
                if rec.po_candidate.po_number in r["text"]:
                    r["facts"] = {
                        **r["facts"],
                        "po_id": rec.po_candidate.po_id,
                        "po_number": rec.po_candidate.po_number,
                    }
        await exc_repo.upsert_recommendation(
            conn,
            exception_id=exc_id,
            action=rec.action,
            pay_amount=rec.pay_amount,
            hold_amount=rec.hold_amount,
            confidence=rec.confidence,
            reasons=reasons,
        )
    await inv_repo.update_invoice(conn, run.invoice_id, status="awaiting_decision")
    await journal.write_audit(
        conn,
        org_id=run.org_id,
        actor_type="core",
        actor_id="match",
        action="exception.opened",
        entity_type="exception",
        entity_id=exc_id,
        after={
            "type": etype,
            "reason": reason,
            "amount_at_risk": at_risk,
            "verdict": m.verdict,
            "recommendation": {
                "action": rec.action,
                "pay_amount": rec.pay_amount,
                "hold_amount": rec.hold_amount,
                "confidence": rec.confidence,
            }
            if rec
            else None,
        },
        at=now,
    )
    if at_risk > Decimal(pol["controller_notify_threshold"]):
        await journal.emit_event(
            conn,
            org_id=run.org_id,
            run_id=run.id,
            stage="notify",
            status="succeeded",
            at=now,
            summary=f"Controller notified — {fmt_money(at_risk)} at risk",
            detail={"exception_id": exc_id},
        )
    if ctx.options.drafts and not ctx.options.fast:
        from ..drafts import enqueue_draft

        ctx.after_commit.append(lambda: enqueue_draft(exc_id))
    detail.update(
        {"exception_id": exc_id, "exception_type": etype, "recommendation": rec.action if rec else None}
    )
    head = TYPE_LABEL.get(etype, etype)
    return Transition.advance(
        Stage.APPROVAL,
        f"{head} — {reason}",
        f"{m.verified_text} · {fmt_money(at_risk)} at risk · {ROLE_LABEL[role]}",
        detail=detail,
        context=ctx_patch,
        duration_ms=duration,
    )
