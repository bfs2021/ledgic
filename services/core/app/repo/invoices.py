"""invoices, invoice_fields, invoice_lines, screening and match rows."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from psycopg import AsyncConnection

from .sql import J, all_, one

_INVOICE_COLUMNS = {
    "vendor_id",
    "entity_id",
    "location_id",
    "po_id",
    "invoice_number",
    "invoice_date",
    "due_date",
    "terms",
    "currency",
    "subtotal",
    "tax",
    "freight",
    "total",
    "status",
    "doc_confidence",
    "gate_passed",
    "duplicate_of",
    "touchless",
    "human_touches",
    "extraction_mode",
    "posted_at",
}


async def get_invoice(conn: AsyncConnection, invoice_id: str) -> dict[str, Any] | None:
    return await one(
        conn,
        "select i.*, v.display_name as vendor_name, l.code as location_code, l.name as location_name, "
        " e.short_code as entity_short_code, e.name as entity_name, e.erp as entity_erp, "
        " po.po_number, d.channel, d.source_meta, d.phash, d.sha256 "
        "from invoices i left join vendors v on v.id = i.vendor_id left join locations l on l.id = i.location_id "
        "left join entities e on e.id = i.entity_id left join purchase_orders po on po.id = i.po_id "
        "join documents d on d.id = i.document_id where i.id = %s",
        (invoice_id,),
    )


async def upsert_extracted_invoice(
    conn: AsyncConnection, invoice_id: str, org_id: str, document_id: str, **cols: Any
) -> None:
    """Create (or reset, on re-extraction) the invoice row for a document."""
    bad = set(cols) - _INVOICE_COLUMNS - {"received_at"}
    if bad:
        raise ValueError(f"unknown invoice columns: {bad}")
    names = ["id", "org_id", "document_id", *cols]
    vals = [invoice_id, org_id, document_id, *cols.values()]
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols)
    await conn.execute(
        f"insert into invoices ({', '.join(names)}) values ({', '.join(['%s'] * len(names))}) "
        f"on conflict (id) do update set {updates}",
        vals,
    )
    # re-extraction replaces machine-read fields and lines (human versions are never deleted)
    await conn.execute(
        "delete from invoice_fields where invoice_id = %s and source <> 'human'", (invoice_id,)
    )
    await conn.execute("delete from invoice_lines where invoice_id = %s", (invoice_id,))


async def update_invoice(conn: AsyncConnection, invoice_id: str, **cols: Any) -> None:
    bad = set(cols) - _INVOICE_COLUMNS
    if bad:
        raise ValueError(f"unknown invoice columns: {bad}")
    if not cols:
        return
    sets = ", ".join(f"{c} = %s" for c in cols)
    await conn.execute(f"update invoices set {sets} where id = %s", (*cols.values(), invoice_id))


async def insert_field(
    conn: AsyncConnection,
    *,
    invoice_id: str,
    field: str,
    value: str | None,
    confidence: Decimal,
    raw_confidence: Decimal | None,
    bbox: Any,
    source: str,
    at: datetime,
    version: int = 1,
) -> None:
    await conn.execute(
        "insert into invoice_fields (invoice_id, field, value, confidence, raw_confidence, bbox, source, version, set_at) "
        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict (invoice_id, field, version) do nothing",
        (
            invoice_id,
            field,
            value,
            confidence,
            raw_confidence,
            J(bbox) if bbox else None,
            source,
            version,
            at,
        ),
    )


async def insert_line(
    conn: AsyncConnection,
    *,
    invoice_id: str,
    line_no: int,
    description: str,
    qty: Decimal | None,
    uom: str | None,
    unit_price: Decimal | None,
    amount: Decimal | None,
    confidences: dict[str, Any],
    bboxes: dict[str, Any] | None,
) -> str:
    row = await one(
        conn,
        "insert into invoice_lines (invoice_id, line_no, description, qty, uom, unit_price, amount, confidences, bboxes) "
        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s) returning id::text as id",
        (
            invoice_id,
            line_no,
            description,
            qty,
            uom,
            unit_price,
            amount,
            J(confidences),
            J(bboxes) if bboxes else None,
        ),
    )
    assert row is not None
    return row["id"]


async def lines(conn: AsyncConnection, invoice_id: str) -> list[dict[str, Any]]:
    return await all_(
        conn,
        "select *, id::text as sid from invoice_lines where invoice_id = %s order by line_no",
        (invoice_id,),
    )


async def update_line(conn: AsyncConnection, line_id: str, **cols: Any) -> None:
    allowed = {"item_id", "gl_account_id", "gl_confidence", "po_line_id"}
    bad = set(cols) - allowed
    if bad:
        raise ValueError(f"unknown line columns: {bad}")
    sets = ", ".join(f"{c} = %s" for c in cols)
    await conn.execute(f"update invoice_lines set {sets} where id = %s", (*cols.values(), line_id))


async def latest_fields(conn: AsyncConnection, invoice_id: str) -> dict[str, dict[str, Any]]:
    rows = await all_(
        conn,
        "select distinct on (field) field, value, confidence, raw_confidence, source, version, set_by "
        "from invoice_fields where invoice_id = %s order by field, version desc",
        (invoice_id,),
    )
    return {r["field"]: r for r in rows}


# ------------------------------------------------------------------------------------------- screening


async def duplicate_candidates(
    conn: AsyncConnection,
    *,
    org_id: str,
    vendor_id: str,
    invoice_id: str,
    number: str | None,
    total: Decimal | None,
    invoice_date: date | None,
    at: datetime,
) -> list[dict[str, Any]]:
    """Same vendor, last 180 days, and (same normalized number, or total within $0.01 and invoice date
    within 7 days) — 06 §7. Copies already marked duplicate/rejected are not candidates."""
    return await all_(
        conn,
        "select i.id::text as id, i.invoice_number, i.total, i.invoice_date, i.received_at, i.vendor_id::text as vendor_id, "
        " d.sha256, d.phash, d.channel, d.source_meta "
        "from invoices i join documents d on d.id = i.document_id "
        "where i.org_id = %s and i.vendor_id = %s and i.id <> %s and i.received_at >= %s - interval '180 days' "
        " and i.received_at <= %s and i.status not in ('duplicate','rejected') and ("
        "  (%s::text is not null and i.invoice_number_norm = upper(regexp_replace(%s::text, '[^A-Za-z0-9]', '', 'g'))) "
        "  or (%s::numeric is not null and abs(i.total - %s::numeric) <= 0.01 "
        "      and i.invoice_date between %s::date - 7 and %s::date + 7)) "
        "order by i.received_at limit 20",
        (org_id, vendor_id, invoice_id, at, at, number, number, total, total, invoice_date, invoice_date),
    )


async def insert_duplicate_candidate(
    conn: AsyncConnection,
    *,
    org_id: str,
    invoice_id: str,
    candidate_id: str,
    similarity: Decimal,
    evidence: dict[str, Any],
    at: datetime,
) -> None:
    await conn.execute(
        "delete from duplicate_candidates where invoice_id = %s and candidate_invoice_id = %s",
        (invoice_id, candidate_id),
    )
    await conn.execute(
        "insert into duplicate_candidates (org_id, invoice_id, candidate_invoice_id, similarity, evidence, created_at) "
        "values (%s, %s, %s, %s, %s, %s)",
        (org_id, invoice_id, candidate_id, similarity, J(evidence), at),
    )


async def replace_fraud_signals(
    conn: AsyncConnection,
    *,
    org_id: str,
    invoice_id: str,
    vendor_id: str | None,
    signals: list[dict[str, Any]],
    at: datetime,
) -> None:
    await conn.execute("delete from fraud_signals where invoice_id = %s", (invoice_id,))
    for s in signals:
        await conn.execute(
            "insert into fraud_signals (org_id, invoice_id, vendor_id, signal, weight, detail, created_at) "
            "values (%s, %s, %s, %s, %s, %s, %s)",
            (org_id, invoice_id, vendor_id, s["signal"], s["weight"], J(s.get("detail") or {}), at),
        )


# ------------------------------------------------------------------------------------------- match


async def replace_match(
    conn: AsyncConnection,
    *,
    invoice_id: str,
    header: dict[str, Any],
    line_rows: list[dict[str, Any]],
    at: datetime,
) -> str:
    await conn.execute("delete from match_results where invoice_id = %s", (invoice_id,))
    row = await one(
        conn,
        "insert into match_results (invoice_id, verdict, tolerance_pct, tolerance_abs, expected_total, total_delta, "
        " lines_verified, lines_total, duration_ms, computed_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id::text as id",
        (
            invoice_id,
            header["verdict"],
            header["tolerance_pct"],
            header["tolerance_abs"],
            header["expected_total"],
            header["total_delta"],
            header["lines_verified"],
            header["lines_total"],
            header["duration_ms"],
            at,
        ),
    )
    assert row is not None
    mr_id = row["id"]
    cols = (
        "invoice_line_id",
        "po_line_id",
        "grn_qty",
        "expected_unit_price",
        "expected_source",
        "contract_price_id",
        "expected_amount",
        "delta_amount",
        "delta_pct",
        "allowed_amount",
        "verdict",
        "note",
        "payable_qty",
    )
    async with conn.cursor() as cur:
        await cur.executemany(
            "insert into match_line_results (match_result_id, invoice_line_id, po_line_id, grn_qty, expected_unit_price, "
            " expected_source, contract_price_id, expected_amount, delta_amount, delta_pct, allowed_amount, verdict, note, "
            " payable_qty) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(mr_id, *(r.get(c) for c in cols)) for r in line_rows],
        )
    return mr_id


async def match_hold_by_line(conn: AsyncConnection, invoice_id: str) -> dict[str, dict[str, Any]]:
    rows = await all_(
        conn,
        "select mlr.invoice_line_id::text as line_id, mlr.verdict, mlr.expected_unit_price, mlr.expected_amount, "
        " mlr.delta_amount, mlr.grn_qty from match_line_results mlr join match_results mr on mr.id = mlr.match_result_id "
        "where mr.invoice_id = %s",
        (invoice_id,),
    )
    return {r["line_id"]: r for r in rows}


async def insert_fields(conn: AsyncConnection, rows: list[dict[str, Any]]) -> None:
    """Batch insert of invoice_fields rows (keys as insert_field)."""
    if not rows:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "insert into invoice_fields (invoice_id, field, value, confidence, raw_confidence, bbox, source, version, set_at) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict (invoice_id, field, version) do nothing",
            [
                (
                    r["invoice_id"],
                    r["field"],
                    r["value"],
                    r["confidence"],
                    r["raw_confidence"],
                    J(r["bbox"]) if r.get("bbox") else None,
                    r["source"],
                    r.get("version", 1),
                    r["at"],
                )
                for r in rows
            ],
        )


async def insert_lines(conn: AsyncConnection, rows: list[dict[str, Any]]) -> None:
    """Batch insert of invoice_lines rows (keys as insert_line)."""
    if not rows:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "insert into invoice_lines (invoice_id, line_no, description, qty, uom, unit_price, amount, confidences, bboxes) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    r["invoice_id"],
                    r["line_no"],
                    r["description"],
                    r["qty"],
                    r["uom"],
                    r["unit_price"],
                    r["amount"],
                    J(r["confidences"]),
                    J(r["bboxes"]) if r.get("bboxes") else None,
                )
                for r in rows
            ],
        )


async def update_lines(conn: AsyncConnection, updates: list[tuple[str, dict[str, Any]]]) -> None:
    """Batch `update_line`: updates grouped by column set, one executemany per group."""
    allowed = {"item_id", "gl_account_id", "gl_confidence", "po_line_id"}
    groups: dict[tuple[str, ...], list[tuple[Any, ...]]] = {}
    for line_id, cols in updates:
        bad = set(cols) - allowed
        if bad:
            raise ValueError(f"unknown line columns: {bad}")
        key = tuple(sorted(cols))
        groups.setdefault(key, []).append((*(cols[c] for c in key), line_id))
    async with conn.cursor() as cur:
        for key, params in groups.items():
            sets = ", ".join(f"{c} = %s" for c in key)
            await cur.executemany(f"update invoice_lines set {sets} where id = %s", params)
