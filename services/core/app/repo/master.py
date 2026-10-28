"""Master data reads/writes: org policy, vendors, items, POs/GRNs, contracts, GL accounts."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from psycopg import AsyncConnection

from app.domain import ContractPrice, VendorRecord

from .sql import J, all_, one, val

DEFAULT_POLICY: dict[str, Any] = {
    "confidence_gate": Decimal("0.900"),
    "field_floor": Decimal("0.750"),
    "dup_hold_threshold": Decimal("0.900"),
    "dup_suggest_threshold": Decimal("0.750"),
    "fraud_hold_threshold": Decimal("0.700"),
    "over_receipt_allowance": Decimal("0"),
    "manager_limit": Decimal("5000.00"),
    "controller_notify_threshold": Decimal("500.00"),
    "sla_hours": 24,
    "tolerance_pct": Decimal("0.02"),
    "tolerance_abs": Decimal("25.00"),
    "timezone": "America/New_York",
    "is_demo": False,
    "slug": None,
}


async def org_policy(conn: AsyncConnection, org_id: str) -> dict[str, Any]:
    row = await one(
        conn,
        "select o.timezone, o.is_demo, o.slug::text as slug, s.confidence_gate, s.field_floor, s.dup_hold_threshold, "
        " s.dup_suggest_threshold, s.fraud_hold_threshold, s.over_receipt_allowance, s.manager_limit, "
        " s.controller_notify_threshold, s.sla_hours, "
        " (select pct from tolerance_policies t where t.org_id = o.id and t.effective_from <= now() "
        "   order by effective_from desc limit 1) as tolerance_pct, "
        " (select abs_amount from tolerance_policies t where t.org_id = o.id and t.effective_from <= now() "
        "   order by effective_from desc limit 1) as tolerance_abs "
        "from orgs o left join org_settings s on s.org_id = o.id where o.id = %s",
        (org_id,),
    )
    out = dict(DEFAULT_POLICY)
    if row:
        out.update({k: v for k, v in row.items() if v is not None})
    return out


# ------------------------------------------------------------------------------------------- vendors


async def vendor_records(conn: AsyncConnection, org_id: str) -> list[VendorRecord]:
    rows = await all_(
        conn,
        "select v.id::text as id, v.display_name, v.tax_id_hash, "
        " coalesce(array_agg(a.alias) filter (where a.alias is not null), '{}') as aliases "
        "from vendors v left join vendor_aliases a on a.vendor_id = v.id where v.org_id = %s "
        "group by v.id order by v.created_at",
        (org_id,),
    )
    return [VendorRecord(r["id"], r["display_name"], tuple(r["aliases"]), r["tax_id_hash"]) for r in rows]


async def get_vendor(conn: AsyncConnection, vendor_id: str) -> dict[str, Any] | None:
    return await one(conn, "select * from vendors where id = %s", (vendor_id,))


async def create_unverified_vendor(
    conn: AsyncConnection, *, org_id: str, name: str, remit: Any, at: datetime
) -> dict[str, Any]:
    row = await one(
        conn,
        "insert into vendors (org_id, display_name, legal_name, category, remit_address, remit_address_changed_at, "
        " w9_on_file, created_at) values (%s, %s, %s, 'Unverified', %s, %s, false, %s) returning *",
        (org_id, name, name, J(remit) if remit else None, at if remit else None, at),
    )
    assert row is not None
    return row


async def vendor_history_totals(
    conn: AsyncConnection, org_id: str, vendor_id: str, before: datetime, exclude_invoice: str
) -> list[Decimal]:
    rows = await all_(
        conn,
        "select total from invoices where org_id = %s and vendor_id = %s and id <> %s and total is not null "
        " and received_at < %s and received_at >= %s - interval '90 days' "
        " and status not in ('duplicate','rejected') order by received_at desc limit 500",
        (org_id, vendor_id, exclude_invoice, before, before),
    )
    return [r["total"] for r in rows]


# ------------------------------------------------------------------------------------------- locations / GL


async def location(conn: AsyncConnection, location_id: str) -> dict[str, Any] | None:
    return await one(
        conn,
        "select l.*, e.short_code as entity_short_code, e.name as entity_name, e.erp, e.erp_company_ref "
        "from locations l join entities e on e.id = l.entity_id where l.id = %s",
        (location_id,),
    )


async def entity(conn: AsyncConnection, entity_id: str) -> dict[str, Any] | None:
    return await one(conn, "select * from entities where id = %s", (entity_id,))


async def gl_by_number(
    conn: AsyncConnection, org_id: str, entity_id: str | None, number: str
) -> dict[str, Any] | None:
    return await one(
        conn,
        "select * from gl_accounts where org_id = %s and number = %s and (entity_id = %s or entity_id is null) "
        "order by entity_id nulls last limit 1",
        (org_id, number, entity_id),
    )


async def gl_neighbours(
    conn: AsyncConnection, org_id: str, vendor_id: str, item_ids: list[str], at: datetime, k: int = 7
) -> dict[str, list[dict[str, Any]]]:
    """Historical coded lines for kNN, per item: the k most recent coded lines of the same vendor and item
    in the last 90 days (similarity 1.0 — exact item neighbours). Items without history get no
    neighbours and predict_gl falls back to the item default GL."""
    if not item_ids:
        return {}
    rows = await all_(
        conn,
        "select item_id, gl_code, similarity from ("
        " select l.item_id::text as item_id, g.number as gl_code, 1.0 as similarity, "
        "  row_number() over (partition by l.item_id order by i.received_at desc) as rn "
        " from invoices i join invoice_lines l on l.invoice_id = i.id join gl_accounts g on g.id = l.gl_account_id "
        " where i.org_id = %s and i.vendor_id = %s and l.item_id = any(%s) "
        "  and i.received_at between %s - interval '90 days' and %s) x where rn <= %s",
        (org_id, vendor_id, item_ids, at, at, k),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["item_id"], []).append(r)
    return out


# ------------------------------------------------------------------------------------------- items


async def items_by_ids(conn: AsyncConnection, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    rows = await all_(
        conn,
        "select id::text as id, item_key, description, uom, default_gl from items where id = any(%s)",
        (ids,),
    )
    return {r["id"]: r for r in rows}


async def org_items(
    conn: AsyncConnection, org_id: str, key_prefix: str | None = None
) -> list[dict[str, Any]]:
    if key_prefix:
        return await all_(
            conn,
            "select id::text as id, item_key, description, uom, default_gl from items where org_id = %s and item_key like %s",
            (org_id, key_prefix + "%"),
        )
    return await all_(
        conn,
        "select id::text as id, item_key, description, uom, default_gl from items where org_id = %s",
        (org_id,),
    )


# ------------------------------------------------------------------------------------------- POs / GRNs


async def po_by_number(conn: AsyncConnection, org_id: str, number: str) -> dict[str, Any] | None:
    exact = await one(
        conn, "select * from purchase_orders where org_id = %s and po_number = %s", (org_id, number.strip())
    )
    if exact is not None:
        return exact
    return await one(
        conn,
        "select * from purchase_orders where org_id = %s and "
        " upper(regexp_replace(po_number, '[^A-Za-z0-9]', '', 'g')) = upper(regexp_replace(%s, '[^A-Za-z0-9]', '', 'g')) "
        "limit 1",
        (org_id, number),
    )


async def get_po(conn: AsyncConnection, po_id: str) -> dict[str, Any] | None:
    return await one(conn, "select * from purchase_orders where id = %s", (po_id,))


async def po_lines(conn: AsyncConnection, po_id: str) -> list[dict[str, Any]]:
    return await all_(
        conn,
        "select pl.*, pl.id::text as sid, "
        " (select sum(g.qty_received) from grn_lines g where g.po_line_id = pl.id) as grn_qty "
        "from po_lines pl where pl.po_id = %s order by line_no",
        (po_id,),
    )


# Matched and later: an invoice consumes from the moment it passes match, so two invoices for the same goods
# can't both pass while the first waits for a decision. Rejected, duplicate and failed invoices consume nothing.
BILLED_STATUSES = ("matched", "awaiting_decision", "approved", "posting", "posted")


async def invoiced_qty_by_po_line(
    conn: AsyncConnection, po_id: str, exclude_invoice_id: str
) -> dict[str, Any]:
    """Quantity already billed against each line of the PO by *other* invoices (BILLED_STATUSES).

    A line that failed the quantity check on an invoice that was then short-paid consumes only its payable
    quantity at match time, so the vendor can still bill the balance; everything else consumes what it billed.
    """
    rows = await all_(
        conn,
        "select il.po_line_id::text as po_line_id, "
        " sum(case when mlr.verdict in ('qty_over_receipt', 'qty_over_order') and mlr.payable_qty is not null "
        "      and exists (select 1 from approvals a join exceptions e on e.id = a.exception_id "
        "                  where e.invoice_id = i.id and a.action = 'short_pay') "
        "     then least(il.qty, mlr.payable_qty) else il.qty end) as qty "
        "from invoice_lines il "
        "join invoices i on i.id = il.invoice_id "
        "join po_lines pl on pl.id = il.po_line_id "
        "left join match_results mr on mr.invoice_id = i.id "
        "left join match_line_results mlr on mlr.match_result_id = mr.id and mlr.invoice_line_id = il.id "
        "where pl.po_id = %s and il.invoice_id <> %s and i.status = any(%s::public.invoice_status[]) "
        "group by il.po_line_id",
        (po_id, exclude_invoice_id, list(BILLED_STATUSES)),
    )
    return {r["po_line_id"]: r["qty"] for r in rows}


async def grn_numbers(conn: AsyncConnection, po_id: str) -> list[str]:
    rows = await all_(
        conn, "select grn_number from goods_receipts where po_id = %s order by received_at", (po_id,)
    )
    return [r["grn_number"] for r in rows]


async def po_candidates(
    conn: AsyncConnection, org_id: str, vendor_id: str, location_id: str | None, total: Decimal, at: datetime
) -> list[dict[str, Any]]:
    """Open POs for vendor (+location) ordered within 30 days whose total is within 10% (06 §6)."""
    return await all_(
        conn,
        "select id::text as id, po_number, total from purchase_orders where org_id = %s and vendor_id = %s "
        " and (%s::uuid is null or location_id = %s::uuid) and status = 'open' "
        " and ordered_at between %s - interval '30 days' and %s "
        " and abs(total - %s) <= 0.10 * %s order by abs(total - %s) limit 3",
        (org_id, vendor_id, location_id, location_id, at, at, total, total, total),
    )


async def upsert_po(conn: AsyncConnection, org_id: str, po: dict[str, Any]) -> None:
    """Persist a planned PO (corpus dict shape) and its lines; idempotent by (org, po_number)."""
    ent = await val(conn, "select entity_id from locations where id = %s", (po["location_id"],))
    if ent is None:
        raise LookupError(f"location {po['location_id']} not seeded")
    po_id = await val(
        conn,
        "insert into purchase_orders (id, org_id, entity_id, location_id, vendor_id, po_number, status, ordered_at, "
        " source, total) values (%s, %s, %s, %s, %s, %s, 'open', %s, %s, %s) "
        "on conflict (org_id, po_number) do update set ordered_at = excluded.ordered_at, total = excluded.total "
        "returning id",
        (
            po["id"],
            org_id,
            ent,
            po["location_id"],
            po["vendor_id"],
            po["po_number"],
            po["ordered_at"],
            po.get("source", "quickbooks"),
            po["total"],
        ),
    )
    async with conn.cursor() as cur:
        await cur.executemany(
            "insert into po_lines (po_id, line_no, item_id, description, qty, uom, unit_price, amount) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s) on conflict (po_id, line_no) do update set "
            " item_id = excluded.item_id, description = excluded.description, qty = excluded.qty, uom = excluded.uom, "
            " unit_price = excluded.unit_price, amount = excluded.amount",
            [
                (
                    po_id,
                    ln["line_no"],
                    ln.get("item_id"),
                    ln["description"],
                    ln["qty"],
                    ln["uom"],
                    ln["unit_price"],
                    ln["amount"],
                )
                for ln in po["lines"]
            ],
        )


async def upsert_grn(conn: AsyncConnection, org_id: str, grn: dict[str, Any]) -> None:
    po_id = await val(
        conn,
        "select id from purchase_orders where org_id = %s and po_number = %s",
        (org_id, grn["po_number"]),
    )
    if po_id is None:
        raise LookupError(f"PO {grn['po_number']} not persisted")
    grn_id = await val(
        conn,
        "insert into goods_receipts (id, org_id, po_id, grn_number, received_at, received_by) "
        "values (%s, %s, %s, %s, %s, %s) on conflict (org_id, grn_number) do update set "
        " received_at = excluded.received_at, po_id = excluded.po_id returning id",
        (grn["id"], org_id, po_id, grn["grn_number"], grn["received_at"], grn.get("received_by")),
    )
    await conn.execute("delete from grn_lines where grn_id = %s", (grn_id,))
    async with conn.cursor() as cur:
        await cur.executemany(
            "insert into grn_lines (grn_id, po_line_id, qty_received, uom) "
            "select %s, pl.id, %s, %s from po_lines pl where pl.po_id = %s and pl.line_no = %s",
            [(grn_id, ln["qty_received"], ln["uom"], po_id, ln["po_line_no"]) for ln in grn["lines"]],
        )


# ------------------------------------------------------------------------------------------- contracts


async def contract_prices(
    conn: AsyncConnection, vendor_id: str, item_ids: list[str]
) -> dict[str, list[ContractPrice]]:
    if not item_ids:
        return {}
    rows = await all_(
        conn,
        "select cp.item_id::text as item_id, cp.id::text as cp_id, cp.unit_price, cp.uom, cp.clause_ref, "
        " c.effective_from, c.effective_to, c.title from contract_prices cp join contracts c on c.id = cp.contract_id "
        "where c.vendor_id = %s and cp.item_id = any(%s)",
        (vendor_id, item_ids),
    )
    out: dict[str, list[ContractPrice]] = {}
    for r in rows:
        out.setdefault(r["item_id"], []).append(
            ContractPrice(
                unit_price=r["unit_price"],
                clause_ref=r["clause_ref"],
                effective_from=r["effective_from"],
                effective_to=r["effective_to"],
                uom=r["uom"],
                contract_price_id=r["cp_id"],
                contract_title=r["title"],
            )
        )
    return out


async def prior_short_pays(
    conn: AsyncConnection,
    org_id: str,
    vendor_id: str,
    before: datetime,
    exclude_invoice: str,
    *,
    item_id: str | None = None,
    location_id: str | None = None,
) -> list[Decimal]:
    """Hold amounts of accepted short-pays for "the same variance" — same vendor, and when given the same
    item at the same location — in the 90 days before `before`, oldest first (01 §4.6, 06 §8)."""
    rows = await all_(
        conn,
        "select a.hold_amount from approvals a join exceptions e on e.id = a.exception_id "
        "join invoices i on i.id = e.invoice_id where a.org_id = %s and i.vendor_id = %s and a.action = 'short_pay' "
        " and i.id <> %s and a.decided_at < %s and a.decided_at >= %s - interval '90 days' "
        " and a.hold_amount is not null "
        " and (%s::uuid is null or i.location_id = %s::uuid) "
        " and (%s::uuid is null or exists (select 1 from invoice_lines l where l.invoice_id = i.id and l.item_id = %s::uuid)) "
        "order by a.decided_at",
        (org_id, vendor_id, exclude_invoice, before, before, location_id, location_id, item_id, item_id),
    )
    return [r["hold_amount"] for r in rows]


def day_bounds(d: date) -> tuple[date, date]:
    return d, d + timedelta(days=1)
