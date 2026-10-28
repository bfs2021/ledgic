"""Quantity check (01 §4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from .money import ZERO, D, Numberish, fmt_qty

MatchMode = Literal["three_way", "two_way"]


@dataclass(frozen=True)
class QtyResult:
    invoiced: Decimal
    received: Decimal | None  # None for two-way vendors
    ordered: Decimal | None
    prior_invoiced: Decimal  # already billed against this PO line by other invoices
    max_orderable: Decimal | None
    over_receipt: bool
    over_order: bool
    payable_qty: Decimal  # largest qty that passes both checks (≤ invoiced)
    note: str  # 'ordered 12 · received 12 · invoiced 12'

    @property
    def passes(self) -> bool:
        return not (self.over_receipt or self.over_order)

    @property
    def verdict(self) -> str:
        if self.over_receipt:
            return "qty_over_receipt"
        if self.over_order:
            return "qty_over_order"
        return "matched"


def qty_check(
    invoiced: Numberish,
    grn_received: Numberish | None,
    po_qty: Numberish | None,
    over_receipt_allowance: Numberish = ZERO,
    match_mode: MatchMode = "three_way",
    prior_invoiced: Numberish = ZERO,
) -> QtyResult:
    """invoiced ≤ received − prior AND invoiced ≤ po_qty × (1 + allowance) − prior, where prior is what other
    invoices already billed against the same PO line. Two-way vendors skip the GRN check.
    In three-way mode a missing GRN means nothing has been received yet (received = 0)."""
    inv = D(invoiced)
    prior = max(D(prior_invoiced), ZERO)
    ordered = D(po_qty) if po_qty is not None else None
    max_orderable = ordered * (1 + D(over_receipt_allowance)) if ordered is not None else None
    received: Decimal | None
    if match_mode == "two_way":
        received = None
        over_receipt = False
    else:
        received = D(grn_received) if grn_received is not None else ZERO
        over_receipt = inv > received - prior
    over_order = max_orderable is not None and inv > max_orderable - prior

    payable = inv
    if received is not None:
        payable = min(payable, received - prior)
    if max_orderable is not None:
        payable = min(payable, max_orderable - prior)

    parts: list[str] = []
    if ordered is not None:
        parts.append(f"ordered {fmt_qty(ordered)}")
    if match_mode == "two_way":
        if prior:
            parts.append(f"already invoiced {fmt_qty(prior)}")
        parts.append(f"invoiced {fmt_qty(inv)}")
        parts.append("two-way, no receipt check")
    else:
        parts.append(f"received {fmt_qty(grn_received)}" if grn_received is not None else "no receipt")
        if prior:
            parts.append(f"already invoiced {fmt_qty(prior)}")
        parts.append(f"invoiced {fmt_qty(inv)}")
    return QtyResult(
        invoiced=inv,
        received=received,
        ordered=ordered,
        prior_invoiced=prior,
        max_orderable=max_orderable,
        over_receipt=over_receipt,
        over_order=over_order,
        payable_qty=max(payable, ZERO),
        note=" · ".join(parts),
    )
