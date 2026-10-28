"""Three-way match engine (06 §8): per-line price + quantity verdicts, invoice verdict, amount at risk,
queue reason and the tolerance sentence for the Three-way match screen."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .money import ZERO, D, Numberish, cents, fmt_qty, quantize
from .qty import MatchMode, QtyResult, qty_check
from .tolerance import DEFAULT_ABS, DEFAULT_PCT, ToleranceResult, line_tolerance, tolerance_sentence

FAILURE_TO_EXCEPTION = {"price": "price_variance", "qty": "qty_mismatch", "unmatched": "unmatched_line"}
_WORD = re.compile(r"[\w][\w\-/]*")


@dataclass(frozen=True)
class ContractPrice:
    unit_price: Decimal
    clause_ref: str
    effective_from: date
    effective_to: date | None = None
    uom: str | None = None
    contract_price_id: str | None = None
    contract_title: str | None = None

    def active_on(self, d: date) -> bool:
        return self.effective_from <= d and (self.effective_to is None or d <= self.effective_to)


def active_contract(prices: Sequence[ContractPrice], on: date) -> ContractPrice | None:
    """The contract price active on `on`; latest effective_from wins when several overlap."""
    active = [p for p in prices if p.active_on(on)]
    return max(active, key=lambda p: p.effective_from) if active else None


def item_short(description: str) -> str:
    """'Shrimp 16/20 IQF 2lb × 10' -> 'shrimp' (first word, lowercased)."""
    m = _WORD.search(description)
    return m.group(0).lower() if m else description.strip().lower()


@dataclass(frozen=True)
class MatchLineInput:
    line_no: int  # 1-based, as shown on screen
    description: str
    qty: Decimal
    unit_price: Decimal
    amount: Decimal
    uom: str | None = None
    po_unit_price: Decimal | None = None  # None (and po_qty None) → line not paired to a PO line
    po_qty: Decimal | None = None
    grn_qty: Decimal | None = None  # Σ grn_lines.qty_received for the PO line
    prior_invoiced_qty: Decimal = ZERO  # already billed against the PO line by other invoices
    contract_prices: tuple[ContractPrice, ...] = ()
    invoice_line_id: str | None = None
    po_line_id: str | None = None

    @classmethod
    def of(
        cls,
        line_no: int,
        description: str,
        qty: Numberish,
        unit_price: Numberish,
        amount: Numberish,
        *,
        po_unit_price: Numberish | None = None,
        po_qty: Numberish | None = None,
        grn_qty: Numberish | None = None,
        prior_invoiced_qty: Numberish = ZERO,
        contract_prices: Sequence[ContractPrice] = (),
        **kw: str | None,
    ) -> MatchLineInput:
        opt = lambda v: D(v) if v is not None else None  # noqa: E731
        return cls(
            line_no,
            description,
            D(qty),
            D(unit_price),
            D(amount),
            po_unit_price=opt(po_unit_price),
            po_qty=opt(po_qty),
            grn_qty=opt(grn_qty),
            prior_invoiced_qty=D(prior_invoiced_qty),
            contract_prices=tuple(contract_prices),
            **kw,
        )

    @property
    def paired(self) -> bool:
        return self.po_unit_price is not None or self.po_qty is not None


@dataclass(frozen=True)
class LineMatch:
    line_no: int
    description: str
    verdict: str  # matched | price_variance | qty_over_receipt | qty_over_order | unmatched_line | favorable
    failures: frozenset[str]  # subset of {'price', 'qty', 'unmatched'}
    expected_unit: Decimal | None
    expected_source: str | None  # 'contract' | 'po'
    contract: ContractPrice | None
    tolerance: ToleranceResult | None
    qty: QtyResult | None
    hold_amount: Decimal  # positive amount beyond what is payable on this line
    note: str
    reason: str | None
    invoice_line_id: str | None = None
    po_line_id: str | None = None

    @property
    def expected_ext(self) -> Decimal | None:
        return self.tolerance.expected_ext if self.tolerance else None

    @property
    def delta(self) -> Decimal | None:
        return self.tolerance.delta if self.tolerance else None

    @property
    def grn_qty(self) -> Decimal | None:
        return self.qty.received if self.qty else None

    @property
    def verified(self) -> bool:
        return self.verdict in ("matched", "favorable")


@dataclass(frozen=True)
class MatchResult:
    verdict: str  # clean | price_variance | qty_mismatch | no_po | unmatched_line | mixed
    lines: tuple[LineMatch, ...]
    amount_at_risk: Decimal
    expected_total: Decimal
    total_delta: Decimal
    exception_type: str | None  # dominant exception type, None when clean
    reason: str | None  # one-line queue reason, e.g. 'shrimp 12% over contract'
    tolerance_sentence: str | None
    tolerance_pct: Decimal = DEFAULT_PCT
    tolerance_abs: Decimal = DEFAULT_ABS
    failure_types: frozenset[str] = field(default_factory=frozenset)

    @property
    def lines_total(self) -> int:
        return len(self.lines)

    @property
    def lines_verified(self) -> int:
        return sum(1 for ln in self.lines if ln.verified)

    @property
    def verified_text(self) -> str:
        return f"{self.lines_verified} of {self.lines_total} lines auto-verified"

    @property
    def is_clean(self) -> bool:
        return self.verdict == "clean"

    @property
    def failing_lines(self) -> tuple[LineMatch, ...]:
        return tuple(ln for ln in self.lines if ln.failures)


def _billed(q: QtyResult) -> str:
    return f", {fmt_qty(q.prior_invoiced)} already invoiced" if q.prior_invoiced else ""


def match_line(
    line: MatchLineInput,
    *,
    invoice_date: date,
    match_mode: MatchMode = "three_way",
    over_receipt_allowance: Numberish = ZERO,
    pct: Numberish = DEFAULT_PCT,
    abs_amount: Numberish = DEFAULT_ABS,
) -> LineMatch:
    short = item_short(line.description)
    if not line.paired:
        return LineMatch(
            line.line_no,
            line.description,
            "unmatched_line",
            frozenset({"unmatched"}),
            None,
            None,
            None,
            None,
            None,
            cents(line.amount),
            "not on the purchase order",
            f"{short} not on PO",
            line.invoice_line_id,
            line.po_line_id,
        )
    contract = active_contract(line.contract_prices, invoice_date)
    if contract is not None:
        expected_unit, source = contract.unit_price, "contract"
    elif line.po_unit_price is not None:
        expected_unit, source = line.po_unit_price, "po"
    else:  # paired by qty only with no price reference: bill at invoice price
        expected_unit, source = line.unit_price, "po"
    tol = line_tolerance(expected_unit, line.qty, line.amount, pct, abs_amount)
    q = qty_check(
        line.qty, line.grn_qty, line.po_qty, over_receipt_allowance, match_mode, line.prior_invoiced_qty
    )

    failures: set[str] = set()
    if not tol.passes:
        failures.add("price")
    if not q.passes:
        failures.add("qty")

    if "price" in failures:
        verdict = "price_variance"
    elif "qty" in failures:
        verdict = q.verdict
    elif tol.favorable and (tol.exceeds_pct or tol.exceeds_abs):
        verdict = "favorable"  # under expected beyond tolerance: passes, recorded as favorable
    else:
        verdict = "matched"

    hold = ZERO
    if failures:
        pay_qty = q.payable_qty if "qty" in failures else line.qty
        pay_unit = expected_unit if "price" in failures else line.unit_price
        hold = max(ZERO, cents(line.amount - cents(pay_qty * pay_unit)))

    reason: str | None = None
    if "price" in failures and tol.delta_pct is not None:
        pct_int = quantize(tol.delta_pct * 100, 0)
        reason = f"{short} {pct_int}% over {'contract' if source == 'contract' else 'PO'}"
    elif "price" in failures:
        reason = f"{short} over {'contract' if source == 'contract' else 'PO'}"
    elif q.over_receipt:
        reason = f"{short} invoiced {fmt_qty(q.invoiced)}, received {fmt_qty(q.received or ZERO)}{_billed(q)}"
    elif q.over_order:
        reason = f"{short} invoiced {fmt_qty(q.invoiced)}, ordered {fmt_qty(q.ordered or ZERO)}{_billed(q)}"

    return LineMatch(
        line.line_no,
        line.description,
        verdict,
        frozenset(failures),
        expected_unit,
        source,
        contract,
        tol,
        q,
        hold,
        q.note,
        reason,
        line.invoice_line_id,
        line.po_line_id,
    )


def match_invoice(
    lines: Sequence[MatchLineInput],
    *,
    invoice_date: date,
    has_po: bool = True,
    invoice_total: Numberish | None = None,
    match_mode: MatchMode = "three_way",
    over_receipt_allowance: Numberish = ZERO,
    pct: Numberish = DEFAULT_PCT,
    abs_amount: Numberish = DEFAULT_ABS,
) -> MatchResult:
    p, a = D(pct), D(abs_amount)
    amounts_sum = sum((ln.amount for ln in lines), ZERO)
    if not has_po:
        total = D(invoice_total) if invoice_total is not None else amounts_sum
        no_po_lines = tuple(
            LineMatch(
                ln.line_no,
                ln.description,
                "unmatched_line",
                frozenset({"unmatched"}),
                None,
                None,
                None,
                None,
                None,
                cents(ln.amount),
                "no purchase order",
                None,
                ln.invoice_line_id,
                None,
            )
            for ln in lines
        )
        return MatchResult(
            "no_po",
            no_po_lines,
            cents(total),
            ZERO,
            ZERO,
            "no_po",
            "no purchase order",
            None,
            p,
            a,
            frozenset({"no_po"}),
        )

    results = tuple(
        match_line(
            ln,
            invoice_date=invoice_date,
            match_mode=match_mode,
            over_receipt_allowance=over_receipt_allowance,
            pct=p,
            abs_amount=a,
        )
        for ln in lines
    )
    failure_types = frozenset(f for r in results for f in r.failures)
    if not failure_types:
        verdict = "clean"
    elif len(failure_types) == 1:
        verdict = FAILURE_TO_EXCEPTION[next(iter(failure_types))]
    else:
        verdict = "mixed"

    at_risk = sum((r.hold_amount for r in results), ZERO)
    expected_total = sum((r.expected_ext for r in results if r.expected_ext is not None), ZERO)
    total_delta = sum((r.delta for r in results if r.delta is not None), ZERO)

    exception_type: str | None = None
    reason: str | None = None
    if failure_types:
        by_type: dict[str, Decimal] = {}
        for r in results:
            for f in r.failures:
                by_type[f] = by_type.get(f, ZERO) + r.hold_amount
        order = ("price", "qty", "unmatched")
        dominant = max(failure_types, key=lambda f: (by_type.get(f, ZERO), -order.index(f)))
        exception_type = FAILURE_TO_EXCEPTION[dominant]
        failing = sorted((r for r in results if r.failures), key=lambda r: (-r.hold_amount, r.line_no))
        reason = failing[0].reason
        if len(failing) > 1:
            reason = f"{reason} (+{len(failing) - 1} more)"

    sentence: str | None = None
    price_failing = [r for r in results if "price" in r.failures and r.tolerance]
    if price_failing:
        first = price_failing[0]
        assert first.tolerance is not None
        sentence = tolerance_sentence(first.line_no, first.tolerance)
        if len(price_failing) > 1:
            n = len(price_failing) - 1
            sentence += (
                f" {n} other line{'s' if n != 1 else ''} also exceed{'s' if n == 1 else ''} tolerance."
            )

    return MatchResult(
        verdict,
        results,
        cents(at_risk),
        cents(expected_total),
        cents(total_delta),
        exception_type,
        reason,
        sentence,
        p,
        a,
        failure_types,
    )
