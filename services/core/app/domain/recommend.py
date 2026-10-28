"""Deterministic recommendation rules (01 §4.6, 06 §8). First match wins; no LLM prose."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from . import templates as t
from .match_engine import LineMatch, MatchResult
from .money import ZERO, D, Numberish, cents, fmt_conf, fmt_money, quantize

BASE_CONFIDENCE = Decimal("0.80")
PER_PRIOR = Decimal("0.07")
MAX_PRIOR = 2
CAP = Decimal("0.95")
PO_PRICE_CONFIDENCE = Decimal("0.70")
QTY_CONFIDENCE = Decimal("0.85")
NO_PO_CONFIDENCE = Decimal("0.50")


@dataclass(frozen=True)
class PoCandidate:
    po_number: str
    score: Decimal
    po_id: str | None = None


@dataclass(frozen=True)
class Recommendation:
    rule: int
    action: str  # decision_action: short_pay | link_po | approve_without_po
    pay_amount: Decimal | None
    hold_amount: Decimal | None
    confidence: Decimal
    reasons: tuple[t.ReasonBullet, ...]
    headline: str  # 'Short-pay to contract price'
    confidence_note: str  # 'Recommendation confidence 0.94, from contract terms and payment history'
    requires_role: str | None = None
    po_candidate: PoCandidate | None = None

    @property
    def summary(self) -> str | None:
        """'Pay $2,759.60, hold $88.00'."""
        if self.pay_amount is None or self.hold_amount is None:
            return None
        return f"Pay {fmt_money(self.pay_amount)}, hold {fmt_money(self.hold_amount)}"

    @property
    def primary_button(self) -> str:
        if self.action == "short_pay" and self.pay_amount is not None:
            return f"Short-pay {fmt_money(self.pay_amount)} and notify vendor"
        if self.action == "link_po" and self.po_candidate:
            return f"Link {self.po_candidate.po_number}"
        return "Approve without PO"

    @property
    def reason_texts(self) -> tuple[str, ...]:
        return tuple(r.text for r in self.reasons)


def short_pay_confidence(prior_accepted: int) -> Decimal:
    """0.80 + 0.07 × min(2, prior accepted short-pays in 90 days), capped at 0.95."""
    return min(CAP, BASE_CONFIDENCE + PER_PRIOR * min(MAX_PRIOR, max(0, prior_accepted)))


def _note(conf: Decimal, source: str) -> str:
    return f"Recommendation confidence {fmt_conf(conf)}, from {source}"


def _grn_full(ln: LineMatch) -> bool:
    q = ln.qty
    return q is not None and q.received is not None and q.received >= q.invoiced and q.passes


def recommend(
    match: MatchResult,
    *,
    invoice_total: Numberish,
    prior_short_pays: Sequence[Numberish] = (),
    invoice_date: date | None = None,
    po_candidate: PoCandidate | None = None,
) -> Recommendation | None:
    """Rules 1–5 of 06 §8. `prior_short_pays` = amounts of this vendor's accepted short-pays in the last
    90 days (most recent first). Returns None for rule 5 (UI shows templates.NO_RECOMMENDATION)."""
    total = D(invoice_total)
    prior = [cents(D(p)) for p in prior_short_pays]
    failing = match.failing_lines
    price_failing = [ln for ln in failing if "price" in ln.failures]
    lead = max(failing, key=lambda ln: (ln.hold_amount, -ln.line_no)) if failing else None

    # Rule 1: price variance, every failing line priced by contract, goods received in full.
    if (
        match.verdict == "price_variance"
        and price_failing
        and all(ln.expected_source == "contract" and ln.contract for ln in price_failing)
        and all(_grn_full(ln) for ln in price_failing)
    ):
        assert lead is not None and lead.contract is not None and lead.tolerance is not None
        hold = cents(sum((ln.hold_amount for ln in price_failing), ZERO))
        conf = quantize(short_pay_confidence(len(prior)), 3)
        bullets = [
            t.contract_price_bullet(
                lead.contract.unit_price,
                lead.contract.uom,
                lead.contract.effective_to,
                lead.tolerance.amount / lead.tolerance.qty if lead.tolerance.qty else ZERO,
                invoice_date,
            )
        ]
        pb = t.prior_short_pays_bullet(prior)
        if pb:
            bullets.append(pb)
        bullets += [t.grn_full_bullet(), t.vendor_notice_bullet(lead.contract.clause_ref)]
        return Recommendation(
            1,
            "short_pay",
            cents(total - hold),
            hold,
            conf,
            tuple(bullets),
            "Short-pay to contract price",
            _note(conf, "contract terms and payment history"),
        )

    # Rule 2: price variance against PO price (no contract on the failing lines).
    if (
        match.verdict == "price_variance"
        and price_failing
        and all(ln.expected_source == "po" for ln in price_failing)
    ):
        assert lead is not None and lead.tolerance is not None and lead.expected_unit is not None
        hold = cents(sum((ln.hold_amount for ln in price_failing), ZERO))
        conf = PO_PRICE_CONFIDENCE
        inv_unit = lead.tolerance.amount / lead.tolerance.qty if lead.tolerance.qty else ZERO
        bullets = [t.po_price_bullet(lead.expected_unit, None, inv_unit), t.no_contract_bullet()]
        pb = t.prior_short_pays_bullet(prior)
        if pb:
            bullets.append(pb)
        bullets.append(t.vendor_notice_bullet(None))
        return Recommendation(
            2,
            "short_pay",
            cents(total - hold),
            hold,
            conf,
            tuple(bullets),
            "Short-pay to PO price",
            _note(conf, "purchase order terms"),
        )

    # Rule 3: quantity mismatch where less was received than invoiced. Not when other invoices already billed
    # the receipt: that is a possible re-bill, not a short delivery, so a person decides.
    short_received = [ln for ln in failing if ln.qty is not None and ln.qty.over_receipt]
    rebilled = any(ln.qty is not None and ln.qty.prior_invoiced > 0 and not ln.qty.passes for ln in failing)
    if match.verdict == "qty_mismatch" and short_received and not rebilled:
        hold = cents(sum((ln.hold_amount for ln in failing), ZERO))
        conf = QTY_CONFIDENCE
        q = short_received[0].qty
        assert q is not None and q.received is not None
        bullets = (
            t.received_short_bullet(q.received, q.invoiced),
            t.pay_received_bullet(),
            t.vendor_notice_bullet(None),
        )
        return Recommendation(
            3,
            "short_pay",
            cents(total - hold),
            hold,
            conf,
            bullets,
            "Short-pay to received quantity",
            _note(conf, "goods receipt"),
        )

    # Rule 4: no PO.
    if match.verdict == "no_po":
        if po_candidate is not None:
            conf = quantize(D(po_candidate.score), 3)
            return Recommendation(
                4,
                "link_po",
                None,
                None,
                conf,
                (t.no_po_bullet(), t.po_candidate_bullet(po_candidate.po_number, conf)),
                f"Link {po_candidate.po_number}",
                _note(conf, "open purchase orders"),
                po_candidate=po_candidate,
            )
        conf = NO_PO_CONFIDENCE
        return Recommendation(
            4,
            "approve_without_po",
            cents(total),
            ZERO,
            conf,
            (t.no_po_bullet(), t.no_po_candidate_bullet(), t.approve_without_po_bullet()),
            "Approve without PO",
            _note(conf, "open purchase orders"),
            requires_role="ap_manager",
        )

    return None  # Rule 5
