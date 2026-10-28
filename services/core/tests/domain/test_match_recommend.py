from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.domain.match_engine import (
    ContractPrice,
    MatchLineInput,
    active_contract,
    item_short,
    match_invoice,
)
from app.domain.recommend import PoCandidate, recommend, short_pay_confidence
from app.domain.templates import (
    contract_price_bullet,
    prior_short_pays_bullet,
    through_text,
)

INVOICE_DATE = date(2026, 7, 13)
SPEC_CONTRACT = ContractPrice(
    Decimal("3.68"), "4.2", date(2026, 3, 1), date(2026, 9, 30), "lb", contract_title="Vendor agreement 2026"
)
HERO_BULLETS = (
    "The agreement on file fixes this item at $3.68/lb through September; the invoice bills $4.12.",
    "The same variance appeared twice this quarter ($61.60 and $47.20). "
    "Both were short-paid and both were accepted.",
    "Goods were received in full, so there is no quantity basis to dispute — only price.",
    "A vendor notice is drafted with the clause reference attached.",
)


def hero_lines(hero: dict, contract: ContractPrice = SPEC_CONTRACT) -> list[MatchLineInput]:
    out = []
    for i, ln in enumerate(hero["lines"], start=1):
        po_unit = ln.get("po_unit", ln["unit_price"])
        out.append(
            MatchLineInput.of(
                i,
                ln["description"],
                ln["qty"],
                ln["unit_price"],
                ln["amount"],
                uom=ln["uom"],
                po_unit_price=po_unit,
                po_qty=ln["qty"],
                grn_qty=ln["qty"],
                contract_prices=[contract] if "contract_unit" in ln else [],
            )
        )
    return out


def test_item_short() -> None:
    assert item_short("Shrimp 16/20 IQF 2lb × 10") == "shrimp"
    assert item_short("Flat-leaf parsley 60ct") == "flat-leaf"
    assert item_short("  ") == ""


def test_active_contract_selection() -> None:
    newer = ContractPrice(Decimal("3.50"), "5.1", date(2026, 6, 1), None)
    assert active_contract([SPEC_CONTRACT, newer], INVOICE_DATE) is newer
    assert active_contract([SPEC_CONTRACT], date(2026, 10, 1)) is None
    assert active_contract([SPEC_CONTRACT], date(2026, 2, 28)) is None


def test_ps88214_match(heroes: dict) -> None:
    hero = heroes["PS-88214"]
    exp = hero["expected"]
    m = match_invoice(hero_lines(hero), invoice_date=INVOICE_DATE, invoice_total=hero["total"])
    assert m.verdict == exp["verdict"] == "price_variance"
    assert m.exception_type == "price_variance"
    assert m.amount_at_risk == Decimal(exp["hold"])
    assert m.reason == "shrimp 12% over contract"
    assert m.verified_text == "4 of 5 lines auto-verified"
    assert m.lines_verified == 4 and m.lines_total == 5 and not m.is_clean
    assert m.tolerance_sentence == (
        "Line 1 is +12.0% against a ±2% limit and $88.00 against a $25 limit — it fails both, "
        "so the invoice holds for approval."
    )
    shrimp = m.lines[0]
    assert shrimp.verdict == "price_variance" and shrimp.expected_source == "contract"
    assert shrimp.contract is SPEC_CONTRACT and shrimp.contract.clause_ref == "4.2"
    assert shrimp.expected_ext == Decimal(exp["expected_ext"]) and shrimp.delta == Decimal(exp["delta"])
    assert shrimp.tolerance is not None and shrimp.tolerance.allowed == Decimal(exp["allowed"])
    assert shrimp.grn_qty == Decimal(200)
    assert m.lines[1].note == "ordered 12 · received 12 · invoiced 12"
    assert m.lines[1].verdict == "matched" and m.lines[1].expected_source == "po"
    assert m.expected_total == Decimal("2759.60") and m.total_delta == Decimal("88.00")


def test_ps88214_recommendation(heroes: dict) -> None:
    hero = heroes["PS-88214"]
    exp = hero["expected"]
    m = match_invoice(hero_lines(hero), invoice_date=INVOICE_DATE)
    rec = recommend(
        m, invoice_total=hero["total"], prior_short_pays=exp["prior_short_pays"], invoice_date=INVOICE_DATE
    )
    assert rec is not None and rec.rule == 1
    assert rec.action == "short_pay"
    assert rec.pay_amount == Decimal(exp["pay"]) == Decimal("2759.60")
    assert rec.hold_amount == Decimal(exp["hold"])
    assert rec.confidence == Decimal(exp["recommendation_confidence"])
    assert rec.reason_texts == HERO_BULLETS
    assert rec.headline == "Short-pay to contract price"
    assert rec.summary == "Pay $2,759.60, hold $88.00"
    assert rec.confidence_note == "Recommendation confidence 0.94, from contract terms and payment history"
    assert rec.primary_button == "Short-pay $2,759.60 and notify vendor"
    assert rec.reasons[0].as_json()["template_id"] == "contract_price"


def test_rolling_contract_end_month_from_effective_to(heroes: dict) -> None:
    # D8: seed computes effective_to as the last day of the 5th month after seed day.
    rolling = ContractPrice(Decimal("3.68"), "4.2", date(2026, 3, 1), date(2027, 3, 31), "lb")
    seed_day = date(2026, 10, 14)
    m = match_invoice(hero_lines(heroes["PS-88214"], rolling), invoice_date=seed_day)
    rec = recommend(m, invoice_total="2847.60", prior_short_pays=["61.60", "47.20"], invoice_date=seed_day)
    assert rec is not None
    assert rec.reason_texts[0] == (
        "The agreement on file fixes this item at $3.68/lb through March 2027; the invoice bills $4.12."
    )
    same_year = ContractPrice(Decimal("3.68"), "4.2", date(2026, 3, 1), date(2026, 12, 31), "lb")
    m2 = match_invoice(hero_lines(heroes["PS-88214"], same_year), invoice_date=seed_day)
    rec2 = recommend(m2, invoice_total="2847.60", invoice_date=seed_day)
    assert rec2 is not None and "through December;" in rec2.reason_texts[0]


@pytest.mark.parametrize(
    ("prior", "conf", "bullet"),
    [
        ([], "0.80", None),
        (
            ["61.60"],
            "0.87",
            "The same variance appeared once this quarter ($61.60). It was short-paid and accepted.",
        ),
        (
            ["61.60", "47.20", "30.00"],
            "0.94",
            "The same variance appeared 3 times this quarter ($61.60, $47.20 and $30.00). "
            "All were short-paid and all were accepted.",
        ),
    ],
)
def test_prior_variants(heroes: dict, prior: list[str], conf: str, bullet: str | None) -> None:
    m = match_invoice(hero_lines(heroes["PS-88214"]), invoice_date=INVOICE_DATE)
    rec = recommend(m, invoice_total="2847.60", prior_short_pays=prior, invoice_date=INVOICE_DATE)
    assert rec is not None and rec.confidence == Decimal(conf)
    if bullet is None:
        assert len(rec.reasons) == 3
    else:
        assert rec.reason_texts[1] == bullet


def test_confidence_formula_capped() -> None:
    assert short_pay_confidence(0) == Decimal("0.80")
    assert short_pay_confidence(2) == Decimal("0.94")
    assert short_pay_confidence(10) == Decimal("0.94")
    assert short_pay_confidence(-1) == Decimal("0.80")


def test_rule2_po_price_variance() -> None:
    lines = [
        MatchLineInput.of(
            1, "Butter 36ct", "10", "55.00", "550.00", po_unit_price="50.00", po_qty="10", grn_qty="10"
        )
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.reason == "butter 10% over PO"
    rec = recommend(m, invoice_total="550.00", prior_short_pays=["12.00"])
    assert rec is not None and rec.rule == 2 and rec.confidence == Decimal("0.70")
    assert rec.pay_amount == Decimal("500.00") and rec.hold_amount == Decimal("50.00")
    assert rec.reason_texts[0] == "The purchase order prices this item at $50.00; the invoice bills $55.00."
    assert rec.reason_texts[-1] == "A vendor notice is drafted with the purchase order reference attached."
    assert len(rec.reasons) == 4
    assert recommend(m, invoice_total="550.00").reasons[1].template_id == "no_contract"


def test_contract_variance_without_full_receipt_has_no_recommendation() -> None:
    lines = [
        MatchLineInput.of(
            1,
            "Shrimp",
            "200",
            "4.12",
            "824.00",
            po_unit_price="3.68",
            po_qty="200",
            grn_qty="200",
            contract_prices=[SPEC_CONTRACT],
        )
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE, match_mode="two_way")
    assert m.verdict == "price_variance"
    assert recommend(m, invoice_total="824.00") is None


def test_rebilled_receipt_routes_to_qty_exception() -> None:
    line = dict(po_unit_price="50.00", po_qty="10", grn_qty="10")
    first = match_invoice(
        [MatchLineInput.of(1, "Butter 36ct", "10", "50.00", "500.00", **line)], invoice_date=INVOICE_DATE
    )
    assert first.is_clean
    again = match_invoice(
        [MatchLineInput.of(1, "Butter 36ct", "10", "50.00", "500.00", prior_invoiced_qty="10", **line)],
        invoice_date=INVOICE_DATE,
    )
    assert not again.is_clean and again.verdict == "qty_mismatch"
    assert again.lines[0].verdict == "qty_over_receipt"
    assert again.reason == "butter invoiced 10, received 10, 10 already invoiced"
    assert again.lines[0].hold_amount == Decimal("500.00")
    # A possible re-bill is not a short delivery: no short-pay recommendation, a person decides.
    assert recommend(again, invoice_total="500.00") is None


def test_rule3_qty_short_receipt() -> None:
    lines = [
        MatchLineInput.of(
            1, "Cod loin 10lb case", "12", "68.40", "820.80", po_unit_price="68.40", po_qty="12", grn_qty="10"
        ),
        MatchLineInput.of(
            2, "Oysters", "4", "84.35", "337.40", po_unit_price="84.35", po_qty="4", grn_qty="4"
        ),
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.verdict == "qty_mismatch" and m.exception_type == "qty_mismatch"
    assert m.lines[0].verdict == "qty_over_receipt"
    assert m.amount_at_risk == Decimal("136.80")
    assert m.reason == "cod invoiced 12, received 10"
    assert m.tolerance_sentence is None
    rec = recommend(m, invoice_total="1158.20")
    assert rec is not None and rec.rule == 3 and rec.confidence == Decimal("0.85")
    assert rec.pay_amount == Decimal("1021.40") and rec.hold_amount == Decimal("136.80")
    assert rec.reason_texts[0] == "The goods receipt shows 10 received against 12 invoiced."


def test_over_order_only_has_no_recommendation() -> None:
    lines = [
        MatchLineInput.of(1, "Cod", "12", "10.00", "120.00", po_unit_price="10.00", po_qty="10", grn_qty="12")
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.lines[0].verdict == "qty_over_order" and m.reason == "cod invoiced 12, ordered 10"
    assert m.amount_at_risk == Decimal("20.00")
    assert recommend(m, invoice_total="120.00") is None


def test_rule4_no_po() -> None:
    lines = [MatchLineInput.of(1, "Mystery item", "1", "2214.85", "2214.85")]
    m = match_invoice(lines, invoice_date=INVOICE_DATE, has_po=False, invoice_total="2214.85")
    assert m.verdict == "no_po" and m.amount_at_risk == Decimal("2214.85") and m.reason == "no purchase order"
    link = recommend(m, invoice_total="2214.85", po_candidate=PoCandidate("PO-CM-0412", Decimal("0.82")))
    assert link is not None and link.action == "link_po" and link.confidence == Decimal("0.82")
    assert link.primary_button == "Link PO-CM-0412" and link.summary is None
    assert (
        link.reason_texts[1]
        == "PO-CM-0412 is an open purchase order for this vendor and amount (match score 0.82)."
    )
    manual = recommend(m, invoice_total="2214.85")
    assert (
        manual is not None and manual.action == "approve_without_po" and manual.requires_role == "ap_manager"
    )
    assert manual.confidence == Decimal("0.50") and manual.primary_button == "Approve without PO"
    assert manual.pay_amount == Decimal("2214.85") and manual.hold_amount == 0
    assert match_invoice(lines, invoice_date=INVOICE_DATE, has_po=False).amount_at_risk == Decimal("2214.85")


def test_unmatched_line_and_mixed() -> None:
    lines = [
        MatchLineInput.of(
            1,
            "Shrimp 16/20",
            "200",
            "4.12",
            "824.00",
            po_unit_price="3.68",
            po_qty="200",
            grn_qty="200",
            contract_prices=[SPEC_CONTRACT],
        ),
        MatchLineInput.of(2, "Fuel surcharge", "1", "35.00", "35.00"),
        MatchLineInput.of(3, "Cod", "5", "10.00", "50.00", po_unit_price="10.00", po_qty="5", grn_qty="4"),
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.verdict == "mixed" and m.exception_type == "price_variance"
    assert m.lines[1].verdict == "unmatched_line" and m.lines[1].reason == "fuel not on PO"
    assert m.amount_at_risk == Decimal("88.00") + Decimal("35.00") + Decimal("10.00")
    assert m.reason == "shrimp 12% over contract (+2 more)"
    assert recommend(m, invoice_total="909.00") is None
    only_unmatched = match_invoice(lines[1:2], invoice_date=INVOICE_DATE)
    assert only_unmatched.verdict == "unmatched_line" and only_unmatched.exception_type == "unmatched_line"


def test_price_and_qty_on_same_line_and_multi_price_sentence() -> None:
    lines = [
        MatchLineInput.of(
            1,
            "Shrimp",
            "200",
            "4.12",
            "824.00",
            po_unit_price="3.68",
            po_qty="200",
            grn_qty="150",
            contract_prices=[SPEC_CONTRACT],
        ),
        MatchLineInput.of(
            2, "Butter", "10", "55.00", "550.00", po_unit_price="50.00", po_qty="10", grn_qty="10"
        ),
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.verdict == "mixed"
    assert m.lines[0].failures == frozenset({"price", "qty"})
    assert m.lines[0].hold_amount == Decimal("824.00") - Decimal("552.00")  # pay 150 × 3.68
    assert m.tolerance_sentence is not None and m.tolerance_sentence.endswith(
        " 1 other line also exceeds tolerance."
    )
    three = lines + [
        MatchLineInput.of(3, "Cream", "1", "40.00", "40.00", po_unit_price="30.00", po_qty="1", grn_qty="1")
    ]
    m3 = match_invoice(three, invoice_date=INVOICE_DATE)
    assert m3.tolerance_sentence is not None and m3.tolerance_sentence.endswith(
        " 2 other lines also exceed tolerance."
    )


def test_favorable_line_and_qty_only_pairing() -> None:
    lines = [
        MatchLineInput.of(1, "Cod", "10", "9.00", "90.00", po_unit_price="10.00", po_qty="10", grn_qty="10"),
        MatchLineInput.of(2, "Ice", "2", "5.00", "10.00", po_qty="2", grn_qty="2"),
        MatchLineInput.of(3, "Zero", "0", "0", "0", po_unit_price="0", po_qty="0", grn_qty="0"),
    ]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.verdict == "clean" and m.exception_type is None and m.reason is None
    assert m.lines[0].verdict == "favorable" and m.lines[1].verdict == "matched"
    assert m.amount_at_risk == 0 and m.failing_lines == ()


def test_price_fail_with_zero_expected_reason() -> None:
    lines = [MatchLineInput.of(1, "Sample", "1", "5.00", "5.00", po_unit_price="0", po_qty="1", grn_qty="1")]
    m = match_invoice(lines, invoice_date=INVOICE_DATE)
    assert m.reason == "sample over PO"


def test_template_helpers() -> None:
    assert through_text(None, None) == "with no end date"
    assert through_text(date(2026, 9, 30), None) == "through September"
    b = contract_price_bullet(Decimal("3.68"), None, None, Decimal("4.12"), None)
    assert (
        b.text == "The agreement on file fixes this item at $3.68 with no end date; the invoice bills $4.12."
    )
    assert prior_short_pays_bullet([]) is None


cents_st = st.integers(min_value=1, max_value=500_000).map(lambda n: Decimal(n).scaleb(-2))


@given(cents_st, st.integers(1, 500), cents_st, st.integers(0, 600), st.integers(0, 5))
def test_property_pay_plus_hold_equals_total(
    unit: Decimal, qty: int, amount: Decimal, grn: int, prior: int
) -> None:
    line = MatchLineInput.of(
        1,
        "Item",
        qty,
        unit,
        amount,
        po_unit_price=unit,
        po_qty=qty,
        grn_qty=grn,
        contract_prices=[SPEC_CONTRACT],
    )
    m = match_invoice([line], invoice_date=INVOICE_DATE)
    assert m.amount_at_risk >= 0
    assert m.amount_at_risk <= amount
    rec = recommend(m, invoice_total=amount, prior_short_pays=["1.00"] * prior)
    if rec is not None and rec.pay_amount is not None and rec.hold_amount is not None:
        assert rec.pay_amount + rec.hold_amount == amount
        assert rec.confidence <= Decimal("0.95")
