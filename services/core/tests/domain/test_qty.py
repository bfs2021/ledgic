from decimal import Decimal

from app.domain.qty import qty_check


def test_all_equal_passes_with_note() -> None:
    r = qty_check("12", "12", "12")
    assert r.passes and r.verdict == "matched"
    assert r.note == "ordered 12 · received 12 · invoiced 12"
    assert r.payable_qty == Decimal(12)


def test_over_receipt() -> None:
    r = qty_check("12", "10", "12")
    assert r.over_receipt and not r.over_order and r.verdict == "qty_over_receipt"
    assert r.payable_qty == Decimal(10)


def test_over_order_with_and_without_allowance() -> None:
    r = qty_check("13", "13", "12")
    assert r.over_order and not r.over_receipt and r.verdict == "qty_over_order"
    assert r.payable_qty == Decimal(12)
    assert qty_check("13", "13", "12", over_receipt_allowance="0.10").passes


def test_both_failures_report_over_receipt_first() -> None:
    r = qty_check("15", "10", "12")
    assert r.over_receipt and r.over_order and r.verdict == "qty_over_receipt"
    assert r.payable_qty == Decimal(10)


def test_two_way_skips_grn() -> None:
    r = qty_check("12", None, "12", match_mode="two_way")
    assert r.passes and r.received is None
    assert r.note == "ordered 12 · invoiced 12 · two-way, no receipt check"
    assert not qty_check("13", None, "12", match_mode="two_way").passes


def test_three_way_without_grn_means_nothing_received() -> None:
    r = qty_check("5", None, "5")
    assert r.over_receipt and r.received == 0
    assert r.note == "ordered 5 · no receipt · invoiced 5"


def test_no_po_qty_only_checks_grn() -> None:
    r = qty_check("5", "5", None)
    assert r.passes and r.note == "received 5 · invoiced 5"


def test_rebilling_an_already_invoiced_receipt_is_over_receipt() -> None:
    # 12 received, another invoice already billed all 12: a second bill for 12 must not pass.
    r = qty_check("12", "12", "12", prior_invoiced="12")
    assert r.over_receipt and r.over_order and r.verdict == "qty_over_receipt"
    assert r.payable_qty == Decimal(0)
    assert r.note == "ordered 12 · received 12 · already invoiced 12 · invoiced 12"


def test_partial_receipts_across_invoices() -> None:
    # PO for 20; 12 received and 8 already billed: 4 may still be billed.
    assert qty_check("4", "12", "20", prior_invoiced="8").passes
    r = qty_check("6", "12", "20", prior_invoiced="8")
    assert r.over_receipt and not r.over_order and r.payable_qty == Decimal(4)


def test_split_order_remaining_quantity_two_way() -> None:
    # Two-way vendors skip the receipt, but the PO quantity is still consumed.
    assert qty_check("5", None, "10", match_mode="two_way", prior_invoiced="5").passes
    r = qty_check("6", None, "10", match_mode="two_way", prior_invoiced="5")
    assert r.over_order and r.payable_qty == Decimal(5)
    assert r.note == "ordered 10 · already invoiced 5 · invoiced 6 · two-way, no receipt check"


def test_no_prior_keeps_the_original_note() -> None:
    assert qty_check("12", "12", "12", prior_invoiced="0").note == "ordered 12 · received 12 · invoiced 12"
