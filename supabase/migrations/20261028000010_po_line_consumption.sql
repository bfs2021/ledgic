-- Match consumes PO-line quantity already billed by other invoices (#32) (2026-10-28).
-- payable_qty: the quantity a line could be paid for at match time, so a short-paid invoice consumes only
-- what it paid and the vendor can still bill the balance. Null on rows written before this migration.
alter table public.match_line_results add column if not exists payable_qty numeric(12,3);
create index if not exists invoice_lines_po_line_id_idx on public.invoice_lines (po_line_id);
