from dataclasses import dataclass
from pathlib import Path

import pytest

from ocr_extractor.conversion import split_pages
from ocr_extractor.pdfgen import text_pdf
from ocr_extractor.triage import TriageLimits, looks_empty, read_pages, text_coverage, triage
from pdfs import FULL_PAGE, image_pdf

LIMITS = TriageLimits()

# A real document's worth of visible text: well past IMAGE_PAGE_MIN_CHARS
DOCUMENT_LINES = [
    "Northwind Supplies Ltd - Invoice 2026-0042",
    "Billed to: Example Corporation   Issued: 12/03/2026",
    "Paper A4 80 g/m2, box of 500 sheets   10 x 4.20 = 42.00",
    "Toner cartridge, black, high yield   2 x 38.50 = 77.00",
    "Stapler, heavy duty   1 x 12.90 = 12.90",
    "Shipping and handling   1 x 9.00 = 9.00",
    "Subtotal 140.90   VAT 21% 29.59",
    "Total due 170.49 within 30 days",
]


@dataclass
class FakeOcrErrorResult:
    labels: list[str]


def all_good(texts: list[str]) -> FakeOcrErrorResult:
    return FakeOcrErrorResult(["good"] * len(texts))


def all_bad(texts: list[str]) -> FakeOcrErrorResult:
    return FakeOcrErrorResult(["bad"] * len(texts))


def never_called(texts: list[str]) -> FakeOcrErrorResult:
    raise AssertionError("pages that need OCR shouldn't reach the OCR-error model")


def write(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_reads_the_text_layer_of_a_digital_pdf(digital_pdf: Path):
    [page] = read_pages(digital_pdf)
    assert "Toner cartridge 2 x 38.50" in page.text
    assert page.visible_text_objects > 0
    assert page.invisible_text_objects == 0
    assert page.image_objects == 0
    assert page.image_coverage == 0


def test_digital_pdf_goes_to_the_text_lane(digital_pdf: Path):
    checked: list[str] = []

    def model(texts: list[str]) -> FakeOcrErrorResult:
        checked.extend(texts)
        return all_good(texts)

    result = triage(digital_pdf, model, LIMITS)
    assert result.lane == "text"
    assert result.pages == 1
    # The page text went through the OCR-error model
    assert any("Toner cartridge" in text for text in checked)


def test_scanned_pdf_goes_to_the_ocr_lane_without_calling_the_model(scanned_pdf: Path):
    result = triage(scanned_pdf, never_called, LIMITS)
    assert result.lane == "ocr"
    assert "no visible text" in result.reason
    assert not result.replace_ocr_layer


def test_scanner_ocr_layer_goes_to_the_ocr_lane_and_is_replaced(ocr_layer_pdf: Path):
    [page] = read_pages(ocr_layer_pdf)
    assert page.visible_text_objects == 0 and page.invisible_text_objects > 0

    result = triage(ocr_layer_pdf, never_called, LIMITS)
    assert result.lane == "ocr"
    # Marker would otherwise keep the scanner's invisible text instead of OCRing the page
    assert result.replace_ocr_layer


@pytest.mark.parametrize(
    "stamp",
    [
        ["Digitally signed by J. Smith - 12/03/2026"],
        ["Page 1 of 2", "Northwind Supplies - Tel 555-1234 - www.example.com"],
    ],
)
def test_scan_with_a_digital_stamp_goes_to_the_ocr_lane(tmp_path: Path, stamp: list[str]):
    """A scanned page with a short visible stamp or footer: the stamp's text isn't the document."""
    path = write(tmp_path, "stamped.pdf", image_pdf(stamp, image_box=FULL_PAGE))
    [page] = read_pages(path)
    assert page.visible_text_objects > 0 and page.chars >= LIMITS.min_page_chars
    assert page.image_coverage == pytest.approx(1.0)

    result = triage(path, never_called, LIMITS)
    assert result.lane == "ocr"
    assert result.reason == "page 1 is mostly an image"


def test_document_over_a_full_page_letterhead_stays_in_the_text_lane(tmp_path: Path):
    path = write(tmp_path, "letterhead.pdf", image_pdf(DOCUMENT_LINES, image_box=FULL_PAGE))
    [page] = read_pages(path)
    assert page.chars >= LIMITS.image_page_min_chars

    assert triage(path, all_good, LIMITS).lane == "text"


def test_short_document_with_a_small_logo_stays_in_the_text_lane(tmp_path: Path):
    path = write(tmp_path, "logo.pdf", image_pdf(DOCUMENT_LINES[:3], image_box=(450, 700, 120, 60)))
    [page] = read_pages(path)
    assert page.image_objects == 1 and page.image_coverage < 0.05

    assert triage(path, all_good, LIMITS).lane == "text"


def test_garbled_text_layer_goes_to_the_ocr_lane(digital_pdf: Path):
    result = triage(digital_pdf, all_bad, LIMITS)
    assert result.lane == "ocr"
    assert result.reason == "garbled text on page 1"


def test_blank_pdf_stays_in_the_text_lane(tmp_path: Path):
    path = write(tmp_path, "blank.pdf", text_pdf([]))
    assert triage(path, all_bad, LIMITS).lane == "text"


def test_looks_empty():
    assert looks_empty("")
    assert looks_empty("  Total \n\n ")
    assert not looks_empty("Dear customer, thank you for your order")
    assert not looks_empty("| Paper A4 | 10 | 4.20 | 42.00 |")


def test_looks_empty_with_required_digits():
    assert looks_empty("Northwind Supplies\n\nTerms and conditions", require_digits=True)
    assert not looks_empty("| Paper A4 | 10 | 4.20 | 42.00 |", require_digits=True)


def test_text_coverage_is_the_share_of_the_text_layers_tokens_the_markdown_kept():
    layer = "Monocitos 0,5 10 mg/dL"
    assert text_coverage(layer, "| Monocitos | 0,5 | 10 | mg/dL |") == 1
    assert text_coverage(layer, "| 0,5 | 10 | mg/dL |") == 0.8
    assert text_coverage(layer, "") == 0


def test_text_coverage_ignores_case_accents_and_markdown():
    assert text_coverage("Ácido ÚRICO plasmático", "**Acido urico** plasmatico") == 1


def test_text_coverage_keeps_numbers_whole_with_comma_or_point():
    assert text_coverage("1,5 and 1.5", "1,5 and 1.5") == 1
    assert text_coverage("1,5 2.25", "1 5 2 25") == 0
    assert text_coverage("1.234,5", "1.234,5") == 1


def test_text_coverage_counts_repeats():
    assert text_coverage("mg mg mg mg", "mg") == 0.25


def test_text_coverage_of_a_text_layer_without_tokens_is_complete():
    assert text_coverage("  -- ", "") == 1


def test_a_text_lane_triage_carries_the_text_layer(digital_pdf: Path):
    result = triage(digital_pdf, all_good, LIMITS)
    assert "Toner cartridge 2 x 38.50" in result.text


def test_split_pages_reads_markers_page_breaks():
    separator = "-" * 48
    markdown = f"{{0}}{separator}\n\nPaper A4 42.00\n\n{{1}}{separator}\n\n\n\n{{2}}{separator}\n\nTotal 119.00"
    assert split_pages(markdown) == {0: "Paper A4 42.00", 1: "", 2: "Total 119.00"}
