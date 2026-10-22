from dataclasses import dataclass
from pathlib import Path

import pytest

from ocr_extractor.conversion import split_pages
from ocr_extractor.pdfgen import text_pdf
from ocr_extractor.triage import TriageLimits, looks_empty, read_pages, triage
from pdfs import FULL_PAGE, image_pdf

LIMITS = TriageLimits()

# A real report's worth of visible text: well past IMAGE_PAGE_MIN_CHARS
REPORT_LINES = [
    "Laboratorio Central - Informe de resultados",
    "Paciente: Juan Perez   Fecha de toma: 12/03/2026",
    "Hemoglobina 14.2 g/dL (13.0 - 17.0)",
    "Hematocrito 42.1 % (40.0 - 52.0)",
    "Leucocitos 6.8 10^3/uL (4.0 - 10.0)",
    "Plaquetas 250 10^3/uL (150 - 400)",
    "Glucosa 92 mg/dL (70 - 110)",
    "Colesterol total 180 mg/dL (< 200)",
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
    assert "Hemoglobina 14.2 g/dL" in page.text
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
    assert any("Hematocrito" in text for text in checked)


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
        ["Firmado digitalmente por Dr. Perez - 12/03/2026"],
        ["Pagina 1 de 2", "Laboratorio Central - Tel 555-1234 - www.labcentral.com"],
    ],
)
def test_scan_with_a_digital_stamp_goes_to_the_ocr_lane(tmp_path: Path, stamp: list[str]):
    """A scanned page with a short visible stamp or footer: the stamp's text isn't the report."""
    path = write(tmp_path, "stamped.pdf", image_pdf(stamp, image_box=FULL_PAGE))
    [page] = read_pages(path)
    assert page.visible_text_objects > 0 and page.chars >= LIMITS.min_page_chars
    assert page.image_coverage == pytest.approx(1.0)

    result = triage(path, never_called, LIMITS)
    assert result.lane == "ocr"
    assert result.reason == "page 1 is mostly an image"


def test_report_over_a_full_page_letterhead_stays_in_the_text_lane(tmp_path: Path):
    path = write(tmp_path, "letterhead.pdf", image_pdf(REPORT_LINES, image_box=FULL_PAGE))
    [page] = read_pages(path)
    assert page.chars >= LIMITS.image_page_min_chars

    assert triage(path, all_good, LIMITS).lane == "text"


def test_short_report_with_a_small_logo_stays_in_the_text_lane(tmp_path: Path):
    path = write(tmp_path, "logo.pdf", image_pdf(REPORT_LINES[:3], image_box=(450, 700, 120, 60)))
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
    assert looks_empty("Laboratorio Central\n\nInforme de resultados y observaciones")
    assert not looks_empty("| Hemoglobina | 14.2 | g/dL | 13.0 - 17.0 |")


def test_split_pages_reads_markers_page_breaks():
    separator = "-" * 48
    markdown = f"{{0}}{separator}\n\nHemoglobina 14.2\n\n{{1}}{separator}\n\n\n\n{{2}}{separator}\n\nGlucosa 92"
    assert split_pages(markdown) == {0: "Hemoglobina 14.2", 1: "", 2: "Glucosa 92"}
