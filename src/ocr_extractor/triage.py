"""Decides whether a PDF's own text layer is good enough, or the document needs OCR.

Runs inside a text-lane worker, before conversion. It never calls the OCR model, so it stays cheap on CPU.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

Lane = Literal["text", "ocr"]

INVISIBLE_RENDER_MODES = {pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE, pdfium_c.FPDF_TEXTRENDERMODE_UNKNOWN}

# A number keeps its decimal and thousands separators ("1.234,5"); a word is a run of letters
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")

# A word hyphenated across lines ("hemo-" then "globina"), and the tags Marker writes for exponents
LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])-[ \t]*\r?\n[ \t]*(?=[^\W\d_A-Z])")
EXPONENT_TAG = re.compile(r"</?su[bp]>")


class OcrErrorResult(Protocol):
    labels: Sequence[str]


# Marker's OCR-error model (surya OCRErrorPredictor): returns "good"/"bad" per text
OcrErrorModel = Callable[[list[str]], OcrErrorResult]


@dataclass(frozen=True)
class TriageLimits:
    # Fewer non-whitespace characters than this, on a page with images, means the page is an image
    min_page_chars: int = 20
    # A page whose images cover at least this share of it (Marker uses the same 0.65 to skip a page's text)...
    image_coverage: float = 0.65
    # ...needs at least this much visible text to count as digital. Less is a scan with a digital stamp or footer
    image_page_min_chars: int = 200
    # Output without a single digit counts as empty, for documents that always contain numbers
    require_digits: bool = False
    # Text-lane output that kept less than this share of the text layer's words and numbers is lossy
    min_text_coverage: float = 0.9


@dataclass
class PageText:
    index: int
    text: str
    visible_text_objects: int
    invisible_text_objects: int
    image_objects: int
    # Share of the page covered by images, 0 to 1
    image_coverage: float

    @property
    def chars(self) -> int:
        return len("".join(self.text.split()))

    @property
    def blank(self) -> bool:
        return not (self.visible_text_objects or self.invisible_text_objects or self.image_objects)


@dataclass
class TriageResult:
    lane: Lane
    reason: str
    pages: int
    # The PDF carries a scanner's invisible OCR layer: OCR should replace it rather than trust it
    replace_ocr_layer: bool = False
    # Every page's text layer, for checking what the conversion kept
    text: str = ""


def _image_coverage(page: pdfium.PdfPage, images: list[tuple[float, float, float, float]]) -> float:
    width, height = page.get_size()
    if width <= 0 or height <= 0:
        return 0.0
    covered = 0.0
    for left, bottom, right, top in images:
        # Clip to the page; overlapping images can add up past 1, so cap the total
        covered += max(0.0, min(right, width) - max(left, 0.0)) * max(0.0, min(top, height) - max(bottom, 0.0))
    return min(1.0, covered / (width * height))


def read_pages(path: Path) -> list[PageText]:
    pdf = pdfium.PdfDocument(str(path))
    try:
        pages = []
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            try:
                visible = invisible = 0
                images: list[tuple[float, float, float, float]] = []
                for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE]):
                    if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                        images.append(obj.get_bounds())
                    elif pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw) in INVISIBLE_RENDER_MODES:
                        invisible += 1
                    else:
                        visible += 1
                pages.append(
                    PageText(
                        index=index,
                        text=textpage.get_text_bounded(),
                        visible_text_objects=visible,
                        invisible_text_objects=invisible,
                        image_objects=len(images),
                        image_coverage=_image_coverage(page, images),
                    )
                )
            finally:
                textpage.close()
                page.close()
        return pages
    finally:
        pdf.close()


def _ocr_reason(page: PageText, limits: TriageLimits) -> str | None:
    """Why this page needs OCR, or None if its text layer may be good."""
    number = page.index + 1
    if page.visible_text_objects == 0:
        # An image-only page is a scan; invisible-only text is the OCR layer a scanner adds on top of one
        return f"page {number} has no visible text"
    if page.invisible_text_objects > page.visible_text_objects:
        return f"page {number} is mostly a scanner's OCR layer"
    if page.image_coverage >= limits.image_coverage and page.chars < limits.image_page_min_chars:
        # A scan with a digital stamp, signature or footer on top: the visible text isn't the document
        return f"page {number} is mostly an image"
    if page.chars < limits.min_page_chars and page.image_objects:
        return f"page {number} is mostly an image"
    return None


def triage(path: Path, ocr_error_model: OcrErrorModel, limits: TriageLimits) -> TriageResult:
    pages = read_pages(path)
    content = [page for page in pages if not page.blank]

    reasons = [reason for page in content if (reason := _ocr_reason(page, limits))]
    if reasons:
        has_scanner_layer = any(page.invisible_text_objects for page in content)
        return TriageResult("ocr", reasons[0], len(pages), replace_ocr_layer=has_scanner_layer)

    # Near-empty digital pages are too short to judge
    to_check = [page for page in content if page.chars >= limits.min_page_chars]
    if to_check:
        # Catches broken font encodings: text that extracts as letters but isn't the words on the page
        labels = ocr_error_model([page.text for page in to_check]).labels
        bad = [str(page.index + 1) for page, label in zip(to_check, labels) if label == "bad"]
        if bad:
            return TriageResult("ocr", f"garbled text on page {', '.join(bad)}", len(pages))

    return TriageResult("text", "text layer looks good", len(pages), text="\n".join(page.text for page in pages))


def looks_empty(markdown: str, require_digits: bool = False) -> bool:
    """Nearly empty output means the text layer didn't hold the real content. With require_digits, so does output
    without any number: for documents that always have numbers, that text layer is a stamp or a header."""
    content = "".join(markdown.split())
    if len(content) < 10:
        return True
    return require_digits and not any(char.isdigit() for char in content)


def _tokens(text: str) -> Counter[str]:
    text = unicodedata.normalize("NFKC", EXPONENT_TAG.sub("", text))
    text = "".join(char for char in text if char in "\t\n\r" or unicodedata.category(char) not in ("Cc", "Cf"))
    text = unicodedata.normalize("NFD", LINE_BREAK_HYPHEN.sub("", text))
    plain = "".join(char for char in text if not unicodedata.combining(char))
    return Counter(TOKEN.findall(plain.casefold()))


def text_coverage(text_layer: str, markdown: str) -> float:
    """The share of the text layer's words and numbers that the markdown kept, counting repeats. 1 when the text
    layer has none."""
    wanted = _tokens(text_layer)
    total = wanted.total()
    if not total:
        return 1.0
    return sum((wanted & _tokens(markdown)).values()) / total
