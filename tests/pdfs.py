"""One-page test PDFs mixing an image with visible text, to exercise the image-coverage rules."""

import zlib

from ocr_extractor.pdfgen import _escape

PAGE_WIDTH, PAGE_HEIGHT = 612, 792


def image_pdf(lines: list[str], *, image_box: tuple[float, float, float, float] | None) -> bytes:
    """Draws a gray image at image_box (x, y, width, height) under the given lines of visible text."""
    operators = []
    if image_box:
        x, y, width, height = image_box
        operators.append(f"q {width} 0 0 {height} {x} {y} cm /Im1 Do Q")
    if lines:
        operators += ["BT", "/F1 11 Tf", "0 Tr", "14 TL", "72 720 Td"]
        operators += [f"({_escape(line)}) Tj T*" for line in lines]
        operators.append("ET")
    content = "\n".join(operators).encode("latin-1")

    pixels = zlib.compress(bytes([180]) * 64)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d]" % (PAGE_WIDTH, PAGE_HEIGHT)
        + b" /Resources << /Font << /F1 4 0 R >> /XObject << /Im1 6 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width 8 /Height 8 /ColorSpace /DeviceGray /BitsPerComponent 8"
        b" /Filter /FlateDecode /Length %d >>\nstream\n" % len(pixels)
        + pixels
        + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


FULL_PAGE = (0, 0, PAGE_WIDTH, PAGE_HEIGHT)
