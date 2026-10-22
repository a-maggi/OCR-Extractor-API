"""Builds tiny one-page PDFs with a real text layer: the boot warmup and the tests use them."""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(lines: list[str], *, invisible: bool = False) -> bytes:
    """invisible=True draws the text in render mode 3, like the OCR layer a scanner adds."""
    operators = ["BT", "/F1 12 Tf", "3 Tr" if invisible else "0 Tr", "14 TL", "72 720 Td"]
    operators += [f"({_escape(line)}) Tj T*" for line in lines]
    operators.append("ET")
    stream = "\n".join(operators).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
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


WARMUP_LINES = [
    "Warmup invoice 2026-0001",
    "Paper A4 80 g/m2 10 x 4.20 = 42.00",
    "Toner cartridge 2 x 38.50 = 77.00",
    "Total 119.00 (VAT included)",
]
