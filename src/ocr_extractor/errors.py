"""Errors a worker raises for the parent to act on. Plain exceptions, so they pickle across processes."""


class OcrServerUnavailable(Exception):
    """The OCR model server stopped answering during a conversion, so pages may be missing. Worth one retry."""


class NoReadableText(Exception):
    """OCR ran but found no readable content in the document."""
