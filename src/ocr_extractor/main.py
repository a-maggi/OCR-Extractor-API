import logging

from ocr_extractor.api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs every request URL at INFO, and a signed or capability URL is enough to download the document
for name in ("httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)

app = create_app()
