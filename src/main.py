import os
import tempfile
import requests
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse
import logging
os.environ["PDFTEXT_CPU_WORKERS"] = "1"

import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import io

from fastapi import FastAPI
from marker.convert import convert_single_pdf
from marker.models import load_all_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_data = {}

@asynccontextmanager
async def lifespan(_):
    app_data["models"] = load_all_models()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
"""
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    url: Annotated[
        Optional[str],
        Field(description="The URL to the PDF file to convert.")
    ] = None
    filepath: Annotated[
        Optional[str],
        Field(description="The path to the PDF file to convert.")
    ] = None
    max_pages: Annotated[
        Optional[int],
        Field(description="The maximum number of pages in the document to convert.", example=None)
    ] = None
    langs: Annotated[
        Optional[str],
        Field(description="The optional languages to use if OCR is needed, comma separated.  Must be either the names or codes from from https://github.com/VikParuchuri/surya/blob/master/surya/languages.py.", example=None)
    ] = None
    force_ocr: Annotated[
        bool,
        Field(description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases).")
    ] = False
    paginate: Annotated[
        bool,
        Field(description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines).")
    ] = False
    extract_images: Annotated[
        bool,
        Field(description="Whether to extract images from the PDF.  Defaults to True.  If set to False, no images will be extracted from the PDF.")
    ] = True


@app.post("/marker")
async def convert_pdf(
    params: CommonParams
):
    
    assert all([
        params.extract_images is True,
        params.paginate is False,
    ]), "Local conversion API does not support image extraction or pagination."

    if params.filepath:
        print(f"Converting {params.filepath} locally.")
        return await convert_pdf_local(params)
    elif params.url:
        print(f"Converting {params.url} using pdf remote.")
        return await convert_pdf_remote(params)
    else:
        raise ValueError("No filepath or url provided.")


async def convert_pdf_remote(params: CommonParams):
    try:
       # Download file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            response = requests.get(params.url, stream=True, timeout=30)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file.flush()
            
            # Create new params with local filepath
            local_params = CommonParams(
                filepath=tmp_file.name,
                max_pages=params.max_pages,
                langs=params.langs,
                force_ocr=params.force_ocr,
                paginate=params.paginate,
                extract_images=params.extract_images
            )
            
            # Use local conversion
            result = await convert_pdf_local(local_params)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return result
            
    except (requests.RequestException, IOError) as e:
        logger.error("Error in convert_pdf_remote: %s", str(e))
        return {
            "success": False,
            "error": str(e)
        }


async def convert_pdf_local(params: CommonParams):
    try:
        # log the params
        logger.info("Converting PDF with parameters: %s", {
            "filepath": params.filepath,
            "max_pages": params.max_pages,
            "langs": params.langs,
            "force_ocr": params.force_ocr,
            "paginate": params.paginate,
            "extract_images": params.extract_images
        })
        
        full_text, images, metadata = convert_single_pdf(
            params.filepath,
            app_data["models"],
            max_pages=params.max_pages,
            langs=params.langs.split(",") if params.langs else None,  # Convert string to list
            ocr_all_pages=params.force_ocr
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format="PNG")
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode("utf-8")

    return {
        "markdown": full_text,
        "images": encoded,
        "metadata": metadata,
        "success": True
    }