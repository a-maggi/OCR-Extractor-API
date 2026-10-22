import os
import tempfile
import requests
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse
import logging

from contextlib import asynccontextmanager
from typing import Optional, Annotated

from fastapi import FastAPI
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_data = {}

@asynccontextmanager
async def lifespan(_):
    app_data["models"] = create_model_dict()

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
    output_format: Annotated[
        str,
        Field(description="The format of the output.  Defaults to 'markdown'.  Can be 'markdown', 'json', or 'html'.")
    ] = "json"


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
        return await convert_pdf_local(params, params.filepath)
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
            
            # Use local conversion
            result = await convert_pdf_local(params, tmp_file.name)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return result
            
    except (requests.RequestException, IOError) as e:
        logger.error("Error in convert_pdf_remote: %s", str(e))
        return {
            "success": False,
            "error": str(e)
        }


async def convert_pdf_local(params: CommonParams, filepath: str):
    try:
        # log the params
        logger.info("Converting PDF with parameters: %s", {
            "filepath": params.filepath,
            "max_pages": params.max_pages,
            "langs": params.langs,
            "force_ocr": params.force_ocr,
            "paginate": params.paginate,
            "extract_images": params.extract_images,
            "output_format": params.output_format
        })
        params_dict = params.model_dump()
        config_parser = ConfigParser(params_dict)
        
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )
        
        result = converter(filepath)
        if params.output_format == "markdown":
            logger.info("Converting to markdown, result type: %s", type(result))
            # Remove the line that tries to access .keys()
            if hasattr(result, "markdown"):
                result = result.markdown
            elif hasattr(result, "text"):
                result = result.text
            elif isinstance(result, str):
                pass  # Result is already a string
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")
        
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

    return {
        "output": result,
        "success": True
    }