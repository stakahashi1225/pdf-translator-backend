"""
PDF Translator Backend — Google Vision OCR + Google Translate
=============================================================
Requirements:
    pip install pymupdf fastapi uvicorn python-multipart deep-translator requests
"""

import io
import os
import base64
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from deep_translator import GoogleTranslator

app = FastAPI(title="PDF Translator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_URL = "https://vision.googleapis.com/v1/images:annotate"

LANG_MAP = {
    "japanese": "ja",
    "korean":   "ko",
    "spanish":  "es",
    "auto":     "auto",
}


# ── Convert PDF page to image ─────────────────────────────────────────────────

def pdf_page_to_image(page: fitz.Page, dpi: int = 200) -> bytes:
    """Render a PDF page to a PNG image at given DPI."""
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ── Google Vision OCR ─────────────────────────────────────────────────────────

def ocr_image(image_bytes: bytes, lang_hint: str = "ja") -> list[dict]:
    """
    Send image to Google Cloud Vision API for OCR.
    Returns list of text blocks with bounding boxes.
    """
    if not VISION_API_KEY:
        raise HTTPException(status_code=500,
            detail="GOOGLE_VISION_API_KEY not set in environment variables.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION", "maxResults": 1}],
            "imageContext": {"languageHints": [lang_hint]}
        }]
    }

    resp = requests.post(
        f"{VISION_URL}?key={VISION_API_KEY}",
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502,
            detail=f"Google Vision error: {resp.status_code} {resp.text}")

    data = resp.json()
    responses = data.get("responses", [{}])
    if not responses or "error" in responses[0]:
        err = responses[0].get("error", {}).get("message", "Unknown error")
        raise HTTPException(status_code=502, detail=f"Vision API error: {err}")

    full_annotation = responses[0].get("fullTextAnnotation")
    if not full_annotation:
        return []

    # Extract paragraphs with bounding boxes
    blocks = []
    pages = full_annotation.get("pages", [])
    for page in pages:
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                # Build text from words
                para_text = ""
                for word in para.get("words", []):
                    word_text = "".join(
                        s.get("text", "") for s in word.get("symbols", [])
                    )
                    para_text += word_text

                para_text = para_text.strip()
                if not para_text:
                    continue

                # Get bounding box (normalized vertices)
                verts = para.get("boundingBox", {}).get("vertices", [])
                if len(verts) < 4:
                    continue

                xs = [v.get("x", 0) for v in verts]
                ys = [v.get("y", 0) for v in verts]
                blocks.append({
                    "text": para_text,
                    "x0": min(xs), "y0": min(ys),
                    "x1": max(xs), "y1": max(ys),
                })

    return blocks


# ── Batch translation ─────────────────────────────────────────────────────────

def translate_all(texts: list[str], src_lang: str) -> list[str]:
    """Batch translate using Google Translate — fast, 1-2 calls total."""
    if not texts:
        return []

    SEP = "\n§§§\n"
    MAX_CHARS = 4000
    results = list(texts)

    try:
        translator = GoogleTranslator(source=src_lang, target="en")
    except Exception as e:
        print(f"Translator init error: {e}")
        return results

    batches = []
    current_texts, current_indices, current_len = [], [], 0

    for i, text in enumerate(texts):
        t = text.strip()
        if not t:
            continue
        if current_len + len(t) > MAX_CHARS and current_texts:
            batches.append((SEP.join(current_texts), current_indices[:]))
            current_texts.clear()
            current_indices.clear()
            current_len = 0
        current_texts.append(t)
        current_indices.append(i)
        current_len += len(t) + len(SEP)

    if current_texts:
        batches.append((SEP.join(current_texts), current_indices[:]))

    print(f"Translating {len(texts)} blocks in {len(batches)} batch(es)...")

    for joined, indices in batches:
        try:
            translated = translator.translate(joined)
            if not translated:
                continue
            parts = [p.strip() for p in translated.split("§§§") if p.strip()]
            for i, part in enumerate(parts):
                if i < len(indices):
                    results[indices[i]] = part
        except Exception as e:
            print(f"Batch error: {e}")

    return results


# ── Build translated PDF ──────────────────────────────────────────────────────

def build_translated_pdf(
    original_bytes: bytes,
    all_page_data: list[list[dict]],
    page_scales: list[tuple]
) -> bytes:
    """
    Overlay English translations onto the original PDF.
    page_scales: list of (scale_x, scale_y) to convert Vision pixel coords → PDF coords.
    """
    doc = fitz.open(stream=original_bytes, filetype="pdf")

    for page_idx, blocks in enumerate(all_page_data):
        if page_idx >= len(doc):
            break
        page = doc[page_idx]
        sx, sy = page_scales[page_idx]

        for block in blocks:
            translated = block.get("translated_text", "").strip()
            if not translated:
                continue

            # Convert pixel coords to PDF coords
            x0 = block["x0"] / sx
            y0 = block["y0"] / sy
            x1 = block["x1"] / sx
            y1 = block["y1"] / sy

            rect = fitz.Rect(x0, y0, x1, y1)
            if rect.width < 5 or rect.height < 5:
                continue

            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
            fontsize = max(6, min(rect.height * 0.72, 10))
            page.insert_textbox(
                rect, translated,
                fontsize=fontsize, fontname="helv",
                color=(0, 0, 0), align=0,
            )

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ── Shared pipeline ───────────────────────────────────────────────────────────

async def run_pipeline(file: UploadFile, source_lang: str):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    pdf_bytes = await file.read()
    src_lang = LANG_MAP.get(source_lang.lower(), "ja")
    lang_hint = src_lang if src_lang != "auto" else "ja"

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_page_blocks = []
    page_scales = []

    print(f"Processing {doc.page_count} page(s) with Google Vision OCR...")

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pdf_w = page.rect.width
        pdf_h = page.rect.height

        # Render page to image
        img_bytes = pdf_page_to_image(page, dpi=200)

        # OCR with Google Vision
        blocks = ocr_image(img_bytes, lang_hint)
        print(f"  Page {page_num+1}: {len(blocks)} text blocks found")

        # Calculate scale factors (image pixels → PDF points)
        # At 200 DPI: pixels = points * (200/72)
        scale = 200 / 72
        page_scales.append((scale, scale))
        all_page_blocks.append(blocks)

    doc.close()

    if not any(all_page_blocks):
        raise HTTPException(status_code=400,
            detail="No text found in PDF. Please check the file.")

    # Translate all blocks
    all_texts = [b["text"] for page in all_page_blocks for b in page]
    translated_flat = translate_all(all_texts, src_lang)

    idx = 0
    for page_blocks in all_page_blocks:
        for block in page_blocks:
            block["translated_text"] = translated_flat[idx] if idx < len(translated_flat) else ""
            idx += 1

    return pdf_bytes, all_page_blocks, page_scales


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PDF Translator API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vision_key_set": bool(VISION_API_KEY),
    }

@app.post("/preview")
async def preview_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("japanese"),
):
    pdf_bytes, all_page_blocks, page_scales = await run_pipeline(file, source_lang)
    result_pages = []
    for page_blocks in all_page_blocks:
        result_pages.append([{
            "original":   b["text"],
            "translated": b["translated_text"],
            "x0": b["x0"], "y0": b["y0"],
            "x1": b["x1"], "y1": b["y1"],
        } for b in page_blocks])
    return JSONResponse({"pages": result_pages})

@app.post("/translate")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("japanese"),
):
    pdf_bytes, all_page_blocks, page_scales = await run_pipeline(file, source_lang)
    translated_pdf = build_translated_pdf(pdf_bytes, all_page_blocks, page_scales)
    filename = file.filename.replace(".pdf", "_translated.pdf")
    return StreamingResponse(
        io.BytesIO(translated_pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
