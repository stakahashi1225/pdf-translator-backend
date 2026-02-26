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

DPI = 200
SCALE = 72.0 / DPI  # pixels → PDF points


# ── Convert PDF page to image ─────────────────────────────────────────────────

def pdf_page_to_image(page: fitz.Page) -> bytes:
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ── Google Vision OCR ─────────────────────────────────────────────────────────

def ocr_image(image_bytes: bytes, lang_hint: str = "ja") -> list[dict]:
    if not VISION_API_KEY:
        raise HTTPException(status_code=500,
            detail="GOOGLE_VISION_API_KEY not set.")

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
        json=payload, timeout=30,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502,
            detail=f"Google Vision error: {resp.status_code} {resp.text}")

    data = resp.json()
    responses = data.get("responses", [{}])
    if not responses or "error" in responses[0]:
        err = responses[0].get("error", {}).get("message", "Unknown")
        raise HTTPException(status_code=502, detail=f"Vision API error: {err}")

    full_annotation = responses[0].get("fullTextAnnotation")
    if not full_annotation:
        return []

    blocks = []
    for page in full_annotation.get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                para_text = ""
                for word in para.get("words", []):
                    para_text += "".join(
                        s.get("text", "") for s in word.get("symbols", [])
                    )

                para_text = para_text.strip()
                if not para_text:
                    continue

                verts = para.get("boundingBox", {}).get("vertices", [])
                if len(verts) < 4:
                    continue

                xs = [v.get("x", 0) for v in verts]
                ys = [v.get("y", 0) for v in verts]

                # Convert pixel coords → PDF points
                blocks.append({
                    "text":  para_text,
                    "x0":    min(xs) * SCALE,
                    "y0":    min(ys) * SCALE,
                    "x1":    max(xs) * SCALE,
                    "y1":    max(ys) * SCALE,
                })

    return blocks


# ── Batch translation ─────────────────────────────────────────────────────────

def translate_all(texts: list[str], src_lang: str) -> list[str]:
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
    cur_texts, cur_idx, cur_len = [], [], 0

    for i, text in enumerate(texts):
        t = text.strip()
        if not t:
            continue
        if cur_len + len(t) > MAX_CHARS and cur_texts:
            batches.append((SEP.join(cur_texts), cur_idx[:]))
            cur_texts.clear(); cur_idx.clear(); cur_len = 0
        cur_texts.append(t); cur_idx.append(i)
        cur_len += len(t) + len(SEP)

    if cur_texts:
        batches.append((SEP.join(cur_texts), cur_idx[:]))

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

def build_translated_pdf(original_bytes: bytes, all_page_data: list[list[dict]]) -> bytes:
    doc = fitz.open(stream=original_bytes, filetype="pdf")

    for page_idx, blocks in enumerate(all_page_data):
        if page_idx >= len(doc):
            break
        page = doc[page_idx]

        for block in blocks:
            translated = block.get("translated_text", "").strip()
            if not translated:
                continue

            rect = fitz.Rect(block["x0"], block["y0"], block["x1"], block["y1"])

            # Skip tiny or invalid rects
            if rect.width < 4 or rect.height < 4 or rect.is_empty or rect.is_infinite:
                continue

            # White out original text
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

            # Fit font size to box height
            fontsize = max(5, min(rect.height * 0.65, 11))
            page.insert_textbox(
                rect, translated,
                fontsize=fontsize,
                fontname="helv",
                color=(0.1, 0.1, 0.6),  # dark blue so it's easy to spot
                align=0,
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

    print(f"Processing {doc.page_count} page(s) with Google Vision OCR...")

    for page_num in range(doc.page_count):
        page = doc[page_num]
        img_bytes = pdf_page_to_image(page)
        blocks = ocr_image(img_bytes, lang_hint)
        print(f"  Page {page_num+1}: {len(blocks)} blocks, page size {page.rect.width:.0f}x{page.rect.height:.0f}pt")
        all_page_blocks.append(blocks)

    doc.close()

    if not any(all_page_blocks):
        raise HTTPException(status_code=400, detail="No text found in PDF.")

    all_texts = [b["text"] for page in all_page_blocks for b in page]
    translated_flat = translate_all(all_texts, src_lang)

    idx = 0
    for page_blocks in all_page_blocks:
        for block in page_blocks:
            block["translated_text"] = translated_flat[idx] if idx < len(translated_flat) else ""
            idx += 1

    return pdf_bytes, all_page_blocks


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PDF Translator API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "vision_key_set": bool(VISION_API_KEY)}

@app.post("/preview")
async def preview_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("japanese"),
):
    pdf_bytes, all_page_blocks = await run_pipeline(file, source_lang)
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
    pdf_bytes, all_page_blocks = await run_pipeline(file, source_lang)
    translated_pdf = build_translated_pdf(pdf_bytes, all_page_blocks)
    filename = file.filename.replace(".pdf", "_translated.pdf")
    return StreamingResponse(
        io.BytesIO(translated_pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
