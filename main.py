"""
PDF Translator Backend — Optimized for Render deployment
=========================================================
Requirements:
    pip install pymupdf fastapi uvicorn python-multipart deep-translator

Usage (local):
    uvicorn main:app --port 8000
"""

import io
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

LANG_MAP = {
    "japanese": "ja",
    "korean":   "ko",
    "spanish":  "es",
    "auto":     "auto",
}


# ── Fast text extraction using PyMuPDF rawdict ────────────────────────────────

def extract_blocks(pdf_bytes: bytes) -> list[list[dict]]:
    """
    Extract text blocks per page using PyMuPDF rawdict mode.
    rawdict reads character-level Unicode — handles CJK fonts correctly.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_pages = []

    for page in doc:
        page_blocks = []
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue

            chars_text = ""
            for line in block.get("lines", []):
                line_str = ""
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        c = ch.get("c", "")
                        if c:
                            line_str += c
                if line_str.strip():
                    chars_text += line_str.strip() + "\n"

            chars_text = chars_text.strip()
            if not chars_text:
                continue

            x0, y0, x1, y1 = block["bbox"]
            page_blocks.append({
                "text": chars_text,
                "x0": x0, "y0": y0,
                "x1": x1, "y1": y1,
            })

        all_pages.append(page_blocks)

    doc.close()
    return all_pages


# ── Fast batch translation via Google Translate ───────────────────────────────

def translate_all(texts: list[str], src_lang: str) -> list[str]:
    """
    Batch all text into as few Google Translate calls as possible.
    Uses a separator trick so we can join + split in one round trip.
    Typical time: 1-3 seconds for an entire document.
    """
    if not texts:
        return []

    SEP = "\n§§§\n"
    MAX_CHARS = 4000
    results = list(texts)  # default to originals

    try:
        translator = GoogleTranslator(source=src_lang, target="en")
    except Exception as e:
        print(f"Translator init error: {e}")
        return results

    # Build batches
    batches = []       # list of (joined_text, [original_indices])
    current_texts = []
    current_indices = []
    current_len = 0

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

    print(f"  Translating {len(texts)} blocks in {len(batches)} batch(es)...")

    for joined, indices in batches:
        try:
            translated = translator.translate(joined)
            if not translated:
                continue
            # Split back — Google sometimes merges/drops separators so we do a best-effort split
            parts = translated.split("§§§")
            parts = [p.strip() for p in parts if p.strip()]
            for i, part in enumerate(parts):
                if i < len(indices):
                    results[indices[i]] = part
        except Exception as e:
            print(f"  Batch error: {e}")

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
    src_lang = LANG_MAP.get(source_lang.lower(), "auto")

    all_page_blocks = extract_blocks(pdf_bytes)

    if not all_page_blocks or not any(all_page_blocks):
        raise HTTPException(status_code=400,
            detail="No text found. If this is a scanned PDF, run OCR first at ilovepdf.com/ocr-pdf")

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
    return {"status": "ok"}

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
