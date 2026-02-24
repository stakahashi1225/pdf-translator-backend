"""
PDF Translator Backend
======================
Requirements:
    py -m pip install pymupdf fastapi uvicorn python-multipart httpx pdfminer.six deep-translator

Usage:
    py -m uvicorn main:app --reload --port 8000
"""

import io
import time
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTAnon, LTChar

app = FastAPI(title="PDF Translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LANG_MAP = {
    "japanese": "ja",
    "korean":   "ko",
    "spanish":  "es",
    "auto":     "auto",
}


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_blocks_pdfminer(pdf_bytes: bytes) -> list[list[dict]]:
    all_pages = []
    try:
        for page_layout in extract_pages(io.BytesIO(pdf_bytes)):
            page_blocks = []
            page_height = page_layout.height
            for element in page_layout:
                if not isinstance(element, LTTextBox):
                    continue
                full_text = ""
                for line in element:
                    if isinstance(line, LTTextLine):
                        line_text = "".join(
                            ch.get_text() for ch in line
                            if isinstance(ch, (LTChar, LTAnon))
                        ).strip()
                        if line_text:
                            full_text += line_text + "\n"
                full_text = full_text.strip()
                if not full_text:
                    continue
                x0, y0_pdf, x1, y1_pdf = element.bbox
                page_blocks.append({
                    "text": full_text,
                    "x0": x0,
                    "y0": page_height - y1_pdf,
                    "x1": x1,
                    "y1": page_height - y0_pdf,
                })
            all_pages.append(page_blocks)
    except Exception as e:
        print(f"pdfminer error: {e}")
    return all_pages


# ── Translation (deep-translator → Google Translate, fast & free) ─────────────

def translate_all(texts: list[str], src_lang: str) -> list[str]:
    """
    Translate all texts in one batch using deep-translator (Google Translate).
    Splits into chunks of 5000 chars max per request.
    Much faster than MyMemory — no per-block delays.
    """
    if not texts:
        return []

    translator = GoogleTranslator(source=src_lang, target="en")
    results = [""] * len(texts)

    # Batch texts together with a separator to minimize API calls
    SEPARATOR = "\n||||\n"
    MAX_CHARS = 4500

    # Group texts into batches
    batch = []
    batch_indices = []
    batch_len = 0

    def flush_batch():
        if not batch:
            return
        joined = SEPARATOR.join(batch)
        try:
            translated = translator.translate(joined)
            if translated:
                parts = translated.split("||||")
                for i, part in enumerate(parts):
                    if i < len(batch_indices):
                        results[batch_indices[i]] = part.strip()
            else:
                # fallback: copy original
                for i in batch_indices:
                    results[i] = texts[i]
        except Exception as e:
            print(f"Batch translation error: {e}")
            for i in batch_indices:
                results[i] = texts[i]

    for idx, text in enumerate(texts):
        text = text.strip()
        if not text:
            results[idx] = ""
            continue

        if batch_len + len(text) > MAX_CHARS and batch:
            flush_batch()
            batch.clear()
            batch_indices.clear()
            batch_len = 0

        batch.append(text)
        batch_indices.append(idx)
        batch_len += len(text) + len(SEPARATOR)

    flush_batch()

    print(f"  Translated {len(texts)} blocks via Google Translate")
    return results


# ── PDF rendering ─────────────────────────────────────────────────────────────

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
            page.insert_textbox(rect, translated, fontsize=fontsize,
                                fontname="helv", color=(0, 0, 0), align=0)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ── Shared processing ─────────────────────────────────────────────────────────

async def process_pdf(file: UploadFile, source_lang: str):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    pdf_bytes = await file.read()
    src_lang = LANG_MAP.get(source_lang.lower(), "auto")

    print(f"Extracting text...")
    all_page_blocks = extract_blocks_pdfminer(pdf_bytes)

    if not all_page_blocks or not any(all_page_blocks):
        raise HTTPException(status_code=400,
            detail="No text found. If scanned, run OCR first at ilovepdf.com/ocr-pdf")

    all_texts = [b["text"] for page in all_page_blocks for b in page]
    print(f"Translating {len(all_texts)} blocks in batch...")

    translated_flat = translate_all(all_texts, src_lang)

    idx = 0
    for page_blocks in all_page_blocks:
        for block in page_blocks:
            block["translated_text"] = translated_flat[idx] if idx < len(translated_flat) else ""
            idx += 1

    return pdf_bytes, all_page_blocks


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "translator": "Google Translate (deep-translator)"}


@app.post("/preview")
async def preview_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("japanese"),
):
    pdf_bytes, all_page_blocks = await process_pdf(file, source_lang)
    result_pages = []
    for page_blocks in all_page_blocks:
        result_pages.append([{
            "original": b["text"],
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
    pdf_bytes, all_page_blocks = await process_pdf(file, source_lang)
    translated_pdf = build_translated_pdf(pdf_bytes, all_page_blocks)
    filename = file.filename.replace(".pdf", "_translated.pdf")
    return StreamingResponse(
        io.BytesIO(translated_pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
