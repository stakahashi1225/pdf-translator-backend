"""
PDF Translator Backend — Google Vision OCR + In-Place English Replacement
=========================================================================
Keeps original layout, replaces Japanese text with English in same position.
Requirements:
    pip install pymupdf fastapi uvicorn python-multipart deep-translator requests
"""

import io
import os
import base64
import requests
import fitz
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from deep_translator import GoogleTranslator

app = FastAPI(title="PDF Translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
LANG_MAP = {"japanese": "ja", "korean": "ko", "spanish": "es", "auto": "auto"}

DPI = 150  # Lower DPI = smaller image = faster, still accurate enough


def page_to_image(page: fitz.Page) -> tuple[bytes, float, float]:
    """Render page to image, return bytes + scale factors."""
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    # scale_x and scale_y convert image pixels back to PDF points
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    return pix.tobytes("png"), scale_x, scale_y


def ocr_page(image_bytes: bytes, lang_hint: str, scale_x: float, scale_y: float) -> list[dict]:
    """OCR a page image, return blocks with PDF-point coordinates."""
    if not VISION_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"requests": [{"image": {"content": b64},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        "imageContext": {"languageHints": [lang_hint]}}]}

    resp = requests.post(f"{VISION_URL}?key={VISION_API_KEY}", json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Vision error: {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    response = data.get("responses", [{}])[0]
    if "error" in response:
        raise HTTPException(status_code=502, detail=f"Vision error: {response['error']['message']}")

    ann = response.get("fullTextAnnotation")
    if not ann:
        return []

    blocks = []
    for pg in ann.get("pages", []):
        for block in pg.get("blocks", []):
            verts = block.get("boundingBox", {}).get("vertices", [])
            if len(verts) < 4:
                continue

            text = ""
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    text += "".join(s.get("text", "") for s in word.get("symbols", []))
                    text += " "
            text = text.strip()
            if not text:
                continue

            # Convert image pixel coords → PDF points using scale factors
            xs = [v.get("x", 0) * scale_x for v in verts]
            ys = [v.get("y", 0) * scale_y for v in verts]

            blocks.append({
                "text": text,
                "x0": min(xs), "y0": min(ys),
                "x1": max(xs), "y1": max(ys),
            })

    print(f"    Vision found {len(blocks)} blocks")
    return blocks


def translate_all(texts: list[str], src_lang: str) -> list[str]:
    if not texts:
        return []
    SEP = "\n§§§\n"
    MAX_CHARS = 4000
    results = list(texts)
    try:
        translator = GoogleTranslator(source=src_lang, target="en")
    except Exception as e:
        return results

    batches, cur_t, cur_i, cur_l = [], [], [], 0
    for i, text in enumerate(texts):
        t = text.strip()
        if not t:
            continue
        if cur_l + len(t) > MAX_CHARS and cur_t:
            batches.append((SEP.join(cur_t), cur_i[:]))
            cur_t.clear(); cur_i.clear(); cur_l = 0
        cur_t.append(t); cur_i.append(i); cur_l += len(t) + len(SEP)
    if cur_t:
        batches.append((SEP.join(cur_t), cur_i[:]))

    print(f"  Translating {len(texts)} blocks in {len(batches)} batch(es)...")
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


def build_translated_pdf(original_bytes: bytes, all_page_blocks: list[list[dict]]) -> bytes:
    """
    Opens original PDF, for each text block:
    1. Draws a white rectangle to cover the Japanese text
    2. Inserts English translation in the same rectangle
    Result: same layout, same structure, English where Japanese was.
    """
    doc = fitz.open(stream=original_bytes, filetype="pdf")

    for page_idx, blocks in enumerate(all_page_blocks):
        if page_idx >= len(doc):
            break
        page = doc[page_idx]

        for block in blocks:
            translated = block.get("translated_text", "").strip()
            if not translated:
                continue

            x0, y0, x1, y1 = block["x0"], block["y0"], block["x1"], block["y1"]

            # Clamp to page bounds
            pw, ph = page.rect.width, page.rect.height
            x0 = max(0, min(x0, pw))
            y0 = max(0, min(y0, ph))
            x1 = max(x0 + 1, min(x1, pw))
            y1 = max(y0 + 1, min(y1, ph))

            rect = fitz.Rect(x0, y0, x1, y1)
            if rect.is_empty or rect.width < 3 or rect.height < 3:
                continue

            # Step 1: white out Japanese text
            page.draw_rect(rect, color=(1,1,1), fill=(1,1,1))

            # Step 2: insert English text in same space
            fontsize = max(5, min(rect.height * 0.75, 11))
            overflow = page.insert_textbox(
                rect, translated,
                fontsize=fontsize,
                fontname="helv",
                color=(0, 0, 0),
                align=0,
            )
            # If text overflows (negative return), try smaller font
            if overflow < 0:
                page.insert_textbox(
                    rect, translated,
                    fontsize=max(4, fontsize * 0.7),
                    fontname="helv",
                    color=(0, 0, 0),
                    align=0,
                )

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


async def run_pipeline(file: UploadFile, source_lang: str):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    pdf_bytes = await file.read()
    src_lang = LANG_MAP.get(source_lang.lower(), "ja")
    lang_hint = src_lang if src_lang != "auto" else "ja"

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_page_blocks = []

    print(f"Processing {doc.page_count} page(s)...")
    for page_num in range(doc.page_count):
        page = doc[page_num]
        img_bytes, scale_x, scale_y = page_to_image(page)
        print(f"  Page {page_num+1}: scale=({scale_x:.3f}, {scale_y:.3f})")
        blocks = ocr_page(img_bytes, lang_hint, scale_x, scale_y)
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


@app.get("/")
def root():
    return {"status": "PDF Translator API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "vision_key_set": bool(VISION_API_KEY)}

@app.post("/preview")
async def preview_pdf(file: UploadFile = File(...), source_lang: str = Form("japanese")):
    pdf_bytes, all_page_blocks = await run_pipeline(file, source_lang)
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
async def translate_pdf(file: UploadFile = File(...), source_lang: str = Form("japanese")):
    pdf_bytes, all_page_blocks = await run_pipeline(file, source_lang)
    output_pdf = build_translated_pdf(pdf_bytes, all_page_blocks)
    out_name = file.filename.replace(".pdf", "_translated.pdf")
    return StreamingResponse(
        io.BytesIO(output_pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
