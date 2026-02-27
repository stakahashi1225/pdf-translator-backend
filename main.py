"""
PDF Translator Backend — Google Vision OCR + Two-Column Translation Report
==========================================================================
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
DPI = 200
PX_TO_PT = 72.0 / DPI


# ── Render PDF page to image ──────────────────────────────────────────────────

def page_to_image(page: fitz.Page) -> bytes:
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    return page.get_pixmap(matrix=mat).tobytes("png")


# ── Google Vision OCR ─────────────────────────────────────────────────────────

def ocr_page(image_bytes: bytes, lang_hint: str = "ja") -> list[dict]:
    if not VISION_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_VISION_API_KEY not set.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"requests": [{"image": {"content": b64},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        "imageContext": {"languageHints": [lang_hint]}}]}

    resp = requests.post(f"{VISION_URL}?key={VISION_API_KEY}", json=payload, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Vision error: {resp.status_code}")

    data = resp.json()
    ann = data.get("responses", [{}])[0].get("fullTextAnnotation")
    if not ann:
        return []

    blocks = []
    for pg in ann.get("pages", []):
        for block in pg.get("blocks", []):
            text = ""
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    text += "".join(s.get("text","") for s in word.get("symbols",[])) + " "
            text = text.strip()
            if text:
                verts = block.get("boundingBox", {}).get("vertices", [])
                ys = [v.get("y", 0) for v in verts]
                blocks.append({"text": text, "y": min(ys) * PX_TO_PT})

    # Sort top to bottom
    blocks.sort(key=lambda b: b["y"])
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


# ── Build two-column PDF report ───────────────────────────────────────────────

def build_report_pdf(
    original_bytes: bytes,
    all_page_blocks: list[list[dict]],
    original_filename: str,
) -> bytes:
    """
    Page 1: Original PDF page as full image
    Page 2+: Two-column table — Japanese left, English right, row by row
    Easy to read, no coordinate issues.
    """
    orig_doc = fitz.open(stream=original_bytes, filetype="pdf")
    new_doc = fitz.open()

    # ── First: include original pages as images ───────────────────────────────
    for page_num in range(orig_doc.page_count):
        orig_page = orig_doc[page_num]
        orig_w = orig_page.rect.width
        orig_h = orig_page.rect.height
        img_bytes = page_to_image(orig_page)

        # Original page
        orig_out = new_doc.new_page(width=orig_w, height=orig_h)
        orig_out.insert_image(fitz.Rect(0, 0, orig_w, orig_h), stream=img_bytes)

        # Label
        orig_out.insert_textbox(
            fitz.Rect(0, 0, orig_w, 14),
            "ORIGINAL",
            fontsize=8, fontname="helv",
            color=(0.5, 0.5, 0.5), align=1,
        )

    orig_doc.close()

    # ── Then: translation report pages ───────────────────────────────────────
    PAGE_W, PAGE_H = 595, 842  # A4
    MARGIN = 30
    COL_W = (PAGE_W - MARGIN * 3) / 2
    ROW_H = 22
    HEADER_H = 50
    FONT_SIZE = 8

    def new_report_page():
        pg = new_doc.new_page(width=PAGE_W, height=PAGE_H)
        # Header
        pg.draw_rect(fitz.Rect(0, 0, PAGE_W, HEADER_H),
                     color=(0.2, 0.3, 0.6), fill=(0.2, 0.3, 0.6))
        pg.insert_textbox(fitz.Rect(MARGIN, 10, PAGE_W - MARGIN, 35),
            f"Translation: {original_filename}",
            fontsize=11, fontname="helv", color=(1,1,1), align=0)
        pg.insert_textbox(fitz.Rect(MARGIN, 32, PAGE_W//2, 48),
            "JAPANESE (Original)", fontsize=8, fontname="helv", color=(0.8,0.9,1), align=0)
        pg.insert_textbox(fitz.Rect(PAGE_W//2 + MARGIN//2, 32, PAGE_W - MARGIN, 48),
            "ENGLISH (Translation)", fontsize=8, fontname="helv", color=(0.8,1,0.8), align=0)
        # Divider
        pg.draw_line(fitz.Point(PAGE_W//2, HEADER_H),
                     fitz.Point(PAGE_W//2, PAGE_H - MARGIN),
                     color=(0.7, 0.7, 0.7), width=0.5)
        return pg, HEADER_H + 8

    current_page, y = new_report_page()

    for page_idx, blocks in enumerate(all_page_blocks):
        if not blocks:
            continue

        # Page separator
        if y > HEADER_H + 8:
            current_page.draw_line(
                fitz.Point(MARGIN, y), fitz.Point(PAGE_W - MARGIN, y),
                color=(0.3, 0.5, 0.8), width=1)
            y += 6
            current_page.insert_textbox(
                fitz.Rect(MARGIN, y, PAGE_W - MARGIN, y + 14),
                f"— Page {page_idx + 1} —",
                fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4), align=1)
            y += 16

        for block in blocks:
            original = block.get("text", "").strip()
            translated = block.get("translated_text", "").strip()
            if not original:
                continue

            # Estimate lines needed
            chars_per_line = int(COL_W / (FONT_SIZE * 0.55))
            lines = max(
                len(original) // max(chars_per_line, 1) + 1,
                len(translated) // max(chars_per_line, 1) + 1,
                1
            )
            row_height = max(ROW_H, lines * (FONT_SIZE + 3) + 6)

            # New page if needed
            if y + row_height > PAGE_H - MARGIN:
                current_page, y = new_report_page()

            # Alternating row background
            row_rect = fitz.Rect(MARGIN, y, PAGE_W - MARGIN, y + row_height)
            if (all_page_blocks[0].index(block) if block in all_page_blocks[0] else 0) % 2 == 0:
                current_page.draw_rect(row_rect, color=(0.97,0.97,0.97), fill=(0.97,0.97,0.97))

            # Japanese text (left)
            jp_rect = fitz.Rect(MARGIN + 2, y + 2, MARGIN + COL_W - 2, y + row_height - 2)
            current_page.insert_textbox(jp_rect, original,
                fontsize=FONT_SIZE, fontname="helv", color=(0.1,0.1,0.1), align=0)

            # English text (right, blue)
            en_rect = fitz.Rect(PAGE_W//2 + 4, y + 2, PAGE_W - MARGIN - 2, y + row_height - 2)
            current_page.insert_textbox(en_rect, translated,
                fontsize=FONT_SIZE, fontname="helv", color=(0.0, 0.15, 0.7), align=0)

            # Row divider
            current_page.draw_line(
                fitz.Point(MARGIN, y + row_height),
                fitz.Point(PAGE_W - MARGIN, y + row_height),
                color=(0.88, 0.88, 0.88), width=0.3)

            y += row_height

    buf = io.BytesIO()
    new_doc.save(buf)
    new_doc.close()
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

    for page_num in range(doc.page_count):
        page = doc[page_num]
        img_bytes = page_to_image(page)
        blocks = ocr_page(img_bytes, lang_hint)
        print(f"  Page {page_num+1}: {len(blocks)} blocks")
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

    return pdf_bytes, all_page_blocks, file.filename


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PDF Translator API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "vision_key_set": bool(VISION_API_KEY)}

@app.post("/preview")
async def preview_pdf(file: UploadFile = File(...), source_lang: str = Form("japanese")):
    pdf_bytes, all_page_blocks, filename = await run_pipeline(file, source_lang)
    result_pages = []
    for page_blocks in all_page_blocks:
        result_pages.append([{
            "original": b["text"],
            "translated": b["translated_text"],
        } for b in page_blocks])
    return JSONResponse({"pages": result_pages})

@app.post("/translate")
async def translate_pdf(file: UploadFile = File(...), source_lang: str = Form("japanese")):
    pdf_bytes, all_page_blocks, filename = await run_pipeline(file, source_lang)
    output_pdf = build_report_pdf(pdf_bytes, all_page_blocks, filename)
    out_name = filename.replace(".pdf", "_translated.pdf")
    return StreamingResponse(
        io.BytesIO(output_pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
