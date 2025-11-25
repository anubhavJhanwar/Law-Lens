import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
import os
try:
    import pypdfium2 as pdfium
    USE_PDFIUM = True
except ImportError:
    from PyPDF2 import PdfReader
    USE_PDFIUM = False

from modules.ocr import extract_text_from_scanned_pdf


# ============================================================
#                PDF TEXT EXTRACTION (OPTIMIZED)
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """Extract PDF text with optimized parallel processing; fallback to OCR if needed."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    try:
        if USE_PDFIUM:
            text = _extract_with_pdfium(temp_pdf_path)
        else:
            text = _extract_with_pypdf2(temp_pdf_path)

        # Fallback to OCR if text is empty or too short
        if not text.strip() or len(text.strip()) < 50:
            text = extract_text_from_scanned_pdf(temp_pdf_path)

    except Exception:
        text = extract_text_from_scanned_pdf(temp_pdf_path)
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_pdf_path)
        except:
            pass

    return clean_extracted_text(text)


def _extract_with_pdfium(pdf_path):
    """Fast extraction using pypdfium2 (3-5x faster than PyPDF2)."""
    pdf = pdfium.PdfDocument(pdf_path)
    page_texts = []
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        if text:
            page_texts.append(text)
    
    return "\n".join(page_texts)


def _extract_with_pypdf2(pdf_path):
    """Fallback extraction using PyPDF2 with parallel processing."""
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    # Use parallel processing for large PDFs
    if num_pages > 10:
        max_workers = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            page_texts = list(executor.map(lambda p: p.extract_text() or "", reader.pages))
    else:
        page_texts = [page.extract_text() or "" for page in reader.pages]
    
    return "\n".join(page_texts)


def clean_extracted_text(text):
    """Clean extracted PDF text before analysis (optimized with compiled regex)."""
    if not text:
        return ""

    # Compile regex patterns once for better performance
    control_chars = re.compile(r'[\x00-\x1F\x7F-\x9F]')
    spaces_tabs = re.compile(r'[ \t]+')
    multi_newlines = re.compile(r'\n\s*\n\s*\n+')
    hyphen_newline = re.compile(r'(\w+)-\s*\n\s*(\w+)')
    extra_newlines = re.compile(r'\n+')
    multi_spaces = re.compile(r'\s+')

    # Apply all regex replacements in sequence
    text = control_chars.sub('', text)
    text = spaces_tabs.sub(' ', text)
    text = multi_newlines.sub('\n\n', text)
    text = hyphen_newline.sub(r'\1\2', text)
    text = extra_newlines.sub('\n', text)

    # Remove page numbers and isolated digits (optimized)
    lines = [line.strip() for line in text.split("\n") 
             if not (line.strip().isdigit() and len(line.strip()) <= 3)]
    
    text = "\n".join(lines)
    text = multi_spaces.sub(' ', text)

    return text.strip()


# ============================================================
#                   PARAGRAPH PROCESSING
# ============================================================

def split_into_paragraphs(text):
    """
    Split text into meaningful legal paragraphs.
    Avoid splitting on '1.', '2.', '3.' etc.
    """
    # First split on double newlines
    candidates = re.split(r"\n\s*\n", text)

    # If still too long, split by period but only when followed by uppercase letter
    final_paragraphs = []
    for chunk in candidates:
        parts = re.split(r'\.\s+(?=[A-Z])', chunk)
        for p in parts:
            p = p.strip()
            if len(p) > 40:  # ignore noise
                final_paragraphs.append(p)

    return final_paragraphs


def compute_keyword_density(paragraph, keywords):
    """Count keyword occurrences using whole-word regex search."""
    if not paragraph or not keywords:
        return 0

    density = 0

    for kw in keywords:
        if not kw:
            continue

        pattern = r'\b' + re.escape(kw) + r'\b'
        hits = len(re.findall(pattern, paragraph, flags=re.IGNORECASE))
        density += hits

    return density


def classify_importance_by_density(density, high_threshold=2):
    """Classify paragraph importance based on keyword density."""
    return "high" if density >= high_threshold else "low"


def prepare_paragraph_importance_data(text, keywords, high_threshold=2):
    """
    Returns list of:
    {
        "paragraph": "...",
        "density": int,
        "importance": "high" or "low"
    }
    """
    paragraphs = split_into_paragraphs(text)
    results = []

    for para in paragraphs:
        density = compute_keyword_density(para, keywords)
        importance = classify_importance_by_density(density, high_threshold)

        results.append({
            "paragraph": para,
            "density": density,
            "importance": importance
        })

    return results
