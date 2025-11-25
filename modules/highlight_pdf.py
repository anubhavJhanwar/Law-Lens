import fitz
import re
from difflib import SequenceMatcher


def normalize(text: str) -> str:
    """Normalize text for matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy_ratio(a: str, b: str) -> float:
    """Fuzzy similarity between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_overlap_stats(a: str, b: str) -> tuple[float, int]:
    """
    Compute token overlap ratio and shared token count between two strings.

    Ratio is:
        |A ∩ B| / min(|A|, |B|)
    where A, B are sets of normalized tokens.
    """
    na = normalize(a)
    nb = normalize(b)

    if not na or not nb:
        return 0.0, 0

    ta = set(na.split())
    tb = set(nb.split())

    if not ta or not tb:
        return 0.0, 0

    inter = ta & tb
    shared = len(inter)
    ratio = shared / min(len(ta), len(tb))
    return ratio, shared


def _pick_color(importance: str):
    """Map importance label to highlight color."""
    if importance == "high":
        return (1, 0, 0)  # red - high priority (score 3)
    if importance == "medium":
        return (1, 1, 0)  # yellow - medium priority (score 2)
    # Low importance (score 1) returns None - no highlight
    return None


def highlight_paragraphs_in_original_pdf(
    input_pdf_path, paragraph_data, output_path: str = "highlighted_output.pdf"
):
    """
    Highlight important paragraphs in the original PDF using bounding boxes.

    Matching is robust across different PDF layouts by combining:
    - token overlap (word-level) similarity
    - fuzzy (SequenceMatcher) similarity for shorter blocks
    """

    doc = fitz.open(input_pdf_path)

    # Pre-normalise paragraph text once
    prepared_items = []
    for item in paragraph_data:
        para_text = item.get("paragraph", "")
        importance = item.get("importance", "low")
        color = _pick_color(importance)

        # All paragraphs get highlighted now (high=red, low=yellow)
        if color is None:
            continue

        norm_para = normalize(para_text)
        if not norm_para:
            continue

        prepared_items.append(
            {
                "paragraph": para_text,
                "norm_paragraph": norm_para,
                "importance": importance,
                "color": color,
            }
        )

    for page in doc:
        blocks = page.get_text("blocks") or []  # (x1, y1, x2, y2, text, block_no)

        # If there are no text blocks at all, this page is likely image-only (scanned PDF)
        # Without text bounding boxes, we cannot reliably highlight on this page.
        if not blocks:
            continue

        for block in blocks:
            bx1, by1, bx2, by2, block_text, *_ = block
            if not block_text or not str(block_text).strip():
                continue

            norm_block = normalize(block_text)
            if not norm_block:
                continue

            # Try to match this block against all important paragraphs
            for prepared in prepared_items:
                norm_para = prepared["norm_paragraph"]
                color = prepared["color"]

                # 1) Token overlap (robust to line breaks / partial matches)
                overlap_ratio, shared_tokens = token_overlap_stats(norm_para, norm_block)

                # 2) Fuzzy ratio (works well when block ~ whole paragraph)
                fuzzy_sim = fuzzy_ratio(norm_para, norm_block)

                # Heuristic thresholds tuned for legal PDFs:
                # - require at least 5 shared tokens to avoid spurious matches
                # - accept if either token overlap OR fuzzy similarity is high enough
                if shared_tokens >= 5 and (overlap_ratio >= 0.5 or fuzzy_sim >= 0.6):
                    rect = fitz.Rect(bx1, by1, bx2, by2)
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=color, fill=color)
                    annot.update()

    doc.save(output_path)
    doc.close()

    return output_path

