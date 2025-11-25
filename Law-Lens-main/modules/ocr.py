import os
import pytesseract  # type: ignore[import]
import logging
from pdf2image import convert_from_path, pdfinfo_from_path  # type: ignore[import]
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv  # type: ignore[import]
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TESSERACT_PATH = os.environ.get("TESSERACT_PATH")
POPPLER_BIN_PATH = os.environ.get("POPPLER_BIN_PATH")

BATCH_SIZE = 20
NUM_CORES = os.cpu_count() or 4

# Check if Tesseract is available (optional - only needed for scanned PDFs)
TESSERACT_AVAILABLE = False
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    TESSERACT_AVAILABLE = True
    logging.info(f"✓ Tesseract found at: {TESSERACT_PATH}")
else:
    logging.warning(f"⚠ Tesseract not configured. OCR for scanned PDFs will not be available.")

# Check if Poppler is available (optional - only needed for scanned PDFs)
POPPLER_AVAILABLE = False
if POPPLER_BIN_PATH and os.path.exists(POPPLER_BIN_PATH):
    POPPLER_AVAILABLE = True
    logging.info(f"✓ Poppler found at: {POPPLER_BIN_PATH}")
else:
    logging.warning(f"⚠ Poppler not configured. OCR for scanned PDFs will not be available.")
    
def _ocr_page(image):
    try:
        # Optimized Tesseract config for speed
        config = "-l eng --oem 1 --psm 3 -c tessedit_do_invert=0"
        
        # Resize large images to reduce processing time
        width, height = image.size
        if width > 2000 or height > 2000:
            scale = min(2000/width, 2000/height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, resample=1)  # LANCZOS
        
        return pytesseract.image_to_string(image, config=config)
    except Exception as e:
        logging.error(f"Error processing page: {e}")
        return ""

def extract_text_from_scanned_pdf(pdf_path, dpi=200):
    # Check if OCR tools are available
    if not TESSERACT_AVAILABLE or not POPPLER_AVAILABLE:
        logging.error("❌ OCR not available. Please install Tesseract and Poppler.")
        return ""
    
    logging.info(f"Starting OCR for: {pdf_path}")
    
    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_BIN_PATH)
        total_pages = info["Pages"]
        logging.info(f"PDF has {total_pages} pages. Processing in batches of {BATCH_SIZE}.")
    except Exception as e:
        logging.error(f"❌ Could not get PDF info: {e}")
        return ""

    all_page_texts = []
    
    for start_page in range(1, total_pages + 1, BATCH_SIZE):
        end_page = min(start_page + BATCH_SIZE - 1, total_pages)
        logging.info(f"Processing pages {start_page} to {end_page}...")
        
        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=POPPLER_BIN_PATH,
                first_page=start_page,
                last_page=end_page,
                thread_count=NUM_CORES
            )
            
            if not images:
                logging.warning(f"No images extracted for pages {start_page}-{end_page}")
                continue

            with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
                batch_texts = list(executor.map(_ocr_page, images))
            
            all_page_texts.extend(batch_texts)
            
        except Exception as e:
            logging.error(f"❌ Failed to process batch {start_page}-{end_page}: {e}")
            continue

    logging.info(f"✅ Successfully processed {len(all_page_texts)} pages from {pdf_path}.")
    
    full_text = "\n".join(all_page_texts)
    return full_text.strip()

if __name__ == "__main__":
    
    NUM_CORES = os.cpu_count() or 4
    
    pdf_file = "scanned_document.pdf"
    
    if os.path.exists(pdf_file):
        print(f"--- Extracting text from {pdf_file} (using {NUM_CORES} cores) ---")
        extracted_text = extract_text_from_scanned_pdf(pdf_file)
        print("\n--- Extracted Text (first 500 chars) ---")
        print(extracted_text[:500] + "...")
    else:
        print(f"Warning: '{pdf_file}' not found.")
        print("Please add a PDF to this directory and rename it to run the example.")