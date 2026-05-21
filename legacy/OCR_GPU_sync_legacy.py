import datetime
import gc
import io
import logging
import os
import time

import fasttext
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# --- Configuration ---
INITIAL_BATCH_SIZE = 50
RETRY_BATCH_SIZE_1 = 10
RETRY_BATCH_SIZE_2 = 1
DPI = 300
FASTTEXT_MODEL_PATH = "/app/models/lid.176.bin"

# Mapping fastText codes to PaddleOCR codes
# fastText returns labels like '__label__zh'
LANG_MAPPING = {
    'zh': 'ch',
    'en': 'en',
    'fr': 'fr',
    'de': 'german',
    'ja': 'japan',
    'ko': 'korean',
    'es': 'es',
    'it': 'it',
    'pt': 'pt',
    'ru': 'ru',
    'ar': 'ar'
}

# --- Logging & Reporting Setup ---
LOG_DIR = "/app/ocr_output/logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"ocr_processing_{timestamp}.log")
csv_report_file = os.path.join(LOG_DIR, f"ocr_issues_{timestamp}.csv")
final_report_file = os.path.join(LOG_DIR, f"ocr_final_report_{timestamp}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load fastText model
logger.info("Loading fastText language identification model...")
try:
    if os.path.exists(FASTTEXT_MODEL_PATH):
        lang_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logger.info("fastText model loaded successfully.")
    else:
        logger.error(f"fastText model NOT found at {FASTTEXT_MODEL_PATH}. Language detection will fail.")
        lang_model = None
except Exception as e:
    logger.error(f"Error loading fastText model: {e}")
    lang_model = None

# Global Trackers
issues_list = []
doc_reports = [] 

def img_to_bytes(img):
    """Converts a PIL Image to JPEG bytes for valid PDF insertion."""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=85)
    return img_byte_arr.getvalue() 

def log_issue(pdf_path, page_num, issue_type, details):
    issues_list.append({
        "Timestamp": datetime.datetime.now(),
        "File": os.path.basename(pdf_path),
        "Page": page_num,
        "Issue Type": issue_type,
        "Details": details
    })
    logger.warning(f"ISSUE [{issue_type}] File: {os.path.basename(pdf_path)} | Page: {page_num} | {details}")

def get_paddle_instance(lang_code):
    logger.info(f"Loading PaddleOCR model for language: {lang_code}")
    try:
        return PaddleOCR(use_angle_cls=True, lang=lang_code)
    except Exception as e:
        logger.error(f"Failed to load PaddleOCR for {lang_code}, falling back to 'en'. Error: {e}")
        return PaddleOCR(use_angle_cls=True, lang='en')

def detect_language(pdf_path):
    if not lang_model:
        return 'en'
        
    logger.info("Detecting document language (pre-scan)...")
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=3, dpi=72)
        if not images:
            return 'en'
        
        sample_text = ""
        for img in images:
            text = pytesseract.image_to_string(img)
            sample_text += text + " "
            if len(sample_text) > 500:
                break
        
        # Clean text for fastText (remove newlines)
        clean_text = sample_text.replace("\n", " ").strip()
        
        if not clean_text or len(clean_text) < 10:
            logger.warning("No sufficient text found during pre-scan. Defaulting to 'en'.")
            return 'en'

        # Predict
        predictions = lang_model.predict(clean_text, k=1)
        # Format: (('__label__en',), array([0.98]))
        label = predictions[0][0]
        confidence = predictions[1][0]
        
        lang_code = label.replace("__label__", "")
        
        paddle_lang = LANG_MAPPING.get(lang_code, 'en')
        
        logger.info(f"Detected: '{lang_code}' (Conf: {confidence:.2f}) -> Mapped to Paddle: '{paddle_lang}'")
        return paddle_lang
        
    except Exception as e:
        logger.warning(f"Language detection failed ({e}). Defaulting to 'en'.")
        return 'en'

# --- Reporting Functions ---

def record_page_status(file_status_dict, page_num, status):
    file_status_dict['details'][page_num] = status
    if status == "FALLBACK_IMAGE_ONLY":
        file_status_dict['missing'].append(page_num)

def generate_final_report():
    with open(final_report_file, 'w', encoding='utf-8') as f:
        f.write(f"OCR PROCESSING REPORT - {datetime.datetime.now()}\n")
        f.write("="*60 + "\n\n")
        
        total_docs = len(doc_reports)
        total_missing_pages_global = sum(len(d['missing']) for d in doc_reports)
        
        f.write(f"Total Documents Processed: {total_docs}\n")
        f.write(f"Total 'Missing' Pages (OCR Failed): {total_missing_pages_global}\n\n")
        
        for doc in doc_reports:
            f.write("-" * 40 + "\n")
            f.write(f"Document: {doc['file']}\n")
            f.write(f"Detected Language: {doc.get('lang', 'N/A')}\n")
            f.write(f"Total Pages: {doc['total']}\n")
            
            if doc['missing']:
                f.write(f"MISSING PAGES (No OCR Text): {', '.join(map(str, doc['missing']))}\n")
            else:
                f.write("MISSING PAGES: None\n")
                
            f.write("\nPage Index:\n")
            sorted_pages = sorted(doc['details'].keys())
            for p in sorted_pages:
                status = doc['details'][p]
                f.write(f"  Page {p}: {status}\n")
            f.write("\n")

# --- Processing Functions ---

def process_page_with_tesseract(page_num, img, pdf_path):
    try:
        start_time = time.time()
        logger.info(f"    > Attempting Tesseract Failover for Page {page_num}...")
        
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        temp_doc = fitz.open()
        width, height = img.size
        page = temp_doc.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), stream=img_to_bytes(img))
        
        n_boxes = len(data['text'])
        word_count = 0
        has_text = False
        extracted_text_page = ""
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:
                text = data['text'][i]
                if text.strip():
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    page.insert_text((x, y), text, fontsize=12, render_mode=3)
                    extracted_text_page += text + " "
                    word_count += 1
                    has_text = True
        
        elapsed = time.time() - start_time
        if has_text:
            logger.info(f"    > Tesseract Success Page {page_num} | Words: {word_count} | Time: {elapsed:.2f}s")
            return temp_doc, True, extracted_text_page
        else:
            logger.warning(f"    > Tesseract found NO text on Page {page_num}.")
            return None, False, ""

    except Exception as e:
        logger.error(f"    > Tesseract Failed for Page {page_num}: {e}")
        return None, False, ""

def process_single_page_image(page_num, img, pdf_path, ocr_engine):
    try:
        start_time = time.time()
        img_np = np.array(img)
        
        result = ocr_engine.ocr(img_np, cls=True)
        
        temp_doc = fitz.open()
        width, height = img.size
        page = temp_doc.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), stream=img_to_bytes(img))
        
        word_count = 0
        has_text = False
        extracted_text_page = ""
        
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                if confidence < 0.6:
                    log_issue(pdf_path, page_num, "Low Confidence", f"Text: '{text[:15]}...' Conf: {confidence:.2f}")

                box = line[0] 
                x, y = box[0]
                page.insert_text((x, y), text, fontsize=12, render_mode=3) 
                extracted_text_page += text + " "
                word_count += 1
                has_text = True
        
        extracted_text_page += "\n"
        
        elapsed = time.time() - start_time
        logger.info(f"    Page {page_num} processed (Paddle) | Words: {word_count} | Time: {elapsed:.2f}s")
        return temp_doc, True, extracted_text_page

    except Exception as e:
        logger.error(f"    Error processing Page {page_num} (Paddle): {e}")
        return None, False, ""

def create_fallback_page(img):
    try:
        temp_doc = fitz.open()
        width, height = img.size
        page = temp_doc.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), stream=img_to_bytes(img))
        return temp_doc
    except Exception:
        return None

def process_chunk_images(images, start_page, pdf_path, current_file_status, ocr_engine):
    page_docs = []
    chunk_statuses = {} 
    chunk_text = ""
    
    for i, img in enumerate(images):
        current_page = start_page + i
        temp_doc, success, page_text = process_single_page_image(current_page, img, pdf_path, ocr_engine)
        
        if success and temp_doc:
            page_docs.append(temp_doc)
            chunk_statuses[current_page] = "PaddleOCR"
            chunk_text += page_text + "\n"
        else:
            for d in page_docs:
                d.close()
            return None, None, ""
            
    return page_docs, chunk_statuses, chunk_text

def attempt_chunk_processing(pdf_path, start_page, end_page, current_batch_size, current_file_status, ocr_engine):
    logger.info(f"  Attempting chunk {start_page}-{end_page} (Size: {len(range(start_page, end_page))+1})")
    
    try:
        images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page, dpi=DPI)
        if not images:
            raise Exception("No images generated.")

        processed_docs, chunk_statuses, chunk_text = process_chunk_images(images, start_page, pdf_path, current_file_status, ocr_engine)
        
        if processed_docs:
            logger.info(f"  > Chunk {start_page}-{end_page} SUCCESS.")
            for p, s in chunk_statuses.items():
                record_page_status(current_file_status, p, s)
            del images
            gc.collect()
            return processed_docs, chunk_text
        
        logger.warning(f"  > Chunk {start_page}-{end_page} FAILED.")

        if current_batch_size > RETRY_BATCH_SIZE_1:
            next_size = RETRY_BATCH_SIZE_1
        elif current_batch_size > RETRY_BATCH_SIZE_2:
            next_size = RETRY_BATCH_SIZE_2
        else:
            logger.warning(f"  > Page {start_page} Paddle failed. Attempting Tesseract Failover...")
            log_issue(pdf_path, start_page, "Paddle Failure", "Switching to Tesseract failover.")
            
            tess_doc, tess_success, tess_text = process_page_with_tesseract(start_page, images[0], pdf_path)
            
            del images
            gc.collect()

            if tess_success and tess_doc:
                record_page_status(current_file_status, start_page, "TesseractOCR")
                return [tess_doc], tess_text
            
            logger.error(f"  > Page {start_page} Tesseract ALSO failed. Using fallback image.")
            log_issue(pdf_path, start_page, "Total Failure", "Paddle and Tesseract failed. Using image-only fallback.")
            record_page_status(current_file_status, start_page, "FALLBACK_IMAGE_ONLY")
            
            try:
                img_fallback = convert_from_path(pdf_path, first_page=start_page, last_page=start_page, dpi=DPI)[0]
                fallback_doc = create_fallback_page(img_fallback)
                return [fallback_doc], ""
            except Exception:
                return [], ""

        logger.info(f"  > Splitting chunk {start_page}-{end_page} into sub-chunks of size {next_size}...")
        del images
        gc.collect()

        collected_docs = []
        collected_text = ""
        for sub_start in range(start_page, end_page + 1, next_size):
            sub_end = min(sub_start + next_size - 1, end_page)
            sub_results, sub_text = attempt_chunk_processing(pdf_path, sub_start, sub_end, next_size, current_file_status, ocr_engine)
            collected_docs.extend(sub_results)
            collected_text += sub_text
            
        return collected_docs, collected_text

    except Exception as e:
        logger.error(f"Error in chunk {start_page}-{end_page}: {e}")
        return [], ""

def ocr_pdf_adaptive(input_pdf_path, output_pdf_path, output_txt_path):
    logger.info(f"Starting file: {input_pdf_path}")
    
    detected_lang = detect_language(input_pdf_path)
    ocr_engine = get_paddle_instance(detected_lang)
    
    try:
        src_doc = fitz.open(input_pdf_path)
        total_pages = src_doc.page_count
        src_doc.close()
        logger.info(f"Total Pages: {total_pages}")
        
        current_file_status = {
            "file": os.path.basename(input_pdf_path),
            "total": total_pages,
            "lang": detected_lang,
            "missing": [],
            "details": {}
        }
        
        doc_out = fitz.open()
        full_document_text = ""
        
        current_page = 1
        while current_page <= total_pages:
            end_page = min(current_page + INITIAL_BATCH_SIZE - 1, total_pages)
            
            chunk_docs, chunk_text = attempt_chunk_processing(input_pdf_path, current_page, end_page, INITIAL_BATCH_SIZE, current_file_status, ocr_engine)
            
            if chunk_docs:
                for d in chunk_docs:
                    doc_out.insert_pdf(d)
                    d.close()
                full_document_text += chunk_text
            else:
                logger.critical(f"Chunk {current_page}-{end_page} returned NO pages!")
                log_issue(input_pdf_path, f"{current_page}-{end_page}", "Critical Data Loss", "Chunk returned no content.")

            if current_page % 100 == 1:
                gc.collect()

            current_page += INITIAL_BATCH_SIZE

        logger.info(f"Saving PDF to: {output_pdf_path}...")
        doc_out.save(output_pdf_path, deflate=True) 
        doc_out.close()
        
        logger.info(f"Saving Text to: {output_txt_path}...")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(full_document_text)

        doc_reports.append(current_file_status)
        logger.info("File completed successfully.\n")
        return True

    except Exception as e:
        logger.critical(f"Critical error on file {input_pdf_path}: {e}")
        log_issue(input_pdf_path, "All", "Critical File Failure", str(e))
        return False

def save_csv_report():
    if issues_list:
        df = pd.DataFrame(issues_list)
        df.to_csv(csv_report_file, index=False)
        logger.info(f"Issue report saved to {csv_report_file}")
    else:
        logger.info("No issues recorded.")

def main():
    source_folder = "/app/ocr_source"
    output_folder = "/app/ocr_output"
    logger.info("Starting Multi-Language OCR Processing (fastText Enabled)")
    
    files_to_process = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                files_to_process.append(os.path.join(root, file))
    
    logger.info(f"Found {len(files_to_process)} PDFs to process.")
    
    for input_path in files_to_process:
        rel_path = os.path.relpath(input_path, source_folder)
        output_sub_dir = os.path.join(output_folder, os.path.dirname(rel_path))
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
            
        output_pdf_path = os.path.join(output_folder, rel_path)
        output_txt_path = os.path.splitext(output_pdf_path)[0] + ".txt"
        
        ocr_pdf_adaptive(input_path, output_pdf_path, output_txt_path)
        
        save_csv_report()
        generate_final_report()

if __name__ == "__main__":
    main()
