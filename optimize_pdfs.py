import argparse
import logging
import os
import subprocess

# --- Configuration ---
INPUT_DIR = "/app/ocr_output"
# Ghostscript PDF Settings:
# /screen   (72 dpi, smallest)
# /ebook    (150 dpi, medium - Recommended)
# /printer  (300 dpi, high)
# /prepress (300 dpi, max quality, preserves color)
DEFAULT_QUALITY = "/prepress"
VALID_QUALITIES = {"/screen", "/ebook", "/printer", "/prepress"}
GHOSTSCRIPT_TIMEOUT = 300  # seconds; prevents hung threads on malformed PDFs

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def optimize_pdf(input_path, quality=DEFAULT_QUALITY):
    if quality not in VALID_QUALITIES:
        logger.error("Invalid Ghostscript quality setting: %s", quality)
        return

    if input_path.endswith("_optimized.pdf"):
        return

    file_name = os.path.basename(input_path)
    # Temp path for ghostscript output
    temp_path = input_path + ".tmp.pdf"

    logger.info(f"Optimizing: {file_name} [Quality: {quality}]...")

    # Ghostscript command
    cmd = [
        "gs",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS={quality}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={temp_path}",
        input_path
    ]

    try:
        # Run Ghostscript
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=GHOSTSCRIPT_TIMEOUT)

        # Check sizes
        original_size = os.path.getsize(input_path)
        if original_size == 0:
            logger.warning("Skipping zero-byte file: %s", input_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return

        if os.path.exists(temp_path):
            new_size = os.path.getsize(temp_path)

            # Integrity check: verify output is a valid PDF before replacing (TD-011)
            if new_size < 100:
                logger.error(f"  > Failed: Output too small ({new_size} bytes), keeping original")
                os.remove(temp_path)
                return

            with open(temp_path, "rb") as check_f:
                header = check_f.read(5)
            if header != b"%PDF-":
                logger.error("  > Failed: Output is not a valid PDF, keeping original")
                os.remove(temp_path)
                return

            reduction = (1 - (new_size / original_size)) * 100

            # Safe replace: backup original, move temp, then delete backup
            backup_path = input_path + ".bak"
            try:
                os.rename(input_path, backup_path)
                os.rename(temp_path, input_path)
                os.remove(backup_path)
            except OSError:
                # Restore from backup if rename failed
                if os.path.exists(backup_path) and not os.path.exists(input_path):
                    os.rename(backup_path, input_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

            logger.info(f"  > Done! Size: {original_size/1024/1024:.2f}MB -> {new_size/1024/1024:.2f}MB (-{reduction:.1f}%)")
        else:
            logger.error("  > Failed: Output file not created.")

    except subprocess.TimeoutExpired:
        logger.warning(
            "Ghostscript timed out after %ds for %s", GHOSTSCRIPT_TIMEOUT, input_path
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"  > Failed to optimize {file_name}: {e.stderr.decode()}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        logger.error(f"  > Error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Optimize PDFs using Ghostscript")
    parser.add_argument("--quality", default=DEFAULT_QUALITY,
                        choices=sorted(VALID_QUALITIES),
                        help="Ghostscript PDF settings")
    args = parser.parse_args()

    logger.info(f"Starting PDF Optimization Scan (Target Quality: {args.quality})")
    
    pdf_count = 0
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".pdf") and not file.lower().endswith("_optimized.pdf"):
                full_path = os.path.join(root, file)
                optimize_pdf(full_path, args.quality)
                pdf_count += 1

    logger.info(f"Scan complete. Processed {pdf_count} potential files.")

if __name__ == "__main__":
    main()
