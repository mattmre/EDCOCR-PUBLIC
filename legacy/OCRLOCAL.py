import os

import fitz  # PyMuPDF

import pytesseract

from PIL import Image

from pdf2image import convert_from_path



# Set path to tesseract executable

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



POPPLER_PATH = r"C:\MM_WORK\poppler-24.08.0\Library\bin"



def ocr_pdf(input_pdf_path, output_pdf_path, failed_files):

    try:

        print(f"Processing: {input_pdf_path}")

        images = convert_from_path(input_pdf_path, poppler_path=POPPLER_PATH)



        doc = fitz.open()

        

        for img in images:

            text = pytesseract.image_to_string(img, config='--psm 6')

            img_pdf = fitz.open()

            rect = fitz.Rect(0, 0, img.width, img.height)

            img_page = img_pdf.new_page(width=img.width, height=img.height)

            img_bytes = img.convert("RGB").tobytes()

            img_page.insert_image(rect, stream=img_bytes)

            img_page.insert_text((50, 50), text, fontsize=12, color=(0, 0, 0))

            doc.insert_pdf(img_pdf)

        

        doc.save(output_pdf_path)

        doc.close()

        print(f"Successfully saved searchable PDF: {output_pdf_path}\n")

        return True

    except Exception as e:

        print(f"Error processing {input_pdf_path}: {str(e)}")

        failed_files.append(input_pdf_path)

        return False



def process_pdfs_in_folder(source_folder, output_folder):

    """

    Processes all PDFs in a folder (including subfolders) and converts them to searchable PDFs.

    """

    if not os.path.exists(output_folder):

        os.makedirs(output_folder)



    total_files = 0

    successful_files = 0

    failed_files = []



    for root, _, files in os.walk(source_folder):

        for file in files:

            if file.lower().endswith(".pdf"):

                total_files += 1

                input_pdf_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, source_folder)

                output_subfolder = os.path.join(output_folder, relative_path)

                if not os.path.exists(output_subfolder):

                    os.makedirs(output_subfolder)

                output_pdf_path = os.path.join(output_subfolder, file)

                

                if ocr_pdf(input_pdf_path, output_pdf_path, failed_files):

                    successful_files += 1

    

    # Summary Output

    print("\nProcessing Summary:")

    print(f"Total PDFs Encountered: {total_files}")

    print(f"Total Successful OCR: {successful_files}")

    if failed_files:

        print("Files that failed OCR:")

        for failed_file in failed_files:

            print(failed_file)

    else:

        print("No OCR failures encountered.")

    

if __name__ == "__main__":

    source_folder = r"C:\\MM_WORK\\EDCOCR\\ocr_source\\"

    output_folder = r"C:\\MM_WORK\\EDCOCR\\ocr_output\\"

    

    print("Starting OCR PDF conversion...")

    process_pdfs_in_folder(source_folder, output_folder)

    print("OCR processing complete.")

    input("Press Enter to exit...")