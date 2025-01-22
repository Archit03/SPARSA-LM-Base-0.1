import os
import hashlib
from tqdm import tqdm
from PyPDF2 import PdfReader
import logging
import pikepdf

# Setup logging
logging.basicConfig(
    filename="logs/pdf_duplicate_removal.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def repair_pdf(file_path):
    """
    Attempt to repair a corrupted PDF file using pikepdf.
    Returns True if the PDF was repaired successfully, False otherwise.
    """
    try:
        with pikepdf.open(file_path) as pdf:
            repaired_path = file_path + ".repaired"
            pdf.save(repaired_path)
            os.replace(repaired_path, file_path)  # Replace the original file with the repaired one
        logging.info(f"Repaired PDF: {file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to repair PDF {file_path}: {e}")
        return False

def calculate_pdf_hash(file_path):
    """
    Calculate MD5 hash of the textual content of a PDF file.
    Repairs the file if necessary and skips if unprocessable.
    """
    hasher = hashlib.md5()
    try:
        # Repair PDF if it fails to open
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            logging.warning(f"Repairing PDF {file_path} due to error: {e}")
            if not repair_pdf(file_path):
                return None
            reader = PdfReader(file_path)  # Retry after repair

        text_content = ""
        for page in reader.pages:
            try:
                text_content += page.extract_text() or ""  # Extract text from each page
            except Exception as e:
                logging.warning(f"Error extracting text from page in {file_path}: {e}")
        hasher.update(text_content.encode("utf-8", errors="ignore"))
    except Exception as e:
        logging.error(f"Error processing PDF {file_path}: {e}")
        return None
    return hasher.hexdigest()

def remove_duplicate_pdfs(directory):
    """
    Scan a directory for duplicate PDF files based on content and remove them.
    Logs all actions and skips problematic files.
    """
    hashes = {}  # Store file hashes and paths
    duplicates = []  # Store duplicate file paths

    files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.lower().endswith(".pdf")
    ]

    for file_path in tqdm(files, desc="Scanning PDFs for duplicates"):
        file_hash = calculate_pdf_hash(file_path)

        if not file_hash:  # Skip files with errors
            continue

        if file_hash in hashes:
            logging.info(f"Duplicate found: {file_path}")
            duplicates.append(file_path)  # Add duplicate to the list
        else:
            hashes[file_hash] = file_path

    # Remove duplicate files
    for duplicate in tqdm(duplicates, desc="Removing duplicates"):
        try:
            os.remove(duplicate)
            logging.info(f"Deleted: {duplicate}")
        except Exception as e:
            logging.error(f"Error deleting {duplicate}: {e}")

    print(f"Completed. Total duplicates removed: {len(duplicates)}")
    logging.info(f"Completed. Total duplicates removed: {len(duplicates)}")

# Specify the directory to scan
if __name__ == "__main__":
    directory_to_scan = r"D:\PubMed\new"  # Change this to your target directory
    remove_duplicate_pdfs(directory_to_scan)

