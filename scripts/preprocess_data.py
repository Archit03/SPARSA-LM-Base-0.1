import os
import re
import logging
from tqdm import tqdm
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import torch

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(device)

# Directory paths
DOWNLOAD_DIR = r"C:\Users\ASUS\Desktop\Data"
OUTPUT_DIR = r"C:\Users\ASUS\Desktop\PreProcessed"

# Logging configuration
logging.basicConfig(
    filename="preprocessing.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def ensure_output_directory():
    """Ensure the output directory exists."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def clean_text(text, language="en"):
    """Clean and normalize the extracted text for training."""
    try:
        # Convert to tensor for processing
        text_tensor = torch.tensor([ord(c) for c in text], device=device)

        # Lowercase text
        text_tensor = torch.tensor(
            [ord(c.lower()) if 'A' <= c <= 'Z' else ord(c) for c in text],
            device=device
        )

        # Convert back to string
        text = ''.join(chr(c) for c in text_tensor.tolist())

        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)

        # Remove special characters and digits (except for medical/scientific terms)
        if language == "en":
            text = re.sub(r'[^a-z\s\.\,\-\_\']', ' ', text)  # Preserve common punctuation
        elif language == "hi":  # Example for Hindi support
            text = re.sub(r'[^\u0900-\u097Fa-z\s\.\,\-\_\']', ' ', text)

        # Remove single-character words (except 'a' and 'i')
        text = re.sub(r'\b[b-hj-z]\b', ' ', text)

        # Deduplicate consecutive spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return ""

def save_text_to_file(text, output_path):
    """Save cleaned text to a .txt file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        logging.info(f"Saved processed text to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save text to {output_path}: {e}")

def process_file(file):
    """Process a single file (PDF or TXT)."""
    file_path = os.path.join(DOWNLOAD_DIR, file)
    output_file = os.path.splitext(file)[0] + ".txt"
    output_path = os.path.join(OUTPUT_DIR, output_file)

    try:
        # Extract raw text
        if file.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        elif file.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                raw_text = txt_file.read()
        else:
            logging.warning(f"Skipping unsupported file format: {file}")
            return

        # Clean text and deduplicate lines
        cleaned_text = clean_text(raw_text)
        deduplicated_text = "\n".join(Counter(cleaned_text.splitlines()))

        # Save processed text
        save_text_to_file(deduplicated_text, output_path)
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")

def process_pdfs_and_txt():
    """Process all PDFs and existing .txt files in the download directory using multi-threading."""
    ensure_output_directory()
    files = os.listdir(DOWNLOAD_DIR)

    logging.info("Processing files with multi-threading...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

def main():
    """Main function to process PDFs and existing .txt files."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(device)

    process_pdfs_and_txt().to(device)
    logging.info("All files processed successfully.")

if __name__ == "__main__":
    main()

