import os
from pathlib import Path
from pqdm.processes import pqdm
import pdfplumber
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_to_txt_conversion.log")
    ]
)
logger = logging.getLogger(__name__)

def convert_pdf_to_txt(pdf_path: Path, output_dir: Path):
    """Convert a single PDF file to a .txt file."""
    try:
        # Read the PDF
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        # Write the text to a file
        output_file = output_dir / f"{pdf_path.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"Successfully converted: {pdf_path.name} -> {output_file.name}")
    except Exception as e:
        logger.error(f"Error converting {pdf_path.name}: {e}")

def process_pdfs(input_dir: str, output_dir: str, max_workers: int = 4):
    """Convert all PDFs in the input directory to .txt files in the output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    # Process PDFs in parallel using pqdm
    pqdm([(pdf_file, output_path) for pdf_file in pdf_files], convert_pdf_to_txt, n_jobs=max_workers, argument_type='args')

if __name__ == "__main__":
    # Specify the input and output directories
    input_directory = r"C:\\Users\\ASUS\\Desktop\\Data"
    output_directory = r"C:\\Users\\ASUS\\Desktop\\PreProcessed"

    # Run the PDF to TXT conversion
    process_pdfs(input_directory, output_directory, max_workers=32)

