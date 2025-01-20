import os
from pathlib import Path
import pdfplumber
from tqdm import tqdm
import re
from statistics import mean

def detect_heading(text: str) -> bool:
    """
    Detect if a line is likely a heading in medical text.
    """
    heading_patterns = [
        r'^(?:ABSTRACT|INTRODUCTION|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)s?\b',
        r'^[1-9]\.\s+[A-Z][^.]+$',  # Numbered sections
        r'^(?:Table|Figure|Box)\s+\d+',  # Tables and figures
        r'^References$',
        r'^(?:Background|Objectives?|Materials|Patients|Statistical Analysis)',
    ]
    return any(re.match(pattern, text.strip(), re.IGNORECASE) for pattern in heading_patterns)

def detect_list_item(text: str) -> bool:
    """
    Detect if a line is part of a list in medical text.
    """
    list_patterns = [
        r'^\s*[•·○●-]\s+',  # Bullet points
        r'^\s*\(?[0-9]+[\)\.]\s+',  # Numbered lists
        r'^\s*\(?[a-z][\)\.]\s+',  # Alphabetical lists
    ]
    return any(re.match(pattern, text.strip()) for pattern in list_patterns)

def calculate_word_spacing(blocks):
    """
    Calculate average word spacing based on the document's layout.
    """
    spacings = []
    for i in range(1, len(blocks)):
        prev_block = blocks[i-1]
        curr_block = blocks[i]
        
        # Only consider blocks on the same line
        if abs(prev_block['top'] - curr_block['top']) < 2:
            space = curr_block['x0'] - prev_block['x1']
            if space > 0:  # Ignore overlapping or touching blocks
                spacings.append(space)
    
    if spacings:
        avg_spacing = mean(spacings)
        return avg_spacing
    return 4.0  # Default spacing if no valid spaces found

def should_add_space(prev_block, curr_block, avg_spacing):
    """
    Determine if a space should be added between blocks based on layout.
    """
    if not prev_block:
        return False
        
    # Same line check with small tolerance
    if abs(prev_block['top'] - curr_block['top']) > 2:
        return True
        
    space_width = curr_block['x0'] - prev_block['x1']
    
    # Handle special cases
    if space_width < 0:  # Overlapping blocks
        return False
        
    # Check if the gap is wide enough to warrant a space
    return space_width >= (avg_spacing * 0.75)

def clean_medical_text(text: str) -> str:
    """
    Clean and normalize medical text while preserving important formatting.
    """
    try:
        # Basic cleanup while preserving intentional spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve common medical abbreviations and symbols
        text = text.replace(' +/-', '±')
        text = text.replace(' +/- ', '±')
        
        # Handle measurements and units
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        text = re.sub(r'(\d+)\s*°C', r'\1°C', text)
        text = re.sub(r'(\d+)\s*°F', r'\1°F', text)
        text = re.sub(r'(\d+)\s*(mg|ml|kg|cm|mm|µm|µg|ng)', r'\1\2', text)
        
        # Preserve statistical notation
        text = re.sub(r'p\s*[<>]\s*0\.', r'p<0.', text)
        text = re.sub(r'\(\s*p\s*[<>]\s*0\.', r'(p<0.', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'[''`′‵՚ꞌ‛]', "'", text)
        text = re.sub(r'["""]', '"', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'(\d)\s+%', r'\1%', text)  # Fix percentage spacing
        text = re.sub(r'\(\s+', r'(', text)  # Fix opening parenthesis
        text = re.sub(r'\s+\)', r')', text)  # Fix closing parenthesis
        
        return text.strip()
    except Exception as e:
        print(f"Warning: Error in clean_medical_text: {str(e)}")
        return text.strip()

def convert_pdf_to_txt(pdf_path: Path, output_path: Path):
    """
    Convert a medical PDF to text while preserving formatting for ML training.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            formatted_lines = []
            
            for page in pdf.pages:
                try:
                    # Extract and sort blocks by position
                    blocks = sorted(
                        page.extract_words(
                            keep_blank_chars=True,
                            x_tolerance=1.5,  # Reduced tolerance for better word separation
                            y_tolerance=3,
                            split_at_punctuation=False  # Keep punctuation with words
                        ),
                        key=lambda b: (round(b['top']), b['x0'])
                    )
                    
                    if not blocks:
                        continue
                        
                    # Calculate average word spacing for this page
                    avg_spacing = calculate_word_spacing(blocks)
                    
                    # Process blocks into lines
                    current_line = []
                    current_y = None
                    
                    for block in blocks:
                        # New line detection
                        if current_y is None:
                            current_y = round(block['top'])
                        elif abs(block['top'] - current_y) > 3:  # Line break threshold
                            # Process completed line
                            if current_line:
                                line_text = ' '.join(current_line)
                                line_text = clean_medical_text(line_text)
                                if line_text.strip():
                                    formatted_lines.append(line_text)
                            current_line = []
                            current_y = round(block['top'])
                        
                        # Add space if needed
                        if current_line and should_add_space(blocks[blocks.index(block)-1], block, avg_spacing):
                            current_line.append(' ')
                            
                        current_line.append(block['text'])
                    
                    # Process last line
                    if current_line:
                        line_text = ' '.join(current_line)
                        line_text = clean_medical_text(line_text)
                        if line_text.strip():
                            formatted_lines.append(line_text)
                    
                    # Add page break
                    formatted_lines.append('\n')
                    
                except Exception as e:
                    print(f"Warning: Error processing page in {pdf_path}: {str(e)}")
                    continue
            
            # Write the formatted text
            with open(output_path, 'w', encoding='utf-8') as f:
                formatted_text = '\n'.join(line.strip() for line in formatted_lines if line.strip())
                formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
                f.write(formatted_text)
        
        print(f"Successfully converted: {pdf_path.name} -> {output_path.name}")
    
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")

def batch_convert_pdfs(input_dir: str, output_dir: str):
    """
    Batch convert medical PDF files to formatted text suitable for LM training.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}.")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert.")
    errors = []
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            txt_file = output_path / f"{pdf_file.stem}.txt"
            convert_pdf_to_txt(pdf_file, txt_file)
        except Exception as e:
            errors.append((pdf_file.name, str(e)))
            continue
    
    if errors:
        print("\nConversion completed with errors:")
        for filename, error in errors:
            print(f"- {filename}: {error}")

if __name__ == "__main__":
    input_directory = r"D:\PubMed"
    output_directory = r"C:\Users\ASUS\Desktop\PreProcessed\processed\pubmed"
    batch_convert_pdfs(input_directory, output_directory)