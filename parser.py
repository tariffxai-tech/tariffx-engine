import os
from pypdf import PdfReader

def extract_invoice_text(pdf_path):
    """
    Extracts raw text from a commercial invoice PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Invoice file not found at: {pdf_path}")
        
    reader = PdfReader(pdf_path)
    extracted_text = ""
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            extracted_text += f"\n--- PAGE {i+1} ---\n" + text
            
    return extracted_text.strip()

if __name__ == "__main__":
    print("Invoice Parser initialized successfully.")