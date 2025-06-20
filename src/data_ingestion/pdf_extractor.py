import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


class PDFExtractor:
    """Extracts and parses text from PDF quotes."""

    def __init__(self, pdf_dir: str):
        """Initialize the PDF extractor.
        
        Args:
            pdf_dir: Path to the directory containing PDF quotes
        """
        self.pdf_dir = Path(pdf_dir)
        self.industries = ['educación', 'industria', 'mineria']
        
        # Regex patterns for field extraction
        self.patterns = {
            #'contratante': r'CONTRATANTE:?\s*(.*?)(?=\n[A-Z]|$)',
            #'proyecto': r'PROYECTO:?\s*(.*?)(?=\n[A-Z]|$)',
            #'area': r'ÁREA\s*INVOLUCRADA:?\s*(.*?)(?=\n[A-Z]|$)',
            'objetivos': r'OBJETIVO:?\s*(.*?)(?=\n[A-Z]|$)',
            'alcances': r'ALCANCE:?\s*(.*?)(?=\n[A-Z]|$)',
            'entregables': r'ENTREGABLES:?\s*(.*?)(?=\n[A-Z]|$)'
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file using OCR if needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            # First try direct text extraction
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                
                # If no text found, use OCR
                if not text.strip():
                    images = convert_from_path(pdf_path)
                    text = ""
                    for image in images:
                        text += pytesseract.image_to_string(image, lang='spa')
                
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def parse_pdf_id(self, filename: str) -> str:
        """Extract ID from PDF filename.
        
        Args:
            filename: PDF filename (e.g. 'ES-22-GCO-CO-000368-00.pdf')
            
        Returns:
            Extracted ID (e.g. 'ES-22-GCO-CO-000368')
        """
        # Remove file extension and trailing digits
        base_name = os.path.splitext(filename)[0]
        return re.sub(r'-\d+$', '', base_name)

    def extract_fields(self, text: str) -> Dict[str, str]:
        """Extract structured fields from unstructured text.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Dictionary of extracted fields
        """
        fields = {}
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            fields[field] = match.group(1).strip() if match else ""
        return fields

    def process_pdf(self, pdf_path: Path, industry: str) -> Dict[str, str]:
        """Process a single PDF file and extract all required fields.
        
        Args:
            pdf_path: Path to the PDF file
            industry: Industry category (educacion/industria/mineria)
            
        Returns:
            Dictionary with all extracted fields
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract fields
        fields = self.extract_fields(text)
        
        # Add metadata
        fields.update({
            'id': self.parse_pdf_id(pdf_path.name),
            'industry': industry,
            'filename': pdf_path.name
        })
        
        return fields

    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """Process all PDFs in the configured directory.
        
        Returns:
            List of dictionaries containing extracted data from all PDFs
        """
        all_data = []
        
        for industry in self.industries:
            industry_dir = self.pdf_dir / industry
            if not industry_dir.exists():
                continue
                
            for pdf_file in industry_dir.glob('*.pdf'):
                try:
                    data = self.process_pdf(pdf_file, industry)
                    all_data.append(data)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
                    continue
        
        return all_data
