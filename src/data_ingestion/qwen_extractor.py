import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from typing import List, Dict
import torch
import random
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import re
from bs4 import BeautifulSoup
import unicodedata
import gc
import time
import psutil
import humanize

class QwenExtractor:
    def __init__(self, pdf_dir: str, output_dir: str, device=None, seed=42):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Use CUDA, then MPS, else CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[QwenExtractor] Using device: {self.device}")

        # Load Qwen model and processor
        self.checkpoint = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.model.to(self.device)

    def get_memory_usage(self) -> str:
        """Get current memory usage and available memory in human-readable format."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        return (
            f"Process: {humanize.naturalsize(memory_info.rss)} | "
            f"Available: {humanize.naturalsize(system_memory.available)} | "
            f"Total: {humanize.naturalsize(system_memory.total)}"
        )

    def process_all_pdfs(self) -> List[Dict]:
        records = []
        for industry_dir in self.pdf_dir.iterdir():
            if not industry_dir.is_dir():
                continue
            for pdf_file in industry_dir.glob("*.pdf"):
                print(f"\nProcessing PDF: {pdf_file} (Industry: {industry_dir.name})")
                fields = self.process_pdf(pdf_file, industry_dir.name)
                if fields:
                    records.append(fields)
        return records

    def process_pdf(self, pdf_path: Path, industria: str) -> Dict:
        pdf_name = pdf_path.stem
        pdf_output_dir = self.output_dir / industria / pdf_name
        pdf_output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Convert PDF to images
        print(f"Phase 1: Converting PDF to images for {pdf_name}")
        image_paths = self.convert_pdf_to_images(pdf_path, pdf_output_dir)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Phase 2: Process each image with Qwen
        print(f"Phase 2: Processing images with Qwen for {pdf_name}")
        html_files = []
        for img_path in image_paths:
            html_file = self.process_single_image(img_path)
            if html_file:
                html_files.append(html_file)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Phase 3: Consolidate HTML
        print(f"Phase 3: Consolidating HTML for {pdf_name}")
        doc_path = pdf_output_dir / f"{pdf_name}_full.html"
        self.consolidate_html(html_files, doc_path)
        gc.collect()

        # Phase 4: Extract sections
        print(f"Phase 4: Extracting sections for {pdf_name}")
        fields = self.extract_sections_from_html(doc_path.read_text())
        fields.update({
            'id': self.parse_pdf_id(pdf_path.name),
            'industria': industria,
            'filename': pdf_path.name
        })
        print(f"Extracted fields: {fields}")
        return fields

    def convert_pdf_to_images(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """Convert PDF to images and return list of image paths."""
        image_paths = []
        images = convert_from_path(str(pdf_path), dpi=100)
        for idx, image in enumerate(images):
            img_path = output_dir / f"page_{idx+1}.png"
            print(f"  Saving image for page {idx+1} to {img_path}")
            image.save(img_path)
            image_paths.append(img_path)
            # Clean up the image object
            del image
        return image_paths

    def process_single_image(self, img_path: Path) -> Path:
        """Process a single image with Qwen and return the path to the cleaned HTML file."""
        start_time = time.time()
        print(f"\nProcessing {img_path.name}:")
        print(f"Memory status: {self.get_memory_usage()}")
        
        # Load and process image
        image = Image.open(img_path)
        html = self.qwen_inference(image)
        print(f"Memory status after inference: {self.get_memory_usage()}")
        
        del image
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Save raw HTML
        raw_html_path = img_path.parent / f"{img_path.stem}_raw.html"
        print(f"Saving raw HTML to {raw_html_path}")
        with open(raw_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        del html
        gc.collect()

        # Clean HTML
        print(f"Cleaning HTML for {img_path.name}...")
        clean_html = self.clean_and_format_html(raw_html_path.read_text())
        clean_html_path = img_path.parent / f"{img_path.stem}_clean.html"
        print(f"Saving cleaned HTML to {clean_html_path}")
        with open(clean_html_path, "w", encoding="utf-8") as f:
            f.write(clean_html)
        del clean_html
        gc.collect()

        total_time = time.time() - start_time
        print(f"Total processing time for {img_path.name}: {total_time:.2f} seconds")

        return clean_html_path

    def consolidate_html(self, html_files: List[Path], output_path: Path):
        """Consolidate all HTML files into a single document."""
        all_html = []
        for html_file in html_files:
            with open(html_file, "r", encoding="utf-8") as f:
                all_html.append(f.read())
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_html))

    def qwen_inference(self, image: Image.Image) -> str:
        """Run Qwen inference on an image and return the HTML output."""
        inference_start = time.time()
        print(f"Starting Qwen inference...")
        
        prompt = "QwenVL HTML, beware of weird table layouts and format them as well as possible."
        system_prompt = (
            "You are an AI specialized in recognizing and extracting text from images. "
            "Your mission is to analyze the image document and generate the result in "
            "QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity."
        )
        image_path = str(image.filename)
        img_url = "file://" + image_path
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": img_url}
            ]}
        ]
        
        print("Preparing inputs...")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        
        print("Generating output...")
        output_ids = self.model.generate(**inputs, max_new_tokens=100000)
        
        print("Decoding output...")
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # Clean up tensors
        del inputs
        del output_ids
        del generated_ids
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        inference_time = time.time() - inference_start
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        return output_text[0]

    def clean_and_format_html(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        # ... (insert your cleaning logic here) ...
        return str(soup)

    def parse_pdf_id(self, filename: str) -> str:
        base_name = os.path.splitext(filename)[0]
        return re.sub(r'-\d+$', '', base_name)

    @staticmethod
    def extract_sections_from_html(html: str) -> dict:
        """
        Extract all sections from consolidated HTML using BeautifulSoup.
        Handles typos, multiple pages, and collects all content between main section headings.
        Cleans HTML tags and removes footer information.
        """
        def normalize(text):
            text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
            return text.lower().strip()

        def is_numbered_section(text: str) -> bool:
            # Check if the text starts with a number followed by a dot or parenthesis
            return bool(re.match(r'^\d+[\.\)]\s+', text.strip()))

        def is_footer_content(text: str) -> bool:
            footer_patterns = [
                r'40 AÑOS DE PROTEGIENDO VIDAS',
                r'Av\. República de Panamá',
                r'www\.essac\.com\.pe',
                r'ES-\d{2}-[A-Z]+-[A-Z]+-\d{6}-\d{2}'
            ]
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in footer_patterns)

        def clean_html_content(html_content: str) -> str:
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove footer content
            for tag in soup.find_all(['p', 'div']):
                if is_footer_content(tag.get_text()):
                    tag.decompose()
            
            # Process lists (both ul and ol)
            for list_tag in soup.find_all(['ul', 'ol']):
                items = []
                for li in list_tag.find_all('li', recursive=False):
                    # Get the text of the current list item
                    item_text = li.get_text().strip()
                    # Check if this list item has nested lists
                    nested_lists = li.find_all(['ul', 'ol'], recursive=False)
                    if nested_lists:
                        # Process nested lists
                        nested_items = []
                        for nested_list in nested_lists:
                            for nested_li in nested_list.find_all('li', recursive=False):
                                nested_items.append(f"  • {nested_li.get_text().strip()}")
                        # Combine main item with nested items
                        items.append(f"• {item_text}")
                        items.extend(nested_items)
                    else:
                        items.append(f"• {item_text}")
                # Replace the list with formatted text
                list_tag.replace_with('\n'.join(items))
            
            # Convert paragraphs to clean text, preserving lettered/numbered points
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                # Check if it's a lettered or numbered point (e.g., "a)", "1.", etc.)
                if re.match(r'^[a-z]\)|^\d+\.', text.lower()):
                    p.replace_with(f"\n{text}")
                else:
                    p.replace_with(text)
            
            # Get the cleaned text and remove extra whitespace
            text = soup.get_text()
            # Remove multiple newlines and spaces
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            # Ensure proper spacing after bullet points
            text = re.sub(r'•\s*', '• ', text)
            return text.strip()

        # Define all sections with their possible variations and typos
        section_variants = {
            'objetivos': [r'objeti[vo]{1,2}s?'],
            'alcances': [r'alcance[s]?', r'alcanz[es]{1,2}'],
            'entregables': [r'entregable[s]?'],
            'introduccion': [r'introducci[oó]n'],
            'antecedentes': [r'antecedente[s]?'],
            'requerimiento_informacion': [r'requerimiento[s]?\s+de\s+informaci[oó]n'],
            'documentacion_requerida': [r'documentaci[oó]n\s+requerida'],
            'condiciones_generales': [r'condiciones\s+generales', r'consideraciones\s+generales'],
            'exclusiones': [r'exclusiones?\s+generales?', r'exclusiones?'],
            'honorarios': [r'honorarios?'],
            'plazos': [r'plazos?'],
            'confidencialidad': [r'confidencialidad'],
            'certificaciones_acreditaciones': [r'certificaciones?\s+y?\s*acreditaciones?'],
            'normatividad_aplicable': [r'normatividad\s+aplicable'],
            'area_involucrada': [r'[aá]rea\s+involucrada'],
            'recursos_esfuerzos': [r'recursos?\s+y?\s*esfuerzos?'],
            'no_solicitacion': [r'no\s*solicitaci[oó]n']
        }
        
        patterns = {k: re.compile('|'.join(v), re.IGNORECASE) for k, v in section_variants.items()}
        main_section_keys = list(patterns.keys())

        soup = BeautifulSoup(html, 'html.parser')
        bodies = soup.find_all('body')
        tags = []
        for body in bodies:
            tags.extend(body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ol', 'ul', 'div'], recursive=True))

        sections = {k: [] for k in main_section_keys}
        current_section = None

        for tag in tags:
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                norm_text = normalize(tag.get_text())
                
                # Check if we hit a numbered section (for sections that should stop at numbered sections)
                if current_section in ['entregables', 'condiciones_generales', 'exclusiones', 'honorarios', 'plazos'] and is_numbered_section(tag.get_text()):
                    current_section = None
                    continue
                
                found_section = None
                for key, pattern in patterns.items():
                    if pattern.search(norm_text):
                        found_section = key
                        break
                if found_section:
                    current_section = found_section
                    continue
            if current_section:
                sections[current_section].append(str(tag))

        # Clean the content of each section
        return {k: clean_html_content('\n'.join(v)) for k, v in sections.items()}