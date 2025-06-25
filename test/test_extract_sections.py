from pathlib import Path
from src.data_ingestion.qwen_extractor import QwenExtractor

# Path to your _full.html file
html_path = Path("data/qwen_outputs/mineria/ES-22-GCO-CO-000627-00/ES-22-GCO-CO-000627-00_full.html")

# Read the HTML content
html_content = html_path.read_text(encoding="utf-8")

# Use the static method directly
sections = QwenExtractor.extract_sections_from_html(html_content)

# Test all sections
all_sections = [
    'objetivos', 'alcances', 'entregables', 'introduccion', 'antecedentes',
    'requerimiento_informacion', 'documentacion_requerida', 'condiciones_generales',
    'exclusiones', 'honorarios', 'plazos', 'confidencialidad',
    'certificaciones_acreditaciones', 'normatividad_aplicable', 'area_involucrada',
    'recursos_esfuerzos', 'no_solicitacion'
]

for section in all_sections:
    print(f"\n{section.upper()}:\n{sections.get(section, 'NOT FOUND')}")
    print("-" * 50)