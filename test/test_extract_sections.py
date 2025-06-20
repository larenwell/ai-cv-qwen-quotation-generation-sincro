from pathlib import Path
from src.data_ingestion.qwen_extractor import QwenExtractor

# Path to your _full.html file
html_path = Path("data/qwen_outputs/mineria/ES-22-GCO-CO-000627-00/ES-22-GCO-CO-000627-00_full.html")

# Read the HTML content
html_content = html_path.read_text(encoding="utf-8")

# Use the static method directly
sections = QwenExtractor.extract_sections_from_html(html_content)

print("OBJETIVOS:\n", sections['objetivos'])
print("\nALCANCES:\n", sections['alcances'])
print("\nENTREGABLES:\n", sections['entregables'])