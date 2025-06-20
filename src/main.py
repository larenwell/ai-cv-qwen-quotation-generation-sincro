from pathlib import Path
from src.knowledge_base.builder import KnowledgeBaseBuilder


def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define paths
    pdf_dir = project_root / 'data' / 'pdfs'
    excel_path = project_root / 'data' / 'templates' / 'elastika.xlsx'
    output_dir = project_root / 'data' / 'qwen_outputs'  # For Qwen outputs
    output_path = project_root / 'data' / 'knowledge_base.xlsx'
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the knowledge base builder
    builder = KnowledgeBaseBuilder(str(pdf_dir), str(excel_path), str(output_dir), device='cpu')
    knowledge_base = builder.build()
    
    # Save the knowledge base
    builder.save_knowledge_base(knowledge_base, str(output_path))
    
    # Print summary
    print("\nKnowledge Base Summary:")
    print(f"Total records: {len(knowledge_base)}")
    print("\nRecords by industry:")
    print(knowledge_base['industria'].value_counts())


if __name__ == '__main__':
    main()
