import pandas as pd
from pathlib import Path
from typing import Dict, List

from ..data_ingestion.qwen_extractor import QwenExtractor
from ..data_ingestion.excel_loader import ExcelLoader

class KnowledgeBaseBuilder:
    """Builds the knowledge base by combining Qwen-extracted PDF and Excel data."""

    def __init__(self, pdf_dir: str, excel_path: str, output_dir: str, device='cuda'):
        """Initialize the knowledge base builder.
        
        Args:
            pdf_dir: Path to the directory containing PDF quotes
            excel_path: Path to the elastika.xlsx file
            output_dir: Path to save Qwen outputs
            device: Device for Qwen model (default 'cuda')
        """
        self.qwen_extractor = QwenExtractor(pdf_dir, output_dir, device=device)
        self.excel_loader = ExcelLoader(excel_path)
        self.output_dir = Path(output_dir)

    def build(self) -> pd.DataFrame:
        """Build the knowledge base by combining Qwen-extracted PDF and Excel data.
        
        Returns:
            DataFrame containing the combined knowledge base
        """
        # Step 1: Process PDFs with QwenExtractor (returns records)
        pdf_records = self.qwen_extractor.process_all_pdfs()
        pdf_df = pd.DataFrame(pdf_records)

        # Step 2: Load Excel data
        excel_df = self.excel_loader.process_all_sheets()

        if pdf_df.empty or excel_df.empty:
            print("Warning: No data found in PDFs or Excel file")
            return pd.DataFrame()

        # Step 3: Merge PDF and Excel data
        merged_df = self.merge_data(pdf_df, excel_df)

        # Step 4: Clean and standardize the final DataFrame
        final_df = self.clean_final_data(merged_df)

        return final_df

    def merge_data(self, pdf_df: pd.DataFrame, excel_df: pd.DataFrame) -> pd.DataFrame:
        """Merge PDF and Excel data.
        
        Args:
            pdf_df: DataFrame containing PDF data
            excel_df: DataFrame containing Excel data
            
        Returns:
            Merged DataFrame
        """
        # Merge on both industry and ID
        merged_df = pd.merge(
            pdf_df,
            excel_df,
            on=['id', 'industria'],
            how='inner',
            suffixes=('', '_excel')
        )
        return merged_df

    def clean_final_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the final knowledge base.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Drop duplicate columns (if any)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Sort columns in a logical order
        column_order = [
            'id',
            'industria',
            'filename',
            'area',
            'tipo_proyecto',
            'tipo_cliente',
            'nombre_cliente',
            'nombre_cotizacion',
            'anio',
            'estado_cotizacion',
            'sector_proyecto',
            'id_cliente',
            'objetivos',
            'alcances',
            'entregables',
            'monto',
            'valor_venta',
            'margen_planeado',
            'margen_actual', 
            'utilidad_planeada',
            'utilidad_real'
        ]


        # Only include columns that exist in the DataFrame
        column_order = [col for col in column_order if col in df.columns]
        # Reorder columns
        df = df[column_order]
        # Sort by industry and ID
        df = df.sort_values(['industria', 'id'])
        return df

    def save_knowledge_base(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the knowledge base to an Excel file.
        
        Args:
            df: Knowledge base DataFrame
            output_path: Path where to save the Excel file
        """
        if df.empty:
            print("Warning: No data to save")
            return
        try:
            df.to_excel(output_path, index=False, sheet_name='knowledge_base')
            print(f"Knowledge base saved to {output_path}")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
