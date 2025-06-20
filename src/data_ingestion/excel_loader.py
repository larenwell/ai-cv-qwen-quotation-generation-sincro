import pandas as pd
from pathlib import Path
from typing import Dict, List


class ExcelLoader:
    """Loads and processes Excel data from elastika.xlsx."""

    def __init__(self, excel_path: str):
        """Initialize the Excel loader.
        
        Args:
            excel_path: Path to the elastika.xlsx file
        """
        self.excel_path = Path(excel_path)
        self.industries = ['educación', 'industria', 'mineria']
        
        # Actual column names in the Excel file (from the image)
        self.column_mapping = {
            'id': 'ID',
            'area': 'ÁREA',
            'tipo_proyecto': 'TIPO DE PROYECTO',
            'tipo_cliente': 'TIPO CLIENTE',
            'nombre_cliente': 'NOMBRE COMERCIAL DE CLIENTE',
            'nombre_cotizacion': 'NOMBRE DE LA COTIZACIÓN',
            'anio': 'AÑO',
            'estado_cotizacion': 'ESTADO DE COTIZACIÓN ',
            'sector_proyecto': 'SECTOR DE PROYECTO',
            'id_cliente': 'ID_CLIENTE',
            'monto':'MONTO',
            'valor_venta': 'VALOR DE VENTA',
            'margen_planeado': 'Margen Planeado',
            'margen_actual': 'Margen Actual',
            'utilidad_planeada': 'UTILIDAD PLANEADA',
            'utilidad_real': 'utilidad_real'
        }

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Load a specific sheet from the Excel file.
        
        Args:
            sheet_name: Name of the sheet to load
            
        Returns:
            DataFrame containing the sheet data
        """
        try:
            df = pd.read_excel(
                self.excel_path,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            
            # Rename columns to standardized names if they exist
            rename_dict = {v: k for k, v in self.column_mapping.items() if v in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Add industry column
            df['industria'] = sheet_name
            
            return df
            
        except Exception as e:
            print(f"Error loading sheet {sheet_name}: {e}")
            return pd.DataFrame()

    def process_all_sheets(self) -> pd.DataFrame:
        """Process all industry sheets from the Excel file.
        
        Returns:
            DataFrame containing data from all sheets
        """
        all_data = []
        
        for industry in self.industries:
            try:
                df = self.load_sheet(industry)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"Error processing sheet {industry}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
            
        # Combine all sheets
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Clean and standardize data
        combined_df = self.clean_data(combined_df)
        
        return combined_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert ID to string and strip whitespace
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str).str.strip()
        
        # Convert numeric columns to float, replacing non-numeric values with NaN
        numeric_cols = ['sales_value', 'costs', 'actual_profit']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with appropriate defaults
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Ensure industry values are lowercase
        if 'industria' in df.columns:
            df['industria'] = df['industria'].str.lower()
        
        return df
