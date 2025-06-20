# AI Automatización de Cotizaciones Sincro

## Descripción del Proyecto

Este proyecto implementa un sistema de automatización inteligente para la generación de cotizaciones comerciales utilizando técnicas de procesamiento de lenguaje natural y visión computacional. El sistema extrae información estructurada de propuestas históricas en formato PDF y la combina con datos de costos y márgenes almacenados en hojas de cálculo Excel para crear una base de conocimiento integral.

## Objetivo

El objetivo principal es automatizar el proceso de generación de cotizaciones comerciales mediante:

1. **Extracción inteligente de datos**: Utilización del modelo Qwen2.5-VL-3B-Instruct para extraer texto estructurado de documentos PDF históricos
2. **Consolidación de información**: Integración de datos extraídos de PDFs con información financiera y de costos de Excel
3. **Creación de base de conocimiento**: Construcción de un repositorio estructurado que facilite la generación automática de nuevas cotizaciones
4. **Mejora de eficiencia**: Reducción significativa del tiempo requerido para crear propuestas comerciales estandarizadas

## Alcance

### Funcionalidades Implementadas

- **Procesamiento de PDFs**: Conversión automática de documentos PDF a imágenes y extracción de texto usando visión computacional
- **Extracción de secciones clave**: Identificación y extracción de "objetivos", "alcances" y "entregables" de las propuestas
- **Carga de datos Excel**: Procesamiento de hojas de cálculo con información financiera y de costos
- **Construcción de base de conocimiento**: Integración y consolidación de datos de múltiples fuentes
- **Gestión de memoria**: Optimización del uso de recursos durante el procesamiento de documentos

### Industrias Soportadas

- **Educación**: Propuestas para instituciones educativas
- **Industria**: Propuestas para el sector industrial
- **Minería**: Propuestas para el sector minero

## Arquitectura del Sistema

### Estructura de Directorios

```
ai-automatizacion-cotizaciones-sincro/
├── src/
│   ├── data_ingestion/          # Módulos de ingesta de datos
│   │   ├── qwen_extractor.py    # Extracción de PDFs con Qwen
│   │   ├── excel_loader.py      # Carga de datos Excel
│   │   └── pdf_extractor.py     # Extracción OCR (legado)
│   ├── knowledge_base/          # Construcción de base de conocimiento
│   │   └── builder.py           # Orquestador principal
│   ├── automation/              # Generación automática (futuro)
│   ├── cost_sheet/              # Manejo de plantillas (futuro)
│   ├── outputs/                 # Exportación de resultados (futuro)
│   └── main.py                  # Punto de entrada principal
├── data/
│   ├── pdfs/                    # Documentos PDF por industria
│   │   ├── educación/
│   │   ├── industria/
│   │   └── mineria/
│   ├── templates/               # Plantillas y datos de referencia
│   │   └── elastika.xlsx        # Datos financieros y de costos
│   └── qwen_outputs/            # Salidas del procesamiento Qwen
├── notebooks/                   # Jupyter notebooks para experimentación
├── test/                        # Pruebas unitarias
└── pyproject.toml              # Configuración de dependencias
```

## Módulos del Sistema

### 1. QwenExtractor (`src/data_ingestion/qwen_extractor.py`)

**Propósito**: Extracción inteligente de texto de documentos PDF utilizando el modelo Qwen2.5-VL-3B-Instruct.

**Funcionalidades principales**:
- Conversión de PDFs a imágenes de alta resolución
- Procesamiento de imágenes con modelo de visión computacional
- Generación de HTML estructurado a partir de contenido visual
- Extracción de secciones específicas: objetivos, alcances, entregables
- Gestión optimizada de memoria para procesamiento de documentos grandes

**Características técnicas**:
- Soporte para múltiples dispositivos (CUDA, MPS, CPU)
- Limpieza y formateo automático de HTML
- Consolidación de múltiples páginas en un solo documento
- Manejo robusto de errores y recuperación de memoria

**Métodos principales**:
- `process_all_pdfs()`: Procesa todos los PDFs en el directorio
- `process_pdf()`: Procesa un PDF individual
- `qwen_inference()`: Ejecuta inferencia del modelo Qwen
- `extract_sections_from_html()`: Extrae secciones específicas del HTML

### 2. ExcelLoader (`src/data_ingestion/excel_loader.py`)

**Propósito**: Carga y procesamiento de datos financieros y de costos desde hojas de cálculo Excel.

**Funcionalidades principales**:
- Carga de múltiples hojas de trabajo por industria
- Mapeo automático de columnas a nombres estandarizados
- Limpieza y validación de datos numéricos
- Consolidación de datos de múltiples industrias

**Campos procesados**:
- Información de identificación (ID, nombre de cliente, cotización)
- Datos financieros (monto, valor de venta, márgenes)
- Metadatos del proyecto (área, tipo, sector, estado)

### 3. KnowledgeBaseBuilder (`src/knowledge_base/builder.py`)

**Propósito**: Orquestador principal que combina datos de PDFs y Excel para crear la base de conocimiento final.

**Funcionalidades principales**:
- Coordinación entre QwenExtractor y ExcelLoader
- Fusión de datos basada en ID e industria
- Limpieza y estandarización del conjunto de datos final
- Exportación a formato Excel para análisis posterior

**Proceso de construcción**:
1. Extracción de datos de PDFs usando Qwen
2. Carga de datos financieros desde Excel
3. Fusión de datasets por ID e industria
4. Limpieza y ordenamiento de columnas
5. Exportación de base de conocimiento consolidada

### 4. Main (`src/main.py`)

**Propósito**: Punto de entrada principal que ejecuta el pipeline completo de construcción de la base de conocimiento.

**Configuración**:
- Definición de rutas de entrada y salida
- Inicialización de componentes del sistema
- Ejecución del pipeline de procesamiento
- Generación de reportes de resumen

## Guía de Implementación

### Requisitos del Sistema

- **Python**: 3.12 o superior
- **Memoria RAM**: Mínimo 8GB (recomendado 16GB+)
- **GPU**: Opcional pero recomendado para acelerar procesamiento Qwen
- **Espacio en disco**: Suficiente para almacenar imágenes temporales y salidas

### Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <repository-url>
   cd <project>
   ```

2. **Crear la rama**:
   ```bash
   git checkout -b <nombre-rama>
   ```

3. **Ejecutar uv**:
   ```bash
   uv sync
   ```

4. **Crear entorno virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

### Configuración de Datos

1. **Estructurar directorio de PDFs**:
   ```
   data/pdfs/
   ├── educación/
   │   ├── propuesta_001.pdf
   │   └── propuesta_002.pdf
   ├── industria/
   │   ├── propuesta_003.pdf
   │   └── propuesta_004.pdf
   └── mineria/
       ├── propuesta_005.pdf
       └── propuesta_006.pdf
   ```

2. **Preparar archivo Excel**:
   - Ubicar `elastika.xlsx` en `data/templates/`
   - Asegurar que contenga hojas para cada industria
   - Verificar que las columnas coincidan con el mapeo definido

### Ejecución

1. **Ejecutar pipeline completo**:
   ```bash
   python src/main.py
   ```

2. **Monitorear progreso**:
   - El sistema mostrará información detallada del procesamiento
   - Se generarán archivos intermedios en `data/qwen_outputs/`
   - La base de conocimiento final se guardará en `data/knowledge_base.xlsx`

### Configuración de Dispositivo

El sistema detecta automáticamente el mejor dispositivo disponible:
- **CUDA**: Si hay GPU NVIDIA disponible
- **MPS**: Si hay GPU Apple Silicon disponible
- **CPU**: Como fallback

Para forzar el uso de CPU:
```python
builder = KnowledgeBaseBuilder(pdf_dir, excel_path, output_dir, device='cpu')
```

## Salidas del Sistema

### Archivos Generados

1. **Base de Conocimiento** (`data/knowledge_base.xlsx`):
   - Dataset consolidado con todos los registros procesados
   - Columnas: ID, industria, objetivos, alcances, entregables, datos financieros
   - Ordenado por industria e ID

2. **Salidas Qwen** (`data/qwen_outputs/`):
   - Imágenes de páginas PDF
   - HTML raw y limpio por página
   - Documentos HTML consolidados por PDF

### Estructura de Datos

La base de conocimiento final contiene:

**Campos de identificación**:
- `id`: Identificador único del proyecto
- `industria`: Sector (educación, industria, mineria)
- `filename`: Nombre del archivo PDF original

**Contenido extraído**:
- `objetivos`: Objetivos del proyecto extraídos del PDF
- `alcances`: Alcances del proyecto extraídos del PDF
- `entregables`: Entregables del proyecto extraídos del PDF

**Datos financieros**:
- `monto`: Monto del proyecto
- `valor_venta`: Valor de venta
- `margen_planeado`: Margen planeado
- `margen_actual`: Margen actual
- `utilidad_planeada`: Utilidad planeada
- `utilidad_real`: Utilidad real

## Optimización y Rendimiento

### Gestión de Memoria

- **Limpieza automática**: Liberación de memoria después de cada página procesada
- **Monitoreo**: Seguimiento del uso de memoria en tiempo real
- **Optimización CUDA**: Limpieza de caché GPU cuando está disponible

### Configuración de Rendimiento

- **DPI de imágenes**: Configurado en 100 para balance entre calidad y velocidad
- **Tokens máximos**: 100,000 para generación de HTML completo
- **Procesamiento por lotes**: Una página a la vez para optimizar memoria

## Próximos Pasos

### Módulos en Desarrollo

1. **QuoteGenerator** (`src/automation/quote_generator.py`):
   - Generación automática de nuevas cotizaciones
   - Aplicación de reglas de negocio

2. **TemplateHandler** (`src/cost_sheet/template_handler.py`):
   - Manejo de plantillas de costos
   - Cálculos automáticos de márgenes

3. **WordExporter** (`src/outputs/word_exporter.py`):
   - Exportación de cotizaciones a Word
   - Formateo profesional de documentos

4. **PPTExporter** (`src/outputs/ppt_exporter.py`):
   - Generación de presentaciones PowerPoint
   - Visualización de datos y métricas

### Mejoras Planificadas

- **Interfaz web**: Dashboard para monitoreo y control
- **API REST**: Endpoints para integración con otros sistemas
- **Análisis avanzado**: Machine learning para optimización de propuestas
- **Validación automática**: Verificación de calidad de extracción

## Troubleshooting

### Problemas Comunes

1. **Error de memoria**:
   - Reducir DPI de conversión de PDF
   - Procesar PDFs más pequeños por lotes
   - Usar dispositivo CPU en lugar de GPU

2. **Extracción incompleta**:
   - Verificar calidad de PDFs originales
   - Revisar logs de procesamiento Qwen
   - Ajustar prompts del modelo si es necesario

3. **Errores de fusión de datos**:
   - Verificar que IDs coincidan entre PDFs y Excel
   - Revisar nombres de industrias (case-sensitive)
   - Validar estructura del archivo Excel

### Logs y Debugging

- **Logs detallados**: El sistema imprime información de progreso
- **Archivos intermedios**: Conservados para debugging
- **Métricas de memoria**: Monitoreo en tiempo real

## Contribución

Para contribuir al proyecto:

1. Crear una rama para nuevas funcionalidades
2. Implementar cambios con pruebas unitarias
3. Documentar nuevas funcionalidades
4. Solicitar pull request con descripción detallada

## Licencia

Este proyecto es propiedad de Sincro y está destinado para uso interno de la empresa. 