# Busqueda Semántica de Reglamento (RAG con PDFs + ChromaDB + Gemini)

Este proyecto implementa un sistema de búsqueda semántica tipo RAG (Retrieval Augmented Generation) sobre documentos PDF, utilizando embeddings de Google Gemini y una base vectorial con ChromaDB.

============================================================
OBJETIVO
============================================================

Permitir realizar preguntas en lenguaje natural sobre documentos PDF (reglamentos, programas de asignaturas, etc.) y obtener respuestas basadas únicamente en el contenido del documento, evitando alucinaciones del modelo.

============================================================
ARQUITECTURA DEL SISTEMA
============================================================

1. Carga de documentos PDF
2. División en fragmentos (chunks)
3. Generación de embeddings (Google Gemini)
4. Almacenamiento en ChromaDB
5. Recuperación semántica (Retriever)
6. Construcción de prompt aumentado (RAG)
7. Respuesta basada en contexto

============================================================
INSTALACIÓN
============================================================

Crear entorno virtual:
python -m venv env
env\Scripts\activate

Instalar dependencias:
pip install langchain langchain-community langchain-chroma langchain-google-genai
pip install langchain-text-splitters
pip install sentence-transformers
pip install python-dotenv
pip install scikit-learn
pip install numpy

============================================================
CONFIGURACIÓN API KEY
============================================================

Crear archivo .env:

GOOGLE_API_KEY=tu_api_key_aqui

============================================================
CHUNKING (PDF → fragmentos)
============================================================

- Lee PDFs de la carpeta /pdfs
- Divide en fragmentos de 500 caracteres
- Guarda en chunks.pkl

chunk_size=500
chunk_overlap=50

============================================================
CHROMADB (BASE VECTORIAL)
============================================================

- Usa embeddings de Google Gemini (gemini-embedding-001)
- Convierte chunks en vectores
- Guarda en ./chroma_db

============================================================
RETRIEVER
============================================================

- Recupera los 5 fragmentos más relevantes

search_type="similarity"
k=5

============================================================
PROMPT AUMENTADO (RAG)
============================================================

Estructura:

[SISTEMA]
Instrucciones del modelo

[CONTEXTO]
Fragmentos recuperados

[PREGUNTA]
Pregunta del usuario

[RESPUESTA]
Generada por el LLM

============================================================
BÚSQUEDA SEMÁNTICA MANUAL
============================================================

- embeddings directos
- cosine similarity con sklearn

============================================================
TECNOLOGÍAS
============================================================

- LangChain
- Google Gemini Embeddings
- ChromaDB
- PyPDFLoader
- Scikit-learn
- NumPy
- Python 3.10+

============================================================
PROBLEMAS RESUELTOS
============================================================

- Embeddings inconsistentes
- persist() obsoleto
- sentence-transformers faltante
- vector_store no definido
- base vectorial vacía
- orden de ejecución incorrecto

============================================================
EJECUCIÓN
============================================================

1. python chunking.py
2. python chromaDB.py
3. python retriever.py

============================================================
RESULTADO
============================================================

Responde preguntas como:
- Temas del curso
- Créditos de asignaturas
- Reglas del reglamento
- Información del PDF

Usando SOLO el contenido del documento.

