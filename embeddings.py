import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 🔹 1. Cargar variables del .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# 🔹 2. Ruta de la carpeta
carpeta = "pdfs"  # ⚠️ evita usar caracteres raros como "ç"

documents = []

# 🔹 3. Cargar PDFs
for archivo in os.listdir(carpeta):
    if archivo.endswith(".pdf"):
        ruta_pdf = os.path.join(carpeta, archivo)
        loader = PyPDFLoader(ruta_pdf)
        docs = loader.load()
        
        # opcional (recomendado)
        for d in docs:
            d.metadata["file_name"] = archivo
        
        documents.extend(docs)

# 🔹 4. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# 🔹 5. Embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=API_KEY
)

# 🔹 6. Ejemplo seguro
index = min(20, len(chunks) - 1)

sample_text = chunks[index].page_content
sample_vector = embeddings_model.embed_query(sample_text)

print("Texto de muestra:")
print(sample_text[:120] + "...")

print("\nEmbedding generado:")
print(f"Dimensiones: {len(sample_vector)}")
print(f"Rango: [{min(sample_vector):.4f}, {max(sample_vector):.4f}]")
print(f"Primeros 8 valores: {[round(v, 4) for v in sample_vector[:8]]}")