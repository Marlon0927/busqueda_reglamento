import os
import shutil
import pickle
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 🔹 1. Cargar API KEY desde .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("No se encontró GOOGLE_API_KEY en el .env")

# 🔹 2. Cargar chunks desde archivo (generado en chunking.py)
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Chunks cargados: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError("No hay chunks. Revisa tu proceso de chunking.")

# 🔹 3. Inicializar modelo de embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=API_KEY
)

# 🔹 4. Configuración de Chroma
PERSIST_DIR = "./chroma_db"

# Limpiar base anterior (opcional)
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"Base de datos anterior eliminada: {PERSIST_DIR}")

# 🔹 5. Crear base vectorial
print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
print("(Esto puede tardar unos minutos...)")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
    collection_name="mi_reglamento",
    collection_metadata={"hnsw:space": "cosine"}
)

# 🔹 6. Validación
total = vector_store._collection.count()

print("\n[OK] Base vectorial creada exitosamente!")
print(f"Ubicación: {PERSIST_DIR}/")
print(f"Fragmentos indexados: {total}")
print("Motor: ChromaDB (cosine similarity)")