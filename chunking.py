import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

# 📁 Ruta de la carpeta con PDFs
carpeta = "pdfs" 

documents = []

# 🔹 1. Recorrer todos los PDFs de la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith(".pdf"):
        ruta_pdf = os.path.join(carpeta, archivo)
        
        loader = PyPDFLoader(ruta_pdf)
        docs = loader.load()
        
        documents.extend(docs)  # agregar todas las páginas

# 🔹 2. Configurar el splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ' ']
)

# 🔹 3. Crear chunks
chunks = text_splitter.split_documents(documents)

# 🔹 4. Métricas
print(f'Documentos originales (páginas): {len(documents)}')
print(f'Fragmentos generados: {len(chunks)}')
print(f'Factor de expansión: {len(chunks)/len(documents):.1f}x')

# 🔹 5. Ejemplo seguro
index = min(20, len(chunks) - 1)

print(f'\nEjemplo - Fragmento #{index}:')
print(f' Fuente: {chunks[index].metadata.get("source", "?")}')
print(f' Página: {chunks[index].metadata.get("page", "?")}')
print(f' Longitud: {len(chunks[index].page_content)} caracteres')
print(f' Contenido:\n{chunks[index].page_content}')

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)