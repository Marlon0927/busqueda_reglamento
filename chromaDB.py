import os
from langchain_chroma import Chroma
import shutil

PERSIST_DIR = './chroma_db_notebook_2'

# Limpiar base de datos anterior para re-ejecutar el notebook limpiamente
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f'Base de datos anterior eliminada: {PERSIST_DIR}')

print(f'Indexando {len(chunks)} fragmentos en ChromaDB...')
print('(Esto puede tardar unos minutos — se generan embeddings para cada fragmento)')

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
    collection_name="mi_reglamento",
    collection_metadata={'hnsw:space': 'cosine'}
)

total = vector_store._collection.count()
print()
print('[OK] Base vectorial creada exitosamente!')
print(f'  Ubicación en disco:     {PERSIST_DIR}/')
print(f'  Fragmentos indexados:   {total}')
print(f'  Motor de búsqueda:      ChromaDB (similitud coseno)')
print(f'  Dimensión de vectores:  768')