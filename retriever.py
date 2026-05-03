import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 1. Cargar PDFs de la carpeta
pdf_folder = 'pdfs'

loader = DirectoryLoader(
    pdf_folder,
    glob='**/*.pdf',
    loader_cls=PyPDFLoader
)
documents = loader.load()

print(f'✓ PDFs cargados: {len(documents)} fragmentos')

# 2. Dividir documentos en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)

print(f'✓ Documentos divididos: {len(split_docs)} chunks')

# 3. Crear vector store desde los documentos
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings_model,
    persist_directory='./chroma_db',
    collection_name="mis_documentos"
)

# 4. Persistir
#vector_store.persist()
#print('✓ Vector store guardado')

# 5. Configurar el retriever
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

# 6. Pregunta de prueba
pregunta = '¿dame 5 frases clave?'

print(f'\nConsulta: "{pregunta}"')
print('Recuperando los 5 fragmentos más relevantes...\n')

# 7. Ejecutar búsqueda
documentos_recuperados = retriever.invoke(pregunta)

print(f'Fragmentos recuperados: {len(documentos_recuperados)}')
print('-' * 60)

# 8. Mostrar resultados
for i, doc in enumerate(documentos_recuperados, 1):
    fuente = os.path.basename(doc.metadata.get('source', 'desconocido'))
    pagina = doc.metadata.get('page', '?')

    print(f'\n[{i}] {fuente} - Pág. {pagina} ({len(doc.page_content)} chars)')
    print(f'    {doc.page_content[:250]}...')
    
#####################################################################################


from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """Eres un asistente académico especializado en la gestión de los programas de asignaturas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no está en el contexto, indica exactamente: "No encontré información sobre esto en el documento."

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta:"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

contexto = "\n\n---\n\n".join(
    f'[Fragmento {i+1} — Pág. {doc.metadata.get("page", "?")}]\\n{doc.page_content}'
    for i, doc in enumerate(documentos_recuperados)
)

prompt_aumentado = prompt_template.invoke({
    "context": contexto,
    "question": pregunta
})

print("Prompt aumentado construido correctamente.")
print(f"  Fragmentos en el contexto: {len(documentos_recuperados)}")
print(f"  Caracteres totales del contexto: {len(contexto)}")
print(f"  Tokens aproximados del contexto: ~{len(contexto)//4}")

print("=" * 65)
print("ESTRUCTURA DEL PROMPT AUMENTADO (RAG)")
print("=" * 65)

print("\n[SISTEMA — Instrucciones para el LLM]")
print("  \"Eres un asistente académico especializado en matemáticas...\"")

print("\n[CONTEXTO — Fragmentos recuperados del vector store]")

for i, doc in enumerate(documentos_recuperados, 1):
    fuente = os.path.basename(doc.metadata.get("source", "?"))
    pagina = doc.metadata.get("page", "?")
    print(f"  [{i}] {fuente} Pág.{pagina} — {len(doc.page_content)} chars")
    print(f"       \"{doc.page_content[:80]}...\"")

print(f"\n[PREGUNTA DEL USUARIO]")
print(f"  \"{pregunta}\"")

print("\n[RESPUESTA]")
print("  (será generada por el LLM en el siguiente paso)")

print("=" * 65)

print("\nEste prompt AUMENTADO se enviará al LLM.")
print("El LLM responderá basándose SOLO en el contexto proporcionado.")