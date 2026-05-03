import os

from langchain_community.document_loaders import PyPDFLoader

pdf_dir = 'pdfs'
pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

documents = []
for pdf_file in pdf_files:
    path = os.path.join(pdf_dir, pdf_file)
    loader = PyPDFLoader(path)
    pages = loader.load()
    documents.extend(pages)
    print(f"{pdf_file}: {len(pages)} páginas cargadas")

print(f"\nTotal de páginas cargadas: {len(documents)}")

#explorar la estructura del documento

doc_ejemplo = documents[0]
print('Estructura de un objeto Document:')
print(f"Tipo: {type(doc_ejemplo)}")
print('Atributos: page_content, metadata')
print()
print('Metadatos (metadata): ')
for k, v in doc_ejemplo.metadata.items():
    print(f"  {k}: {v}")
print()
print('primeros 500 caracteres de page_content:')
print('-' * 50)
print(doc_ejemplo.page_content[:500])