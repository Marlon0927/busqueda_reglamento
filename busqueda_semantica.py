from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app_gemini import API_KEY

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=   API_KEY)

reglamento = [
    "Los alumnos con promedio superior a 4.5 reciben un incentivo económico.",
    "El abandono de los estudios sin previo aviso causa sanción administrativa.",
    "Se pueden pedir exámenes extraordinarios si hay una causa médica comprobada.",
    "La universidad ofrece apoyo financiero para proyectos de investigación."
]

corpus_embeddings = embeddings_model.embed_documents(reglamento)
query = "ayuda de dinero por notas excelentes"
query_embedding = embeddings_model.embed_query(query)

scores = cosine_similarity([query_embedding], corpus_embeddings)
indices_ordenados = np.argsort(scores[0])[::-1]

for idx in indices_ordenados:
    print(f"Score: {scores[0][idx]:.4f} | {reglamento[idx]}")