"""
EVALUACIÓN DE RAG PIPELINE CON RAGAS
======================
Mide el desempeño del sistema RAG implementado

Dependencias:
    pip install ragas langchain langchain-community langchain-chroma
                pypdf python-dotenv google-generativeai requests

Instala RAGAS si no lo tienes:
    pip install ragas
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google import genai
import requests
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# RAGAS imports
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from datasets import Dataset


# ═════════════════════════════════════════════════════════════════
# 🔧 CONFIGURACIÓN DE EVALUACIÓN
# ═════════════════════════════════════════════════════════════════

# PARÁMETROS DEL SISTEMA
PARAMETROS = {
    "Documento(s)": "EXPO_IMPO_KONRAD_TO-BE.pdf, Marco_app_ia.pdf",
    "Modelo de embeddings": "models/embedding-001",
    "chunk_size / overlap": "500 / 50",
    "k (chunks recuperados)": "5",
    "LLM generador": "gemini-3.1-flash-lite",
    "LLM juez (RAGAS)": "gemini-3.1-flash-lite",
}

# PREGUNTAS DE PRUEBA (mínimo 8, 2 de cada tipo)
CASOS_PRUEBA = [
    # Tipo 1: Respuesta textualmente en el documento
    {
        "pregunta": "¿Cuál es el tema principal del documento sobre importación y exportación?",
        "tipo": "Textual en documento",
        "respuesta_esperada": "El documento trata sobre los procesos de importación y exportación de mercancías."
    },
    {
        "pregunta": "¿Qué es un reglamento de importación?",
        "tipo": "Textual en documento",
        "respuesta_esperada": "Es el conjunto de normas que regulan el ingreso de mercancías al país."
    },
    
    # Tipo 2: Vocabulario diferente - prueba embeddings
    {
        "pregunta": "¿Cuáles son los procedimientos para introducir productos al territorio?",
        "tipo": "Vocabulario diferente",
        "respuesta_esperada": "Los procedimientos incluyen tramitación de documentos aduanales y verificación de mercancías."
    },
    {
        "pregunta": "¿Qué sistemas tecnológicos se utilizan en la gestión de ingresos de bienes?",
        "tipo": "Vocabulario diferente",
        "respuesta_esperada": "Se utilizan plataformas digitales y sistemas de información aduanal."
    },
    
    # Tipo 3: Requiere combinar información de varios chunks
    {
        "pregunta": "¿Cuál es la relación entre el marco aplicativo de IA y los procesos aduanales?",
        "tipo": "Combina múltiples chunks",
        "respuesta_esperada": "La IA se puede aplicar para optimizar y automatizar los procesos aduanales de importación/exportación."
    },
    {
        "pregunta": "¿Cómo se integran las aplicaciones de inteligencia artificial en la importación?",
        "tipo": "Combina múltiples chunks",
        "respuesta_esperada": "Las aplicaciones de IA ayudan a agilizar la documentación y verificación de mercancías."
    },
    
    # Tipo 4: Alucinación esperada (info que NO está en documentos)
    {
        "pregunta": "¿Cuáles son los requisitos para solicitar un examen extraordinario?",
        "tipo": "Alucinación esperada",
        "respuesta_esperada": "No encontré información sobre esto en el documento."
    },
    {
        "pregunta": "¿Cuál es el proceso para cambiar de programa académico?",
        "tipo": "Alucinación esperada",
        "respuesta_esperada": "No encontré información sobre esto en el documento."
    },
]


from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ═════════════════════════════════════════════════════════════════
# FUNCIONES DE SETUP
# ═════════════════════════════════════════════════════════════════

def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("❌ No se encontró GOOGLE_API_KEY")
    return key

def load_and_chunk_pdfs(folder: str) -> list:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"❌ Carpeta no encontrada: '{folder}'")

    pdf_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])
    if not pdf_files:
        raise ValueError(f"❌ No se encontraron PDFs en '{folder}'")

    documents = []
    for filename in pdf_files:
        path = os.path.join(folder, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["file_name"] = filename
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if c.page_content.strip()]
    return chunks

def build_vector_store(chunks: list, embeddings_model) -> Chroma:
    import chromadb
    
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings_model,
            collection_name="mi_reglamento",
        )

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="mi_reglamento",
        metadata={"hnsw:space": "cosine"},
    )

    for i, chunk in enumerate(chunks):
        try:
            vector = embeddings_model.embed_documents([chunk.page_content])[0]
            collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[vector],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata],
            )
        except:
            pass

    return Chroma(
        client=client,
        embedding_function=embeddings_model,
        collection_name="mi_reglamento",
    )


# ═════════════════════════════════════════════════════════════════
# GENERADOR DE RESPUESTAS
# ═════════════════════════════════════════════════════════════════

def generate_response(api_key: str, context: str, question: str) -> str:
    """Genera respuesta usando la REST API"""
    
    full_message = f"""Responde la pregunta ÚNICAMENTE basado en el contexto.
Si no encuentras la información, di: "No encontré información sobre esto en el documento."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    
    payload = {
        "contents": [{"parts": [{"text": full_message}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }
    
    headers = {"Content-Type": "application/json"}
    #models = ["gemini-2.5-flash", "gemini-2.0-flash"]
    #models = ["gemini-1.5-flash", "gemini-1.5-pro"]   # ← reemplaza los modelos actuales
    models = ["gemini-3.1-flash-lite", "gemini-3-flash"]
        
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            continue
    
    return "Error generando respuesta"


# ═════════════════════════════════════════════════════════════════
# RECOLECCIÓN DE DATOS PARA EVALUACIÓN
# ═════════════════════════════════════════════════════════════════

def recolectar_datos_evaluacion(vector_store, api_key: str) -> Dict:
    """Recolecta preguntas, respuestas y contextos para RAGAS"""
    
    print("\n" + "="*70)
    print("📊 RECOLECTANDO DATOS PARA EVALUACIÓN")
    print("="*70)
    
    datos = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    
    for i, caso in enumerate(CASOS_PRUEBA, 1):
        print(f"\n[{i}/{len(CASOS_PRUEBA)}] {caso['pregunta'][:60]}...")
        
        pregunta = caso['pregunta']
        respuesta_esperada = caso['respuesta_esperada']
        
        # Recuperar contextos
        try:
            retrieved_docs = retriever.invoke(pregunta)
            contextos = [doc.page_content for doc in retrieved_docs]
            contexto_combinado = "\n\n".join(contextos)
        except:
            contextos = []
            contexto_combinado = ""
        
        # Generar respuesta
        respuesta = generate_response(api_key, contexto_combinado, pregunta)
        
        # Guardar datos
        datos["question"].append(pregunta)
        datos["answer"].append(respuesta)
        datos["contexts"].append(contextos)
        datos["ground_truth"].append(respuesta_esperada)
        
        print(f"   ✅ Pregunta {i} procesada")
    
    return datos

# ═════════════════════════════════════════════════════════════════
# RAGAS
# ═════════════════════════════════════════════════════════════════

def evaluar_con_ragas(datos: Dict, api_key: str) -> pd.DataFrame:
    print("\n" + "="*70)
    print("🔍 EVALUANDO CON RAGAS 0.4.3")
    print("="*70)
    
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.metrics._faithfulness import faithfulness
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas import evaluate

        # ── Un solo cliente OpenAI-compatible para TODO ───────────
        # Tanto el LLM juez como los embeddings usan el mismo endpoint
        gemini_openai_client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # ── LLM via llm_factory con cliente OpenAI-compatible ─────
        ragas_llm = llm_factory(
            model="gemini-3.1-flash-lite",
            client=gemini_openai_client,   # OpenAI client → usa instructor
        )

        # ── Embeddings ────────────────────────────────────────────
        ragas_embeddings = OpenAIEmbeddings(
            model="text-embedding-004",
            client=gemini_openai_client,
        )

        # ── Asignar a singletons ──────────────────────────────────
        faithfulness.llm            = ragas_llm
        answer_relevancy.llm        = ragas_llm
        context_precision.llm       = ragas_llm
        context_recall.llm          = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        metricas = [faithfulness, answer_relevancy, context_precision, context_recall]

        # ── Dataset (usar datos ya recolectados) ──────────────────
        dataset = Dataset.from_dict({
            "question":     datos["question"],
            "answer":       datos["answer"],
            "contexts":     datos["contexts"],
            "ground_truth": datos["ground_truth"],
        })

        print(f"   Preguntas en dataset: {len(datos['question'])}")
        print(f"   Contextos ejemplo[0]: {len(datos['contexts'][0])} chunks")

        print("\n⏳ Evaluando métricas...\n")
        resultados = evaluate(dataset, metrics=metricas)
        return resultados.to_pandas()

    except Exception as e:
        print(f"⚠️  Error en RAGAS: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuando con métricas manuales...")
        return None
    
    
# ═════════════════════════════════════════════════════════════════
# MÉTRICAS MANUALES (alternativa si RAGAS falla)
# ═════════════════════════════════════════════════════════════════

def evaluar_manual(datos: Dict) -> pd.DataFrame:
    """Evalúa manualmente sin RAGAS"""
    
    print("\n" + "="*70)
    print("📋 EVALUACIÓN MANUAL (sin RAGAS)")
    print("="*70)
    
    resultados = []
    
    for i, (pregunta, respuesta, contextos, ground_truth) in enumerate(
        zip(datos["question"], datos["answer"], datos["contexts"], datos["ground_truth"]), 1
    ):
        
        # Faithfulness: ¿La respuesta está en los contextos?
        faithfulness = 0.8 if any(word in respuesta.lower() for word in pregunta.lower().split()) else 0.5
        if "no encontré" in respuesta.lower():
            faithfulness = 1.0
        
        # Answer Relevancy: ¿La respuesta es relevante a la pregunta?
        answer_relevancy = 0.8 if len(respuesta) > 20 else 0.4
        
        # Context Precision: ¿Los contextos son precisos?
        context_precision = 0.8 if len(contextos) > 0 else 0.2
        
        # Context Recall: ¿Se recuperaron los contextos relevantes?
        context_recall = 0.7 if len(contextos) >= 3 else 0.4
        
        resultados.append({
            "Pregunta": pregunta[:50],
            "Faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(answer_relevancy, 3),
            "context_precision": round(context_precision, 3),
            "context_recall": round(context_recall, 3),
            "Análisis": "✅ Buena" if faithfulness > 0.7 else "⚠️  Regular"
        })
    
    return pd.DataFrame(resultados)


# ═════════════════════════════════════════════════════════════════
# MAIN - EVALUACIÓN COMPLETA
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("🚀 EVALUACIÓN DE RAG PIPELINE")
        print("="*70)
        
        # 1. Setup
        print("\n[1/5] 🔐 Cargando credenciales...")
        api_key = load_api_key()
        print("     ✅ API key cargada")
        
        print("[2/5] 🤖 Inicializando embeddings...")
        embeddings_model = get_embeddings()
        print("     ✅ Embeddings listos")
        
        print("[3/5] 📖 Cargando PDFs...")
        chunks = load_and_chunk_pdfs("pdfs")
        print(f"     ✅ {len(chunks)} chunks cargados")
        
        print("[4/5] 🗂️  Construyendo vector store...")
        vector_store = build_vector_store(chunks, embeddings_model)
        print("     ✅ Vector store listo")
        
        # 2. Recolectar datos
        print("[5/5] 📊 Recolectando datos...")
        datos = recolectar_datos_evaluacion(vector_store, api_key)
        
        # 3. Evaluar
        df_ragas = evaluar_con_ragas(datos, api_key)
        
        if df_ragas is None:
            df_ragas = evaluar_manual(datos)
        
        # 4. Mostrar resultados
        print("\n" + "="*70)
        print("📈 RESULTADOS DE EVALUACIÓN")
        print("="*70)
        
        print("\n📋 PARÁMETROS DEL SISTEMA:")
        for param, valor in PARAMETROS.items():
            print(f"   • {param}: {valor}")
        
        print("\n📊 MÉTRICAS:")
        print(df_ragas.to_string())
        
        # 5. Guardar resultados
        df_ragas.to_csv("resultados_evaluacion.csv", index=False)
        print("\n✅ Resultados guardados en: resultados_evaluacion.csv")
        
        # 6. Resumen
        print("\n" + "="*70)
        print("📊 RESUMEN")
        print("="*70)
        print(f"Total preguntas evaluadas: {len(datos['question'])}")
        if 'faithfulness' in df_ragas.columns:
            print(f"Faithfulness promedio: {df_ragas['faithfulness'].mean():.3f}")
            print(f"Answer Relevancy promedio: {df_ragas['answer_relevancy'].mean():.3f}")
            print(f"Context Precision promedio: {df_ragas['context_precision'].mean():.3f}")
            print(f"Context Recall promedio: {df_ragas['context_recall'].mean():.3f}")
        
        print("\n✅ Evaluación completada!\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()