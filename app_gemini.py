# Instalar dependencias
# !pip install langchain langchain-community langchain-google-genai langchain-chroma
# !pip install chromadb pypdf python-dotenv scikit-learn matplotlib

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    print(f"[OK] API Key cargada correctamente")
else:
    print("[ERROR] GOOGLE_API_KEY no encontrada.")