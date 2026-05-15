import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env reliably even if you run the script from another directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Gemini expects GOOGLE_API_KEY.
# If you already have a key stored under a different name, we map it.
if not os.getenv("GOOGLE_API_KEY"):
    for alt in ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        if os.getenv(alt):
            os.environ["GOOGLE_API_KEY"] = os.environ[alt]
            break

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectors = embeddings.embed_query("Hello, how are you?")

print(vectors)
