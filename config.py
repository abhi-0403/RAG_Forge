"""
config.py

Central configuration file for RAG_Forge.

This module stores all configurable settings used across the project,
including paths, chunking parameters, embedding models, vector database
settings, and LLM configuration.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# --------------------------------------------------------------------------
# Load Environment Variables
# --------------------------------------------------------------------------

load_dotenv()

# --------------------------------------------------------------------------
# Project Directories
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstores"
FAISS_DIR = VECTORSTORE_DIR / "faiss"
CHROMA_DIR = VECTORSTORE_DIR / "chroma"

PROMPT_DIR = PROJECT_ROOT / "prompts"

# --------------------------------------------------------------------------
# PDF Processing
# --------------------------------------------------------------------------

SUPPORTED_FILE_TYPES = [".pdf"]

# --------------------------------------------------------------------------
# Text Splitter Configuration
# --------------------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------------------------------------------------------------------------
# Embedding Model
# --------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# Vector Database
# --------------------------------------------------------------------------

VECTOR_DB = "chroma"

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

TOP_K_RESULTS = 3

# --------------------------------------------------------------------------
# Groq Configuration
# --------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"

TEMPERATURE = 0.0

MAX_TOKENS = 1024

# --------------------------------------------------------------------------
# Create Required Directories
# --------------------------------------------------------------------------

for directory in [
    RAW_PDF_DIR,
    PROCESSED_DIR,
    FAISS_DIR,
    CHROMA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)