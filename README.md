# ⚒️ RAG_Forge

> **Building Retrieval-Augmented Generation from First Principles**

RAG_Forge is a modular Retrieval-Augmented Generation (RAG) framework designed to help developers understand, build, and extend modern RAG systems from scratch.

Instead of implementing everything inside a single notebook, RAG_Forge organizes every stage of the RAG pipeline into independent, reusable modules following clean software engineering principles.

---

# 🚀 Features

- 📄 Load single or multiple PDF documents
- ✂️ Intelligent document chunking
- 🧠 Local embedding generation using Sentence Transformers
- 🗄️ Persistent ChromaDB Vector Database
- 🔍 Semantic similarity search
- 🤖 Groq LLM integration
- 💬 Context-aware question answering
- 🧩 Fully modular architecture
- 📦 Easy to extend and customize

---

# 📂 Project Structure

```text
RAG_Forge/
│
├── src/
│   └── ragforge/
│       ├── document_loader.py
│       ├── text_splitter.py
│       ├── embedding_model.py
│       ├── vector_database.py
│       ├── retriever.py
│       ├── prompt_builder.py
│       ├── llm.py
│       ├── pipeline.py
│       └── __init__.py
│
├── data/
│   ├── raw_pdfs/
│   └── processed/
│
├── vectorstores/
│   └── chroma/
│
├── app.py
├── config.py
├── requirements.txt
├── .env
└── README.md
```

---

# 🏗️ Architecture

```text
                    PDF Documents
                          │
                          ▼
                 Document Loader
                          │
                          ▼
                 LangChain Documents
                          │
                          ▼
              Recursive Text Splitter
                          │
                          ▼
                     Text Chunks
                          │
                          ▼
              Sentence Transformers
                          │
                          ▼
                     Embeddings
                          │
                          ▼
                  Chroma Vector DB
                          │
                          ▼
                 Semantic Retriever
                          │
                          ▼
                   Prompt Builder
                          │
                          ▼
                      Groq LLM
                          │
                          ▼
                   Generated Answer
```

---

# 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Groq LLM
- PyPDF
- PyMuPDF
- NumPy

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/<your_username>/RAG_Forge.git
```

Move into the project

```bash
cd RAG_Forge
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate Environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Running the Project

```bash
python app.py
```

---

# 📖 How It Works

1. Load PDF documents
2. Split documents into semantic chunks
3. Generate embeddings using Sentence Transformers
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks for a query
6. Build a contextual prompt
7. Generate answers using Groq LLM

---

# 🎯 Current Version

**Version:** `v1.0`

Current capabilities:

- PDF Loading
- Text Chunking
- Embedding Generation
- ChromaDB Storage
- Semantic Retrieval
- Prompt Engineering
- Groq Integration
- Modular RAG Pipeline

---

# 🚧 Future Roadmap

- FAISS Support
- Hybrid Search
- Multiple Document Formats
- Conversation Memory
- Streaming Responses
- Agentic RAG
- LangGraph Integration
- FastAPI Backend
- React Frontend
- Multimodal RAG
- Production Deployment

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository and submit a pull request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Abhishek Bochare**

If you found this project helpful, consider giving it a ⭐ on GitHub.