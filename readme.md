# 📘 GA02: Multi-Document Hybrid RAG Search Engine (with Real-Time Web Search)

---

## 📌 Project Overview

The **GA02 Multi-Document Hybrid RAG Search Engine** is a production-grade AI application that enables users to **query multiple internal documents and live web data simultaneously** through a **chat-based interface**.

This system mirrors real-world **enterprise AI copilots** by combining:

- Semantic document search  
- Retrieval-Augmented Generation (RAG)  
- Real-time web search  
- Citation-aware answer generation  

The application is built using **LangChain, FAISS, Tavily, Groq LLM, and Streamlit**, and supports **PDF/TXT uploads directly inside the chat interface**.

---

## 🎯 Objective

To design and implement a **medium-complexity Hybrid RAG system** that:

- Builds a searchable knowledge base from multiple documents  
- Performs semantic retrieval using FAISS  
- Integrates real-time web search via Tavily  
- Dynamically routes queries (Document / Web / Hybrid)  
- Generates grounded answers with transparent citations  
- Provides a clean, dark, ChatGPT-style UI  

---

## 🧠 Key Features

### ✅ Multi-Document Semantic Search
- Supports **PDF and text documents**
- Recursive chunking with overlap
- Metadata-preserving ingestion

### ✅ Hybrid Retrieval Logic
Queries are classified into:
- 📄 **Document-based**
- 🌐 **Web-based**
- 🔀 **Hybrid (Document + Web)**

### ✅ Real-Time Web Search
- Powered by **Tavily Search API**
- Retrieves up-to-date information
- Web data kept separate from FAISS index

### ✅ Citation-Aware RAG
- Combines document chunks and web snippets
- Generates grounded answers only from retrieved context
- Displays clear document & web citations

### ✅ Chat-Style UI
- Dark theme
- Chat input with **➕ file attachment**
- Search mode toggle (Document / Web / Hybrid)
- Evidence tabs for explainability

---

## 🏗️ System Architecture

User Query
│
▼
Query Classifier
│
├── Document Search (FAISS)
├── Web Search (Tavily)
└── Hybrid Search
│
▼
Context Builder
│
▼
Groq LLM (RAG Answer Generation)
│
▼
Chat UI + Citations


---

## 🧰 Tech Stack (Strictly Enforced)

| Component | Technology |
|---------|-----------|
| Language | Python 3.11 |
| LLM Orchestration | LangChain |
| Vector Database | FAISS |
| LLM Provider | Groq |
| Web Search | Tavily |
| UI | Streamlit |

---

## 📁 Project Structure

GA02_Multi_Document_RAG/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│ ├── ingestion/
│ ├── preprocessing/
│ ├── vectorstore/
│ ├── retrieval/
│ ├── rag/
│ └── models/
└── data/
└── uploads/


---

## 📄 Supported Data Sources

### Local Knowledge Base
- PDF documents
- Text / Markdown files
- Wikipedia pages (LangChain loaders)

### Real-Time Knowledge
- Tavily web search results
- News, recent research, live statistics

---

## 🧪 Query Examples

| Query | Routing |
|-----|--------|
| Explain attention mechanism | 📄 Document |
| Latest developments in LLMs | 🌐 Web |
| How does RAG compare with current AI tools? | 🔀 Hybrid |

---

## ⚙️ Environment Setup

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Create .env file (local only)
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key   # only if OpenAI embeddings are used
▶️ Run Locally
streamlit run app.py
🌐 Deployment (Streamlit Cloud)
Push the project to GitHub

Go to https://share.streamlit.io

Select the repository and app.py

Set Python version to 3.11

Add API keys under Secrets

Deploy 🚀

📊 Evaluation Criteria
Retrieval Relevance – Correct document/web chunks returned

Answer Grounding – No hallucinations

Transparency – Clear separation of document and web evidence

🔮 Future Enhancements
Streaming responses

Chat memory

LLM-based query routing

Per-document summaries

Authentication & access control

Evaluation metrics dashboard

🎓 Learning Outcomes
By completing this project, you demonstrate:

✅ Multi-document RAG system design
✅ Hybrid retrieval (vector + web)
✅ FAISS vector indexing
✅ Tavily real-time search integration
✅ Citation-aware answer generation
✅ End-to-end GenAI application development

👤 Author
Harsh Narain
AI / Data Analytics Practitioner

📜 License
This project is intended for educational and portfolio purposes.

