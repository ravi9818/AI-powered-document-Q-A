# 📄 AI-Powered Document Question Answering System (RAG)

## 🚀 Overview

This project is a **Generative AI-based Document Question Answering System** that allows users to upload documents (TXT, CSV, JSON) and ask questions.
The system retrieves relevant information using **semantic search** and generates accurate answers using **Retrieval-Augmented Generation (RAG)**.

---

## 🎯 Features

* 📂 Multi-format support (TXT, CSV, JSON)
* 🔍 Semantic search using FAISS
* 🧠 Embedding generation with HuggingFace
* 🤖 LLM-based answer generation (Flan-T5)
* 📊 Source-based answers (traceability)
* 🌐 Interactive UI using Streamlit
* ⚡ Fully offline (no API required)

---

## 🛠️ Tech Stack

* Python
* LangChain
* FAISS (Vector Database)
* HuggingFace Transformers
* Sentence Transformers
* Streamlit

---

## 🧠 How It Works (RAG Pipeline)

1. Load documents (TXT/CSV/JSON)
2. Split text into chunks
3. Generate embeddings
4. Store embeddings in FAISS
5. Accept user query
6. Retrieve relevant chunks
7. Generate answer using LLM
8. Display answer with sources

---

## 📁 Project Structure

```
document-qa-rag/
│
├── data/                  # Input documents
├── app.py                 # Streamlit UI
├── notebook.ipynb         # Development notebook
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/document-qa-rag.git
cd document-qa-rag

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Evaluation Metrics

* Answer relevance
* Context accuracy
* Response completeness
* Prompt effectiveness
* Handling irrelevant queries

---

## ⚠️ Limitations

* Performance depends on chunk quality
* Limited context window
* May struggle with very large datasets
* Lightweight LLM may reduce answer quality

---

## 🔥 Future Improvements

* Multi-document chat history
* Hybrid search (BM25 + vector)
* Better UI/UX (chat interface)
* Model upgrade (LLaMA / Mistral)
* Cloud deployment

---

## 🎤 Use Cases

* 📘 Student learning assistant
* 🏢 Company knowledge base
* 📄 HR policy assistant
* ⚖️ Legal/compliance document QA
* 💬 Customer support system

---

## 🧪 Dataset

You can use:

* Kaggle Document QA Dataset
* SQuAD Dataset
* MLQA Dataset

---

## 💡 Key Concepts

* Generative AI
* Prompt Engineering
* Embeddings
* Semantic Search
* Retrieval-Augmented Generation (RAG)

---

## 👨‍💻 Author

**Ravi Sharma**

