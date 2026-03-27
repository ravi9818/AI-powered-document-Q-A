
import streamlit as st
import os

from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="AI Document QA", layout="wide")

st.title("📄 AI Document QA System (RAG)")
st.write("Upload TXT, CSV, JSON files and ask questions")

# -------------------------
# LOAD FILES
# -------------------------
def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        path = file.name
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".txt"):
            loader = TextLoader(path)
        elif file.name.endswith(".csv"):
            loader = CSVLoader(path)
        elif file.name.endswith(".json"):
            loader = JSONLoader(path, jq_schema=".[]")
        else:
            continue

        file_docs = loader.load()

        for d in file_docs:
            d.metadata["source"] = file.name

        docs.extend(file_docs)

    return docs

# -------------------------
# PROCESS
# -------------------------
def process_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=200)

model = load_model()

# -------------------------
# ANSWER
# -------------------------
def get_answer(query, docs):
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer ONLY from context.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {query}
    """

    result = model(prompt)
    return result[0]["generated_text"].replace(prompt, "").strip()

# -------------------------
# UI
# -------------------------
files = st.file_uploader("Upload files", type=["txt","csv","json"], accept_multiple_files=True)

if files:
    if st.button("Process"):
        with st.spinner("Processing..."):
            docs = load_documents(files)
            st.session_state.vs = process_docs(docs)
        st.success("Done!")

query = st.text_input("Ask question")

if query and "vs" in st.session_state:
    docs = st.session_state.vs.similarity_search(query, k=3)
    answer = get_answer(query, docs)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for d in docs:
        st.write(d.metadata["source"])
        st.write(d.page_content[:200])
        st.write("---")
