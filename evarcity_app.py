"""
Evarcity — Local RAG Chat App
=============================
A Streamlit chat app that lets you upload a PDF and ask questions about it,
grounded strictly in the document content (Retrieval-Augmented Generation).

Built on the same pipeline as your notebook:
    PDF -> split into chunks -> embed (Ollama) -> store in Chroma
        -> retrieve top chunks -> prompt -> local LLM (Ollama) answers

Requirements (install once):
    pip install streamlit langchain langchain-community langchain-core \
        langchain-text-splitters langchain-ollama langchain-chroma chromadb \
        langgraph pypdf

You also need Ollama running locally with two models pulled:
    ollama pull llama3.2:1b          # or any chat model you have
    ollama pull nomic-embed-text     # embedding model

Run with:
    streamlit run evarcity_app.py
"""

import os
import shutil
import tempfile
import uuid
from typing import List

import streamlit as st
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END


# ----------------------------------------------------------------------
# Page config & basic styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Evarcity",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    .evarcity-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .evarcity-subtitle {
        color: #888;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .stChatMessage { padding: 0.75rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    llm_model = st.text_input(
        "Chat model (Ollama)", value="llama3.2:1b",
        help="Must match a model shown by `ollama list`.",
    )
    embed_model = st.text_input(
        "Embedding model (Ollama)", value="nomic-embed-text:latest",
        help="Used to turn text into searchable vectors.",
    )
    top_k = st.slider("Chunks retrieved per question", 2, 10, 4)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, step=50)

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset document", use_container_width=True):
        for key in ("vectorstore", "retriever", "graph", "doc_name", "messages"):
            st.session_state.pop(key, None)
        st.rerun()


# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("graph", None)
st.session_state.setdefault("doc_name", None)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<p class="evarcity-title">🧠 Evarcity</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="evarcity-subtitle">Upload a PDF and ask questions — answers are grounded only in your document.</p>',
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Upload + Build pipeline
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF to chat with", type=["pdf"])

PROMPT = ChatPromptTemplate.from_template(
    """You are a precise research assistant. Answer the QUESTION using ONLY the CONTEXT below.
- If the answer is not in the context, reply exactly: "I could not find this in the document."
- Keep the answer short and clear. Use simple words.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)


class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str


def build_pipeline(pdf_path: str, llm_model: str, embed_model: str, k: int,
                    chunk_size: int, chunk_overlap: int):
    """Load PDF -> split -> embed -> vectorstore -> retriever -> LangGraph."""
    llm = ChatOllama(model=llm_model, temperature=0)
    embeddings = OllamaEmbeddings(model=embed_model)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Fresh, unique collection per upload so runs don't collide
    collection_name = f"evarcity_{uuid.uuid4().hex[:8]}"
    persist_dir = tempfile.mkdtemp(prefix="evarcity_chroma_")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve(state: RAGState):
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    def generate(state: RAGState):
        context_text = "\n\n".join(d.page_content for d in state["context"])
        messages = PROMPT.invoke({"question": state["question"], "context": context_text})
        response = llm.invoke(messages)
        return {"answer": response.content}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    graph = builder.compile()

    return vectorstore, retriever, graph, len(pages), len(chunks)


if uploaded_file is not None and st.session_state.doc_name != uploaded_file.name:
    with st.spinner(f"📚 Reading and indexing **{uploaded_file.name}**... this can take a minute."):
        tmp_dir = tempfile.mkdtemp(prefix="evarcity_upload_")
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            vectorstore, retriever, graph, n_pages, n_chunks = build_pipeline(
                tmp_path, llm_model, embed_model, top_k, chunk_size, chunk_overlap
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.graph = graph
            st.session_state.doc_name = uploaded_file.name
            st.session_state.messages = []
            st.success(f"✅ Indexed **{uploaded_file.name}** — {n_pages} pages → {n_chunks} chunks.")
        except Exception as e:
            st.error(
                f"Could not build the index. Make sure Ollama is running and the models "
                f"`{llm_model}` / `{embed_model}` are pulled (`ollama list`).\n\nError: {e}"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if st.session_state.doc_name:
    st.caption(f"💬 Chatting with: **{st.session_state.doc_name}**")
else:
    st.info("Upload a PDF above to start chatting.")


# ----------------------------------------------------------------------
# Chat window
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption(f"📄 Sources (pages): {msg['sources']}")

question = st.chat_input(
    "Ask a question about your document..."
    if st.session_state.doc_name else "Upload a PDF first to start chatting"
)

if question:
    if not st.session_state.graph:
        st.warning("Please upload a PDF before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.graph.invoke({"question": question})
                    answer = result["answer"]
                    pages_used = sorted({
                        d.metadata.get("page") for d in result.get("context", [])
                    })
                    st.markdown(answer)
                    if pages_used:
                        st.caption(f"📄 Sources (pages): {pages_used}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": pages_used,
                    })
                except Exception as e:
                    err = f"Something went wrong while generating the answer: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
