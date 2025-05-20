import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader


import tempfile
import os

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 Ask Questions About Your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embedding)

        flan_pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)
        llm = HuggingFacePipeline(pipeline=flan_pipe)

        st.success("PDF processed. You can now ask questions!")

        # Input from user
        user_question = st.text_input("Ask a question:")
        if user_question:
            retrieved_docs = vectordb.similarity_search(user_question, k=3)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            with st.spinner("Generating answer..."):
                prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {user_question}"
                answer = llm(prompt)
                st.markdown(f"**Answer:** {answer}")
