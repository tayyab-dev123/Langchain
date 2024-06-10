import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import os


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = GPT4AllEmbeddings(
        model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"
    )
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50]
    )
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings
    )


st.title("Open Source MultiData Rag Agent")

llm = ChatGroq(api_key=groq_api_key, model="llama3-70b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the provided context only, give the most accurate result.
    <contect>
    {context}
    </context>
    Question: {input}    
    """
)

documment_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

retriver = st.session_state.vectors.as_retriever()

retrival_chain = create_retrieval_chain(retriver, documment_chain)

prompt = st.text_input("Ask Any Question Related to Langsmith")

if prompt:
    start = time.process_time()
    response = retrival_chain.invoke({"input": prompt})
    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
