# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.prompts.chat import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import GPT4AllEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain


# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["inference_api_key"] = os.getenv("inference_api_key")


# loader = PyPDFLoader("Jinnah.pdf")
# docs = loader.load()


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# documents = text_splitter.split_documents(docs)


# db = FAISS.from_documents(
#     documents[:20], GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
# )


# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the following question based on the provided on context.
#     Think step by step before giving detialed answer. I will tip you $1000 if you give correct answer
#     <context>{context}</context>
#     Question: {input}
#     """
# )


# documents_chain = create_stuff_documents_chain(
#     llm=ChatGroq(model="llama3-70b-8192"), prompt=prompt
# )


# retriever = db.as_retriever()

# retrieval_chain = create_retrieval_chain(retriever, documents_chain)

# response = retrieval_chain.invoke({"input": "Tell me about education in england"})


# print(response)


# _________________________________________________________________________________


import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings, JinaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
import tempfile


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["inference_api_key"] = os.getenv("inference_api_key")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")


def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(
        documents, GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
    )

    # Clean up the temporary file
    os.remove(tmp_file_path)

    return db


def ask_query(db, query):
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the provided context.
        Think step by step before giving a detailed answer. I will tip you $1000 if you give the correct answer.
        <context>{context}</context>
        Question: {input}
        """
    )

    documents_chain = create_stuff_documents_chain(
        llm=ChatGroq(model="llama3-70b-8192"), prompt=prompt
    )

    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    response = retrieval_chain.invoke({"input": query})
    return response


st.title("PDF Question Answering App")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your query:")
if st.button("Ask") and pdf_file is not None and query:
    with st.spinner("Processing..."):
        db = load_pdf(pdf_file)
        response = ask_query(db, query)
        st.write(response["answer"])
