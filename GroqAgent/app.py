import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.chains.conversation.base import ConversationChain
import tempfile


load_dotenv()


def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    db = Chroma.from_documents(
        collection_name="history",
        documents=documents[:20],
        embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
    )

    # Clean up the temporary file
    os.remove(tmp_file_path)

    return db


def ask_query(db, query):

    # Create your retriever
    retriever = db.as_retriever()

    # Create your VectorStoreRetrieverMemory
    vectorstore_retriever_memory = VectorStoreRetrieverMemory(retriever=retriever)

    vectordb_memory_chain = ConversationChain(
        llm=ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        memory=vectorstore_retriever_memory,
        verbose=True,
    )

    response = vectordb_memory_chain.invoke({"input": query})

    # Save the input query and output response to the memory
    vectorstore_retriever_memory.save_context(
        {"input": query}, {"output": response["response"]}
    )

    # Save to session state history
    st.session_state.history.append({"query": query, "response": response})

    return response


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.title("PDF Question Answering App")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your query:")
if st.button("Ask") and pdf_file is not None and query:
    with st.spinner("Processing..."):
        db = load_pdf(pdf_file)
        response = ask_query(db, query)
        st.write(response["response"])

# Display previous questions and answers
if st.session_state.history:
    st.subheader("Previous Questions and Answers")
    for i, interaction in enumerate(st.session_state.history):
        st.write(f"**Q{i+1}:** {interaction['query']}")
        st.write(f"**A{i+1}:** {interaction['response']['response']}")
