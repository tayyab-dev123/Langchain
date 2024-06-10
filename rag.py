import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings


# Load the environment variable
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["inference_api_key"] = os.getenv("inference_api_key")


# loader = PyPDFLoader("Jinnah.pdf")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# documents = text_splitter.split_documents(docs)


# db = Chroma.from_documents(
#     documents[:20], GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
# )


# qurey = "What was the birth name of Jinnah"
# result = db.similarity_search(qurey)
# result = result[0].page_content


def get_similarity_search_result(query):
    # Load the environment variable
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["inference_api_key"] = os.getenv("inference_api_key")

    loader = PyPDFLoader("Jinnah.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents[:20], GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
    )

    result = db.similarity_search(query)
    result = result[0].page_content

    return result


result = get_similarity_search_result("What was the birth name of jinnah?")

print(result)


# def get_similarity_search_result(query):
#     # Load the environment variable
#     load_dotenv()
#     groq_api_key = os.getenv("GROQ_API_KEY")
#     os.environ["LANGCHAIN_TRACING_V2"] = "true"
#     os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#     os.environ["inference_api_key"] = os.getenv("inference_api_key")

#     loader = PyPDFLoader("Jinnah.pdf")
#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = text_splitter.split_documents(docs)

#     db = Chroma.from_documents(
#         documents[:20], GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
#     )

#     result = db.similarity_search(query)
#     result = result[0].page_content

#     return result
