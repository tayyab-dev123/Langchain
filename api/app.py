from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from langserve import add_routes
from fastapi import FastAPI
from langchain_community.llms import Ollama
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
# Load the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(title="ChatAppUsingFastAPI", version="1.0", description="Langchain App")

add_routes(app, ChatGroq(), path="/groq")

model = ChatGroq()
# ollama
llm = Ollama(model="gemma:2b")

prompt1 = ChatPromptTemplate.from_template("Write an eassy about {topic}")
prompt2 = ChatPromptTemplate.from_template("Write a poem about {topic}")


add_routes(app, model | prompt1, path="/chatgroq")

add_routes(app, llm | prompt2, path="/ollama")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
