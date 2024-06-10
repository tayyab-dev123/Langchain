import argparse
import os
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


st.title("Ollama Chatbot")
input_text = st.text_input("Search the topic u want")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are now chatting with a GROQ model."),
        ("user", "Question:{question}"),
    ]
)


llm = Ollama(model="gemma:2b")


output_parser = StrOutputParser()
chain = prompt | llm | output_parser


if input_text:
    st.write(chain.invoke({"question": input_text}))
