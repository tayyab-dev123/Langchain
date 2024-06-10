import argparse
import os
from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st


# Load the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


st.title("Groq Chatbot")
input_text = st.text_input("Search the topic u want")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are now chatting with a GROQ model."),
        ("user", "Question:{question}"),
    ]
)
# Initialize the ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
)

prompt1 = ChatPromptTemplate.from_template("Ask {Question}")


output_parser = StrOutputParser()
chain = prompt | llm | output_parser


if input_text:
    st.write(chain.invoke({"question": input_text}))
