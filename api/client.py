import requests
import streamlit as st


def get_groq_response(input_text):
    response = requests.post(
        "http://localhost:8080/chatgroq/invoke", json={"input": input_text}
    )
    response = response.json()
    # print("Response", response)

    content = response["output"]["messages"][0]["content"]
    return content


def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8080/ollama/invoke", json={"input": input_text}
    )
    response = response.json()
    content = response["output"]["messages"][0]["content"]
    return content


st.title("Fast API")
text_input1 = st.text_input("Write an eassy on ")
text_input2 = st.text_input("Write a poem on ")

if text_input1:
    st.write(get_groq_response(text_input1))

if text_input2:
    st.write(get_ollama_response(text_input2))
