import streamlit as st
# from langchain_classic.llms import Ollama
from langchain_ollama import ChatOllama as Ollama # pip install langchain langchain-communit

st.title('LangChain Model')

prompt = st.text_area(label="Escreva seu prompt.")
button = st.button("OK")

if button:
    if prompt:
        llm = Ollama(model='llama3.1:8b')
        response = llm.invoke(prompt)
        st.markdown(response)
