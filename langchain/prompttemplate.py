import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.title('LangChain Model')
prompt = st.text_area('Digite seu prompt')
button = st.button('Iniciar')

if button:
    if not prompt:
        st.warning('Escreva algum texto antes de iniciar')
        st.stop()

    try:
        llm = ChatOllama(model='llama3.1:8b')
        template = ChatPromptTemplate.from_template("""
Você é um assitente útil e detalhado.
Responda á pergunta abaixo em formato de listas com intes curtos e claros.
Pergunta: {question}
        """)

        chain = template | llm

        response = chain.invoke({"question": prompt})

        st.markdown(response.content)
    except Exception as ex:
        st.error('Erro ao processar modelo')
        st.exception(ex)


