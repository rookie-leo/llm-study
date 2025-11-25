import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.title('Local LLM com LangChain')

user_name = st.text_input('Seu nome:')
topic = st.text_input('Topico:')
question = st.text_input('Duvida:')
instructions = st.text_area('Instruções:')
button = st.button('Iniciar')

if button:
    if not (user_name and topic and question and instructions):
        st.warning('Por favor, preencha todos os campos.')
        st.stop()

    try:
        llm = ChatOllama(model='llama3.1:8b')
        template = ChatPromptTemplate.from_template("""
Você é um assistente útil e detalhado.

Usuário: {name}
Tema: {topic}

Pergunta:
{question}

Instruções adicionais:
{instructions}

Por favor, produza uma resposta detalhada, clara e útil.
        """)

        chain = template | llm
        response = chain.invoke({
            "name": user_name,
            "topic": topic,
            "question": question,
            "instructions": instructions,
        })


        st.markdown(response.content)
    except Exception as ex:
        st.error('Erro ao processar modelo')
        st.exception(ex)
