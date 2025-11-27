import streamlit as st
import spacy
from langchain_ollama import ChatOllama as Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

nlp = spacy.load('en_core_web_md')

st.title('LangChain Model: Resume textos longos')

long_text = st.text_area('Cole aqui o texto a ser resumido')
question = st.text_input("Faça uma pergunta sobre o texto")
button = st.button('Perguntar')

if button:
    if long_text and question:
        llm = Ollama(model="llama3.1:8b")
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'Você é um assistente muito util. Use apenas o texto que lhe foi fornecido'),
            'human', 'QUESTION:\n{question}\n\nTEXT:\n{context}'
        ])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.split_text(long_text)
        similarities = []

        for chunk in chunks:
            score = nlp(question).similarity(nlp(chunk))
            similarities.append((score, chunk))

        top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

        st.subheader('Cortes mais relevantes')
        context = ""

        for score, chunk in top_chunks:
            st.markdown(f'**Similaridade:** `{score:.2f}`%')
            st.write(chunk)
            context += chunk + '\n\n'

        chain = prompt | llm
        response = chain.invoke({
            'question': question,
            'context': context
        })

        st.subheader('Resposta:')
        st.markdown(response.content)

