from typing import final

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate as pt
from langchain_text_splitters import RecursiveCharacterTextSplitter as rct
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

st.title('Resumo de texto com segmentação')

long_text = st.text_area(label='Cole o texto aqui.')
button = st.button('Resumir')

if button:
    if long_text:
        llm = ChatOllama(model='llama3.1:8b')

        template = """
Você é um assitente muito utili, principalmente para resumir textos
Resuma o seguinte texto:
{text}

Me dê um resumo consistente.
        """

        prompt_template = pt(
            template=template,
            input_variables=["text"]
        )

        text_splitter = rct(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = text_splitter.create_documents([long_text])
        summaries = []
        chain = (
            {"text": RunnablePassthrough()}
            | prompt_template
            | llm
        )

        for chunk in chunks:
            st.markdown('### Chunk:')
            st.info(chunk.page_content)


            summary = chain.invoke(chunk.page_content)
            summaries.append(summary.content)

        st.subheader('Resumos parciais')
        st.write('\n\n'.join(summaries))

        final_prompt = f"""
Combine os seguintes resumos dentro de um unico resumo conciso:
{summaries}
"""
        final_summary = llm.invoke(final_prompt)

        st.subheader('Resumo de texto:')
        st.success(final_summary.content)
