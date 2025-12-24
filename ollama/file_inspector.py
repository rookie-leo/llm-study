import os
import tempfile
import time

import ollama
import streamlit as st


def save_temp_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return temp_path, temp_dir

def extract_text(file_path, file_type):
    if file_type == "application/pdf":
        import fitz

        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)

    elif file_type == "text/plain":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Tipo de arquivo não suportado")

st.title("Analisador de Transcrições")
uploaded_file = st.file_uploader("Insisra o documento de consulta", type=["pdf", "txt"])

if uploaded_file is not None:
    file_path, temp_dir = save_temp_file(uploaded_file)
    file_text = extract_text(file_path, uploaded_file.type)

    st.success("Conteúdo extraído")

    prompt = st.text_area("Faça perguntas baseadas no conteudo do documento.")
    button = st.button("Peguntar")

    if button:
        if prompt:
            combined_prompt = (
                'Analise o conteudo do documento:\n\n'
                f'{file_text}\n\n'
                f'Pergunta: {prompt}'
            )
            response = ollama.generate(model='llama3.1:8b', prompt=combined_prompt)

            st.subheader("Resposta")
            st.markdown(response['response'])

    st.info(f'Arquivo temporario salvo em {temp_dir}\n Será deletado automaticamente')
