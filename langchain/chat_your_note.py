import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
import os

# Carrega modelo SpaCy
nlp = spacy.load("en_core_web_md")

# Cria arquivo caso não exista
if not os.path.exists("note.text"):
    with open("note.text", "w", encoding="utf-8") as f:
        f.write("")

st.title("Chat with Your Notes")


# ==============================
# FORM PARA SALVAR NOTAS
# ==============================
with st.form(key="note_form", clear_on_submit=True):
    note = st.text_area("Paste your note here", height=250)
    submit_button = st.form_submit_button("Save")

if submit_button and note:
    # Lê o conteúdo existente
    with open("note.text", "r", encoding="utf-8") as f:
        content = f.read()

    # Adiciona nova nota apenas se não for duplicada
    if note not in content:
        with open("note.text", "a", encoding="utf-8") as f:
            f.write("\n\n" + note)
        st.success("Note saved!")
    else:
        st.info("This note is already saved.")


# ==============================
# CAMPO DE PERGUNTA
# ==============================
question = st.text_input("Enter your question:")
ask_button = st.button("ASK")


# ==============================
# PROCESSAMENTO DA PERGUNTA
# ==============================
if ask_button and question:

    # Carregar notas salvas
    with open("note.text", "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        st.warning("No notes found. Save a note first.")
        st.stop()

    # Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.create_documents([content])

    # Similaridade entre pergunta e cada chunk
    similarities = []
    question_vec = nlp(question)

    for chunk in chunks:
        chunk_vec = nlp(chunk.page_content)
        score = question_vec.similarity(chunk_vec)
        similarities.append((score, chunk.page_content))

    # Seleciona os 3 melhores chunks
    top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

    selected_text = ""
    st.subheader("Most relevant chunks:")
    for score, text in top_chunks:
        st.markdown(f"**Similarity:** {score:.2f}<br>**Text:** {text}", unsafe_allow_html=True)
        selected_text += text + "\n\n"

    # ==============================
    # MODELO LLM
    # ==============================
    llm = OllamaLLM(model="llama3.1:8b")

    template = """
You are a helpful assistant. Answer the question based ONLY on the text below.

QUESTION:
{question}

RELEVANT TEXT:
{text}

ANSWER:
"""

    prompt = PromptTemplate(
        input_variables=["question", "text"],
        template=template
    )

    final_input = prompt.format(question=question, text=selected_text)

    # Executa o modelo
    answer = llm.invoke(final_input)

    st.subheader("Answer:")
    st.markdown(answer)
