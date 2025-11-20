import streamlit as st
import ollama
import tempfile

st.title("Image Describer")

uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    response = ollama.chat(
        model='llava:7b',
        messages=[{
            'role': 'user',
            'content': 'Descreva a imagem em pt-br.',
            'images': [temp_path]
        }]
    )

    st.markdown(response['message']['content'])
