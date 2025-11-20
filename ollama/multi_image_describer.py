import os

import ollama
import streamlit as st



def save_uploaded_file(uploaded_file):
    save_path = os.getcwd()
    file_path = os.path.join(save_path, uploaded_file.name)

    with open(file_path, 'wb') as file:
        file.write(uploaded_file.getbuffer())

    return st.success(f'Saved file: {uploaded_file.name} to {save_path}')

st.title('Multi Image Describer')

uploaded_files = st.file_uploader('Choose an image', accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

print(f'Uploaded files: {uploaded_files}')

if len(uploaded_files) > 0:
     for uploaded_file in uploaded_files:
         save_uploaded_file(uploaded_file)
         print(f'Uploaded files name: {uploaded_file.name}')
         print(f'Type of uploaded file: {type(uploaded_file.name)}')

         st.image(uploaded_file, caption='Uploaded image.', use_column_width=True)

         response = ollama.chat(model='llava:7b',
                                messages=[{
                                  'role': 'user',
                                    'content': 'Descreva a imagem em pt-br.',
                                    'images': [uploaded_file.name]
                                }])
         st.markdown(response['message']['content'])
         print(response['message']['content'])
