import streamlit as st
import ollama

st.title('Gralha')
prompt = st.text_area(label='Diz aí, o que tu quer?')
button = st.button('Okay')

if button:
    if prompt:
        response = ollama.generate(model='llama3.1:8b', prompt=prompt)
        st.markdown(response['response'])

# To run, enter the following in the prompt:  streamlit run .\streamlit_study.py