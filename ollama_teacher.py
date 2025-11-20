import io
import sys

import streamlit as st
import ollama


st.title('Ollama Teacher')
prompt = st.text_area(label= "Write your prompt")
button = st.button(label= "OK")

if button:
    if prompt:
        while True:
            response = ollama.generate(model='llama3.1:8b', prompt=prompt + "\nNote: Output must have is in pt-br and only python code, do not add any other explanation")
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            code = response["response"].replace("python", "").replace("```", "")

            try:
                exec(code)
            except Exception as ex:
                print(f"Error executing code: {ex}")

            sys.stdout = old_stdout
            output = buffer.getvalue()

            if "Error executing generated code" in output:
                prompt = f"""
the following code has this error: {output}
Code:
{code}
"""
                st.markdown(f'Code has Error, fixing... \n{response["response"]} \n {output}')
            else:
                st.markdown(f'Code is working well. \n{response["response"]} \n {output}')
                st.session_state["response"] = response["response"] = response["response"]
                break

question = st.text_area(label= "Write your question")
ask = st.button("Ask")

if ask:
    st.markdown(st.session_state["response"])
    prompt = f"""
Can you explain why we used this code
"{question}"
in the following code:
{st.session_state["response"]}
"""
    answer = ollama.generate(model='llama3.1:8b', prompt=prompt)
    st.markdown(answer["response"])
