import ollama
from ollama import chat

# response = chat(model='llama3.1', messages=[
#     {
#         'role': 'user',
#         'content': 'Why is the sky blue?'
#     },
# ])
# print(response['message']['content'])

# stream = chat(
#     model='llama3.1',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )
#
# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

response = ollama.generate(model='llama3.1:8b', prompt='Why is the sky blue?')

print(response['response'])
