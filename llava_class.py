import ollama

response = ollama.chat(model='llava:7b',
                       messages=[{
                           'role': 'user',
                           'content': 'Describe the image.',
                           'images': ['image.jpg']
                       }])
print(response['message']['content'])
