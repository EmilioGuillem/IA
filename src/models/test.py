import openai

openai.api_key = ""

modelos = openai.Model.list()

modelo = "text-davinci-002"
prompt = "¿Cuál es la capital de Francia?"

respuesta = openai.Completion.create(engine=modelo, prompt=prompt, n=1)

text = respuesta.choices[0].text.strip()
print(text)