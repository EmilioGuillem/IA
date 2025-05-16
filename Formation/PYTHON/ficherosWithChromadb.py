import os
from transformers import AutoTokenizer, AutoModel
import torch
from chromadb import ChromaDB

# Directorio donde se encuentran los ficheros
directorio = 'ruta/a/tu/directorio'

# Leer el contenido de los ficheros
ficheros = []
for nombre_fichero in os.listdir(directorio):
    ruta_fichero = os.path.join(directorio, nombre_fichero)
    with open(ruta_fichero, 'r', encoding='utf-8') as fichero:
        contenido = fichero.read()
        ficheros.append({'nombre': nombre_fichero, 'contenido': contenido})

# Cargar el modelo y el tokenizador
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Función para generar embeddings
def generar_embeddings(texto):
    inputs = tokenizer(texto, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.tolist()

# Generar embeddings para cada fichero
ficheros_con_embeddings = []
for fichero in ficheros:
    embeddings = generar_embeddings(fichero['contenido'])
    ficheros_con_embeddings.append({'id': fichero['nombre'], 'vector': embeddings, 'metadata': {'contenido': fichero['contenido']}})

# Inicializar el cliente de ChromaDB
chroma_db = ChromaDB()

# Crear una colección
collection = chroma_db.create_collection(name='mi_colección')

# Insertar los datos con embeddings en ChromaDB
collection.insert(ficheros_con_embeddings)

print("Datos insertados en ChromaDB")
