from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
from chromadb import ChromaDB

# Conectar a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['nombre_de_tu_base_de_datos']
collection = db['nombre_de_tu_colección']

# Obtener los datos de MongoDB
datos = collection.find()

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

# Generar embeddings para cada documento
datos_con_embeddings = []
for dato in datos:
    texto = dato['campo_de_texto']  # Reemplaza 'campo_de_texto' con el campo que contiene el texto
    embeddings = generar_embeddings(texto)
    datos_con_embeddings.append({'id': str(dato['_id']), 'vector': embeddings, 'metadata': dato})

# Inicializar el cliente de ChromaDB
chroma_db = ChromaDB()

# Crear una colección
collection = chroma_db.create_collection(name='mi_colección')

# Insertar los datos con embeddings en ChromaDB
collection.insert(datos_con_embeddings)

print("Datos insertados en ChromaDB")
