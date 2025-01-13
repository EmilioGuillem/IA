import chromadb
from chromadb.utils import embedding_functions
import numpy as np

class chromadbConnexion:
    
    def __ini__(self, storePath):
        self.store = storePath
        self.client = None
        self.results = None
        self.collection = None
        self.ef = None
        try:
            self.client = chromadb.PersistentClient(path=storePath)
        except Exception as e:
            print(e)

    def getCollection(self, data:str):
        self.collection = self.client.get_collection(name = data)
        return self.collection
    
    def getDoc(self, data, collection:str): # type: ignore
        newCollection = self.getCollection(collection)
        collectionEmb = self.getCollection("Embeddings")
        self.resultsText = newCollection.query(
            query_texts = data,
            n_results=2,
        
        )
        self.results = collectionEmb.query(
            query_embeddings= data     
        )
    
    def getEmbedding(self, data:[]): # type: ignore
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-ada-002")
        embeddings_data = openai_ef(data)
        self.ef = openai_ef
        return embeddings_data
    
    def createCollection(self, name:str):
        self.collection = self.client.get_or_create_collection(name=name, embedding_function=self.ef)
    
    def insert(self, collection, document, metadataDoc, idDoc):
        newCol = self.getCollection(collection)
        newEmbedding = self.getEmbedding(document)
        collectionEmb = self.client.get_or_create_collection(name="Embeddings", embedding_function=self.ef)
        try:
            newCol.add(
                documents=document,
                metadatas=metadataDoc,
                ids=idDoc,
                embeddings=newEmbedding
            )
            collectionEmb.add(
                ids=idDoc,
                embeddings=newEmbedding
            )
        except Exception as e:
            print(e)