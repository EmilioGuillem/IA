import requests
import json

class ollama:
    def __ini__(self):
            self.url = "http://localhost:11434/api/generate"
            self.headers = {
                "Content-Type" : "application/json"
            }
            self.data = {
                "model": "llama3.3",
                "prompt": "Test",
                "stream":False
            }
            self.resultQuery = "{}"
            
    def getReponse(self, data:json):
        response = requests.post(url=self.url, headers=self.headers, data=json.dumps(data))
        if response.status_code==200:
            response_text = response.text
            data = json.loads(response_text)
            actual_response = data["response"]
            print(actual_response)
            self.resultQuery = actual_response
        else:
            print("Error: ", response.status_code, response.text)