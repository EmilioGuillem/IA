import os
from pathlib import Path
from dotenv import load_dotenv

class EnvironementVariables:
    def initialize(environement:str = 'C:/Users/Emilio Guillem/Documents/GIT/IA/src/ENV'):
        load_dotenv(dotenv_path=Path(environement), override=True, verbose=True)
    
    def eraseEnv():
        os.environ.clear();