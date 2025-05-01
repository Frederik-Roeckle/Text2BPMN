from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------
# Naming convention: <ModelName>LLM
# --------------------------------------
        
class MistralLLM(ChatMistralAI):
    def __init__(self, *args, **kwargs):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing MISTRAL_API_KEY in environment variables")
        
        super().__init__(api_key=api_key, *args, **kwargs)

class OpenAILLM(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")

        super().__init__(api_key=api_key, *args, **kwargs)
