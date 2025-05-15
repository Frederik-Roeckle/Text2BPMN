from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types
import os
from google.genai.types import GenerateContentConfig, ThinkingConfig

from google.genai import Client



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

class GeminiLLM(ChatGoogleGenerativeAI):
    def __init__(self, *args, **kwargs):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables")

        # required
        kwargs.setdefault("model", "gemini-2.5-pro-preview-05-06")
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("google_api_key", api_key)

        # ← here’s the only new line you need:
        kwargs.setdefault("thinking_budget", 4000)

        super().__init__(*args, **kwargs)
