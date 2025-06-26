# models/ollama_model.py
import aiohttp
from typing import List, Dict, Any
from .base_model import BaseLanguageModel

class OllamaModel(BaseLanguageModel):
    def __init__(self, model_name: str = "llama2", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = "http://localhost:11434"  # Default Ollama URL
        self.session = None

    async def initialize(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Generate a response using Ollama"""
        if not self.session:
            await self.initialize()

        try:
            # Convert messages to Ollama format
            prompt = self._convert_messages_to_prompt(messages)
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                }
            ) as response:
                result = await response.json()
                
                return {
                    'response': {
                        'content': result.get('response', ''),
                        'role': 'assistant'
                    },
                    'model': self.model_name
                }
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to Ollama prompt format"""
        formatted_messages = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                formatted_messages.append(f"System: {content}")
            elif role == 'user':
                formatted_messages.append(f"User: {content}")
            elif role == 'assistant':
                formatted_messages.append(f"Assistant: {content}")
        return "\n".join(formatted_messages)
