# models/openai_model.py
import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
from .base_model import BaseLanguageModel

class OpenAIModel(BaseLanguageModel):
    def __init__(self, model_id: str = "gpt-4", temperature: float = 0):
        self.model_id = model_id
        self.temperature = temperature
        self.client = None
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    

    async def initialize(self):
        """Initialize OpenAI client"""
        pass
       

    async def cleanup(self):
        """Cleanup resources"""
        # OpenAI client doesn't need cleanup
        pass

    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Generate a response using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=(tools if tools else None),
                temperature=self.temperature,
            )
            message = response.choices[0].message
            return {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in (message.tool_calls or [])
                ]
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
