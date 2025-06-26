# models/base_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLanguageModel(ABC):
    """Base class for language models"""
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Generate a response from the model"""
        pass

    @abstractmethod
    async def initialize(self):
        """Initialize the model"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass
