from abc import ABC, abstractmethod
from typing import Dict, Optional
import os
from openai import OpenAI

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print(f"Warning: OpenAI API key not found. {self.name} will use mock responses.")
    
    @abstractmethod
    async def process(self, context: Dict) -> str:
        """Process the context and return a response"""
        pass
    
    async def _call_llm(self, user_prompt: str, context: Optional[Dict] = None) -> str:
        """Call OpenAI API or return mock response"""
        if not self.client:
            print(f"Warning: {self.name} has no OpenAI client, using mock response")
            return self._mock_response(user_prompt)
        
        try:
            # Skip GPT-5.2 for now - it's not available and causes hanging
            # Go straight to GPT-4o which is reliable
            print(f"{self.name}: Calling GPT-4o API...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            result = response.choices[0].message.content
            print(f"{self.name}: Successfully got response from GPT-4o (length: {len(result)})")
            return result
                
        except Exception as e:
            print(f"Error calling LLM for {self.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_response(user_prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Return a mock response when API is not available"""
        return f"[Mock {self.name} Response] Based on the context: {prompt[:100]}..."

