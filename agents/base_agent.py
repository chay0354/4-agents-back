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
            # Combine system prompt and user prompt for GPT-5.2 Responses API
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"
            
            # Try GPT-5.2 with Responses API first
            if hasattr(self.client, 'responses'):
                try:
                    print(f"{self.name}: Calling GPT-5.2 API...")
                    response = self.client.responses.create(
                        model="gpt-5.2",
                        input=combined_input,
                        reasoning={
                            "effort": "none"
                        }
                    )
                    print(f"{self.name}: Got response, type: {type(response)}")
                    
                    # Extract text from the response
                    # The Responses API structure can vary, so we try multiple paths
                    result = None
                    
                    # Method 1: Try response.output.content[0].text (most common structure)
                    if hasattr(response, 'output') and response.output:
                        output = response.output
                        if hasattr(output, 'content') and output.content:
                            if isinstance(output.content, list) and len(output.content) > 0:
                                first_item = output.content[0]
                                if hasattr(first_item, 'text'):
                                    result = first_item.text
                                elif isinstance(first_item, dict) and 'text' in first_item:
                                    result = first_item['text']
                        elif hasattr(output, 'text'):
                            result = output.text
                    
                    # Method 2: Direct content access
                    if not result and hasattr(response, 'content'):
                        content = response.content
                        if isinstance(content, list) and len(content) > 0:
                            first_item = content[0]
                            if hasattr(first_item, 'text'):
                                result = first_item.text
                            elif isinstance(first_item, dict) and 'text' in first_item:
                                result = first_item['text']
                    
                    # Method 3: Try to get text directly from response
                    if not result and hasattr(response, 'text'):
                        result = response.text
                    
                    # Method 4: Try model_dump() to inspect the structure
                    if not result and hasattr(response, 'model_dump'):
                        try:
                            response_dict = response.model_dump()
                            # Look for nested text in various possible locations
                            if 'output' in response_dict:
                                output_dict = response_dict['output']
                                if isinstance(output_dict, dict):
                                    if 'content' in output_dict and isinstance(output_dict['content'], list) and len(output_dict['content']) > 0:
                                        first_item = output_dict['content'][0]
                                        if isinstance(first_item, dict) and 'text' in first_item:
                                            result = first_item['text']
                                    elif 'text' in output_dict:
                                        result = output_dict['text']
                            if not result and 'content' in response_dict:
                                content = response_dict['content']
                                if isinstance(content, list) and len(content) > 0:
                                    first_item = content[0]
                                    if isinstance(first_item, dict) and 'text' in first_item:
                                        result = first_item['text']
                            if not result and 'text' in response_dict:
                                result = response_dict['text']
                        except Exception as e:
                            print(f"{self.name}: Error in model_dump extraction: {e}")
                    
                    # Method 5: Try string representation and regex extraction as last resort
                    if not result:
                        try:
                            import re
                            response_str = str(response)
                            # Look for text='...' or text="..." patterns
                            text_match = re.search(r"text=['\"](.*?)['\"]", response_str, re.DOTALL)
                            if text_match:
                                result = text_match.group(1)
                        except Exception as e:
                            print(f"{self.name}: Error in regex extraction: {e}")
                    
                    # If we got a result, clean it up and return
                    if result:
                        result_str = str(result).strip()
                        # Clean up any escaped characters
                        result_str = result_str.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                        
                        if len(result_str) > 10:  # Ensure we got meaningful content
                            print(f"{self.name}: Successfully got response from GPT-5.2 (length: {len(result_str)})")
                            return result_str
                        else:
                            print(f"{self.name}: GPT-5.2 response too short ({len(result_str)} chars), falling back to GPT-4o")
                    else:
                        print(f"{self.name}: Could not extract text from GPT-5.2 response, falling back to GPT-4o")
                        # Debug: print response structure
                        print(f"{self.name}: Response attributes: {dir(response)}")
                        if hasattr(response, 'model_dump'):
                            print(f"{self.name}: Response dict: {response.model_dump()}")
                        
                except AttributeError as e:
                    print(f"{self.name}: responses.create not available: {e}, falling back to GPT-4o")
                except Exception as e:
                    print(f"{self.name}: GPT-5.2 API error: {type(e).__name__}: {e}, falling back to GPT-4o")
                    import traceback
                    traceback.print_exc()
            
            # Fallback to GPT-4o using Chat Completions API
            print(f"{self.name}: Using GPT-4o (Chat Completions API)...")
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

