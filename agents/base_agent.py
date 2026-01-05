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
            # Combine system prompt and user prompt for the new API format
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"
            
            print(f"{self.name}: Calling GPT-5.2 API...")
            
            # Try the new responses.create API first (for GPT-5.2)
            if hasattr(self.client, 'responses'):
                try:
                    print(f"{self.name}: Attempting GPT-5.2 API call...")
                    response = self.client.responses.create(
                        model="gpt-5.2",
                        input=combined_input,
                        reasoning={
                            "effort": "none"
                        }
                    )
                    print(f"{self.name}: Got response, type: {type(response)}")
                    
                    # Extract text from the nested response structure
                    result = None
                    
                    # The response structure is: Response -> content (list) -> ResponseOutputText -> text
                    # Or: ResponseOutputMessage -> content (list) -> ResponseOutputText -> text
                    try:
                        # Try to get content attribute
                        content_attr = None
                        if hasattr(response, 'content'):
                            content_attr = response.content
                        elif hasattr(response, 'output'):
                            # Sometimes it's in output
                            output = response.output
                            if hasattr(output, 'content'):
                                content_attr = output.content
                        
                        if content_attr:
                            # content is a list of ResponseOutputText objects
                            if isinstance(content_attr, list) and len(content_attr) > 0:
                                first_content = content_attr[0]
                                if hasattr(first_content, 'text'):
                                    result = first_content.text
                                    print(f"{self.name}: Found response in content[0].text (length: {len(result)})")
                                elif hasattr(first_content, 'content'):
                                    result = first_content.content
                                    print(f"{self.name}: Found response in content[0].content")
                                elif isinstance(first_content, dict) and 'text' in first_content:
                                    result = first_content['text']
                                    print(f"{self.name}: Found response in content[0]['text']")
                    except Exception as e:
                        print(f"{self.name}: Error extracting from content: {e}")
                    
                    # Fallback to other methods
                    if not result:
                        if hasattr(response, 'output') and response.output:
                            result = response.output
                            print(f"{self.name}: Found response in 'output' attribute")
                        elif hasattr(response, 'text') and response.text:
                            result = response.text
                            print(f"{self.name}: Found response in 'text' attribute")
                        elif hasattr(response, 'response') and response.response:
                            result = response.response
                            print(f"{self.name}: Found response in 'response' attribute")
                        elif hasattr(response, 'choices') and len(response.choices) > 0:
                            choice = response.choices[0]
                            if hasattr(choice, 'text'):
                                result = choice.text
                            elif hasattr(choice, 'content'):
                                result = choice.content
                            elif hasattr(choice, 'message'):
                                result = choice.message.content if hasattr(choice.message, 'content') else str(choice.message)
                            else:
                                result = str(choice)
                            print(f"{self.name}: Found response in 'choices'")
                    
                    # Last resort: try to convert to dict and extract text
                    if not result:
                        try:
                            if hasattr(response, 'model_dump'):
                                response_dict = response.model_dump()
                                # Look for nested content structure
                                if 'content' in response_dict and isinstance(response_dict['content'], list) and len(response_dict['content']) > 0:
                                    first_item = response_dict['content'][0]
                                    if isinstance(first_item, dict):
                                        # Try different possible keys for text
                                        result = first_item.get('text') or first_item.get('content') or first_item.get('message')
                                    else:
                                        # If it's an object, try to get text attribute
                                        if hasattr(first_item, 'text'):
                                            result = first_item.text
                                        else:
                                            result = str(first_item)
                                else:
                                    result = response_dict.get('output') or response_dict.get('text')
                                    
                                # If we got a dict/object string, try to extract text from it
                                if result and isinstance(result, str) and 'ResponseOutputText' in result:
                                    import re
                                    # Try to extract text='...' or text="..."
                                    text_match = re.search(r"text=['\"](.*?)['\"]", result, re.DOTALL)
                                    if text_match:
                                        result = text_match.group(1)
                            elif hasattr(response, 'dict'):
                                response_dict = response.dict()
                                if 'content' in response_dict and isinstance(response_dict['content'], list) and len(response_dict['content']) > 0:
                                    first_item = response_dict['content'][0]
                                    if isinstance(first_item, dict) and 'text' in first_item:
                                        result = first_item['text']
                                    else:
                                        result = response_dict.get('output') or response_dict.get('text')
                                else:
                                    result = response_dict.get('output') or response_dict.get('text')
                            else:
                                # If str(response) gives us the object representation, try to parse it
                                response_str = str(response)
                                if 'ResponseOutputText' in response_str or 'text=' in response_str:
                                    import re
                                    text_match = re.search(r"text=['\"](.*?)['\"]", response_str, re.DOTALL)
                                    if text_match:
                                        result = text_match.group(1)
                                    else:
                                        result = response_str
                                else:
                                    result = response_str
                        except Exception as e:
                            print(f"{self.name}: Error extracting from dict: {e}")
                            import traceback
                            traceback.print_exc()
                            # Last resort: try regex on string representation
                            try:
                                import re
                                response_str = str(response)
                                text_match = re.search(r"text=['\"](.*?)['\"]", response_str, re.DOTALL)
                                if text_match:
                                    result = text_match.group(1)
                                else:
                                    result = response_str
                            except:
                                result = str(response)
                    
                    if result:
                        result_str = str(result).strip()
                        # Clean up any escaped newlines or quotes
                        result_str = result_str.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                        
                        if len(result_str) > 10:  # Make sure we got actual content
                            print(f"{self.name}: Successfully got response from GPT-5.2 (length: {len(result_str)})")
                            return result_str
                        else:
                            print(f"{self.name}: Response too short ({len(result_str)} chars), falling back to GPT-4o")
                    else:
                        print(f"{self.name}: Could not extract text from response, falling back to GPT-4o")
                except AttributeError as e:
                    print(f"{self.name}: responses.create not available: {e}, falling back")
                except Exception as e:
                    print(f"{self.name}: GPT-5.2 API error: {type(e).__name__}: {e}, falling back")
                    import traceback
                    traceback.print_exc()
            
            # Fallback to standard chat.completions API
            print(f"{self.name}: Using standard chat.completions API (GPT-4o)")
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

