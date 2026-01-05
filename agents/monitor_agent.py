from agents.base_agent import BaseAgent
from typing import Dict

class MonitorAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a Monitor Agent in a multi-agent thinking system.
Your role is to:
1. Supervise the thinking process
2. Identify if the process is stuck in loops or deviating from the topic
3. Decide whether another iteration is needed or if the process can stop
4. Ensure the process is making progress toward meaningful insights

Be decisive and clear. Your decision should be based on whether meaningful progress is being made."""
        
        super().__init__(
            name="Monitor Agent",
            role="Process Supervision and Decision",
            system_prompt=system_prompt
        )
    
    async def process(self, context: Dict) -> str:
        problem = context.get("problem", "")
        all_responses = context.get("all_responses", {})
        
        prompt = f"""Review the complete thinking process and provide a summary assessment.

Problem: {problem}

Complete analysis:
Analysis: {all_responses.get('analysis', 'N/A')[:500]}...
Research: {all_responses.get('research', 'N/A')[:500]}...
Critique: {all_responses.get('critique', 'N/A')[:500]}...

Please provide:
1. Overall assessment of the thinking process quality
2. Key strengths of the approach
3. Any concerns or limitations identified
4. Confidence level in the solution
5. Summary of the process completion"""
        
        return await self._call_llm(prompt, context)
