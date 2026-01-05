from agents.base_agent import BaseAgent
from typing import Dict

class CriticAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a Critic Agent in a multi-agent thinking system.
Your role is to:
1. Critically evaluate the proposed solution
2. Identify weaknesses, contradictions, or false assumptions
3. Find potential risks or possible issues
4. Suggest improvements and alternative perspectives

Be thorough, constructive, and honest in your critique. Your goal is to improve the quality of thinking."""
        
        super().__init__(
            name="Critic Agent",
            role="Critical Evaluation and Improvement",
            system_prompt=system_prompt
        )
    
    async def process(self, context: Dict) -> str:
        problem = context.get("problem", "")
        analysis = context.get("analysis", "")
        research = context.get("research", "")
        previous_critique = context.get("critique", "")
        iteration = context.get("iteration", 1)
        
        if iteration > 1 and previous_critique:
            prompt = f"""Critically evaluate the updated thinking and approach.

Problem: {problem}

Current Analysis:
{analysis}

Current Research:
{research}

Previous Critique:
{previous_critique}

Please provide:
1. Evaluation of improvements made since last iteration
2. Remaining weaknesses or gaps
3. New contradictions or issues identified
4. Updated risk assessment
5. Specific suggestions for further improvement"""
        else:
            prompt = f"""Critically evaluate the current thinking and approach.

Problem: {problem}

Analysis:
{analysis}

Research:
{research}

Please provide:
1. Weaknesses or gaps in the current approach
2. Contradictions or logical inconsistencies
3. False assumptions or missing considerations
4. Potential risks or unintended consequences
5. Alternative perspectives or approaches to consider
6. Specific suggestions for improvement"""
        
        return await self._call_llm(prompt, context)
