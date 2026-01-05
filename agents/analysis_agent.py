from agents.base_agent import BaseAgent
from typing import Dict

class AnalysisAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are an Analysis Agent in a multi-agent thinking system.
Your role is to:
1. Understand the problem and break it down into sub-problems
2. Build a structured thinking and solution plan
3. Identify key components and relationships
4. Define clear objectives and success criteria

Be thorough, systematic, and clear in your analysis. Focus on understanding the problem deeply before proposing solutions."""
        
        super().__init__(
            name="Analysis Agent",
            role="Problem Analysis and Planning",
            system_prompt=system_prompt
        )
    
    async def process(self, context: Dict) -> str:
        problem = context.get("problem", "")
        iteration = context.get("iteration", 1)
        previous_analysis = context.get("analysis", "")
        
        if iteration > 1 and previous_analysis:
            prompt = f"""The problem has been analyzed in previous iterations. Please refine and improve the analysis.

Problem: {problem}

Previous Analysis:
{previous_analysis}

Please provide:
1. An updated and more accurate understanding of the problem
2. Improved breakdown into sub-problems
3. Updated thinking plan
4. Updated assumptions and criteria"""
        else:
            prompt = f"""Analyze the following problem in depth:

Problem: {problem}

Please provide:
1. A clear understanding of what the problem is asking
2. Breakdown into key sub-problems or components
3. A structured plan for approaching this problem
4. Key assumptions and constraints
5. Success criteria for a good solution"""
        
        return await self._call_llm(prompt, context)
