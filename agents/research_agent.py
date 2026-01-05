from agents.base_agent import BaseAgent
from typing import Dict

class ResearchAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a Research Agent in a multi-agent thinking system.
Your role is to:
1. Gather relevant knowledge, existing information, professional assumptions
2. Collect theoretical insights or external data (if needed)
3. Identify relevant frameworks, approaches, or methodologies
4. Find examples, case studies, or patterns
5. Note gaps in knowledge that might need further investigation

Be comprehensive in your research approach. Think about what information would be valuable for solving this problem."""
        
        super().__init__(
            name="Research Agent",
            role="Knowledge Gathering and Context",
            system_prompt=system_prompt
        )
    
    async def process(self, context: Dict) -> str:
        problem = context.get("problem", "")
        analysis = context.get("analysis", "")
        iteration = context.get("iteration", 1)
        previous_research = context.get("research", "")
        
        if iteration > 1 and previous_research:
            prompt = f"""Research has been conducted in previous iterations. Please refine and expand the research.

Problem: {problem}

Previous Analysis:
{analysis}

Previous Research:
{previous_research}

Please provide:
1. Additional relevant knowledge and information
2. New theoretical insights or frameworks
3. Additional examples or case studies
4. Updated professional assumptions
5. Any new gaps or areas needing investigation"""
        else:
            prompt = f"""Based on the problem and analysis, gather relevant knowledge and context.

Problem: {problem}

Analysis:
{analysis}

Please provide:
1. Relevant theoretical frameworks or approaches that apply
2. Key concepts, principles, or methodologies that could be useful
3. Important assumptions or constraints to consider
4. Any relevant examples, case studies, or patterns
5. Gaps in knowledge that might need external data or further investigation
6. Professional insights or best practices related to this type of problem"""
        
        return await self._call_llm(prompt, context)
