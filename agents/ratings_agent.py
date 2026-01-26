from agents.base_agent import BaseAgent
from typing import Dict

class RatingsAgent(BaseAgent):
    def __init__(self):
        system_prompt = """You are a Final Ratings Agent in a multi-agent thinking system.
Your role is to:
1. Evaluate and rate each of the 4 agents (Analysis, Research, Critic, Monitor) based on how well they performed their specific tasks
2. Provide ratings on a scale of 1-10 for each agent
3. Give specific feedback on what each agent did well and what could be improved
4. Assess the overall quality and completeness of each agent's contribution
5. Consider how well each agent fulfilled its specific role and responsibilities

Be fair, constructive, and specific in your ratings. Focus on the quality of the work, not just the length of responses."""
        
        super().__init__(
            name="Ratings Agent",
            role="Final Evaluation and Rating",
            system_prompt=system_prompt
        )
    
    async def process(self, context: Dict) -> str:
        problem = context.get("problem", "")
        analysis = context.get("analysis", "")
        research = context.get("research", "")
        critique = context.get("critique", "")
        monitor = context.get("monitor", "")
        
        prompt = f"""Evaluate and rate each of the 4 agents based on how well they performed their specific roles.

**IMPORTANT: You must provide a numerical rating from 1 to 10 for each agent.**
- 1-3: Poor performance, significant issues
- 4-5: Below average, needs improvement
- 6-7: Average performance, acceptable
- 8-9: Good to excellent performance
- 10: Outstanding, exceptional performance

Original Problem: {problem}

## Analysis Agent Response:
{analysis if analysis else "N/A - No response provided"}

## Research Agent Response:
{research if research else "N/A - No response provided"}

## Critic Agent Response:
{critique if critique else "N/A - No response provided"}

## Monitor Agent Response:
{monitor if monitor else "N/A - No response provided"}

For each agent, you MUST provide:
1. **Rating**: A single number from 1 to 10 (e.g., "8" or "7.5")
2. **Strengths**: What the agent did well
3. **Weaknesses**: Areas where the agent could improve
4. **Role Fulfillment**: How well the agent fulfilled its specific role and responsibilities
5. **Quality Assessment**: Overall quality of the response (clarity, depth, relevance, completeness)

Format your response with clear sections for each agent using ## headers for agent names and ### for sub-sections.
Use **bold** for emphasis on important points.

Structure:
## Final Ratings

### Analysis Agent Rating
- **Rating**: [NUMBER from 1-10]/10
- **Strengths**: ...
- **Weaknesses**: ...
- **Role Fulfillment**: ...
- **Quality Assessment**: ...

### Research Agent Rating
- **Rating**: [NUMBER from 1-10]/10
- **Strengths**: ...
- **Weaknesses**: ...
- **Role Fulfillment**: ...
- **Quality Assessment**: ...

### Critic Agent Rating
- **Rating**: [NUMBER from 1-10]/10
- **Strengths**: ...
- **Weaknesses**: ...
- **Role Fulfillment**: ...
- **Quality Assessment**: ...

### Monitor Agent Rating
- **Rating**: [NUMBER from 1-10]/10
- **Strengths**: ...
- **Weaknesses**: ...
- **Role Fulfillment**: ...
- **Quality Assessment**: ...

### Overall Assessment
[Provide a summary of the overall performance of all agents, including average rating if helpful]"""
        
        return await self._call_llm(prompt, context)
