from agents.analysis_agent import AnalysisAgent
from agents.research_agent import ResearchAgent
from agents.critic_agent import CriticAgent
from agents.monitor_agent import MonitorAgent
from typing import Dict, AsyncGenerator
from datetime import datetime

class AgentWorkflow:
    def __init__(self, db_client=None):
        self.analysis_agent = AnalysisAgent()
        self.research_agent = ResearchAgent()
        self.critic_agent = CriticAgent()
        self.monitor_agent = MonitorAgent()
        self.db_client = db_client
    
    async def process_problem_stream(self, problem: str) -> AsyncGenerator[Dict, None]:
        """
        Process problem with 4 agents in sequence
        Each agent is a GPT call with different prompt
        """
        context = {
            "problem": problem,
            "all_responses": {},
            "iteration": 1
        }
        
        # Always run only 1 iteration
        iteration = 1
        context["iteration"] = iteration
        
        # Send immediate start message
        yield {
            "agent": "system",
            "status": "starting",
            "message": "Starting analysis..."
        }
            
        # Stage 1: Analysis Agent - send thinking message immediately
        yield {
            "agent": "analysis",
            "stage": 1,
            "iteration": iteration,
            "status": "thinking",
            "message": "Analyzing the problem and breaking it down into sub-problems..."
        }
        analysis = await self.analysis_agent.process(context)
        context["analysis"] = analysis
        context["all_responses"]["analysis"] = analysis
        yield {
            "agent": "analysis",
            "stage": 1,
            "iteration": iteration,
            "status": "complete",
            "response": analysis
        }
        
        # Stage 2: Research Agent
        yield {
            "agent": "research",
            "stage": 2,
            "iteration": iteration,
            "status": "thinking",
            "message": "Gathering relevant knowledge, existing information, and professional assumptions..."
        }
        research = await self.research_agent.process(context)
        context["research"] = research
        context["all_responses"]["research"] = research
        yield {
            "agent": "research",
            "stage": 2,
            "iteration": iteration,
            "status": "complete",
            "response": research
        }
        
        # Stage 3: Critic Agent
        yield {
            "agent": "critic",
            "stage": 3,
            "iteration": iteration,
            "status": "thinking",
            "message": "Critically evaluating the solution, identifying weaknesses and contradictions..."
        }
        critique = await self.critic_agent.process(context)
        context["critique"] = critique
        context["all_responses"]["critique"] = critique
        yield {
            "agent": "critic",
            "stage": 3,
            "iteration": iteration,
            "status": "complete",
            "response": critique
        }
        
        # Stage 4: Monitor Agent
        yield {
            "agent": "monitor",
            "stage": 4,
            "iteration": iteration,
            "status": "thinking",
            "message": "Supervising the thinking process..."
        }
        monitor_response = await self.monitor_agent.process(context)
        context["monitor"] = monitor_response
        context["all_responses"]["monitor"] = monitor_response
        yield {
            "agent": "monitor",
            "stage": 4,
            "iteration": iteration,
            "status": "complete",
            "response": monitor_response
        }
        
        # Generate final summary using AI
        yield {
            "agent": "summary",
            "stage": 5,
            "status": "thinking",
            "message": "Summarizing all agent responses into final answer..."
        }
        final_summary = await self._generate_ai_summary(context)
        yield {
            "agent": "summary",
            "stage": 5,
            "status": "complete",
            "response": final_summary,
            "done": True
        }

    async def _generate_ai_summary(self, context: Dict) -> str:
        """Generate final AI summary from all agent responses"""
        all_responses = context.get("all_responses", {})
        problem = context.get("problem", "")
        
        # Use the Analysis Agent to generate the summary (it has access to _call_llm)
        summary_prompt = f"""Based on the complete analysis by all 4 agents, provide a comprehensive final answer.

Original Problem: {problem}

Analysis Agent Response:
{all_responses.get('analysis', 'N/A')}

Research Agent Response:
{all_responses.get('research', 'N/A')}

Critic Agent Response:
{all_responses.get('critique', 'N/A')}

Monitor Agent Response:
{all_responses.get('monitor', 'N/A')}

Provide a final, complete answer that:
1. Synthesizes all insights from the 4 agents into a coherent response
2. Directly answers the original problem with clear conclusions
3. Highlights key findings and actionable recommendations
4. Is well-structured with clear sections using ## for main headers and ### for sub-headers
5. Uses **bold** for emphasis on important points
6. Provides a complete answer - do NOT ask for additional information or clarification
7. Treat this as the final response - be definitive and comprehensive

Format your response like a professional analysis with proper markdown headers (## and ###) and bold text (**text**)."""
        
        return await self.analysis_agent._call_llm(summary_prompt, context)
