from agents.analysis_agent import AnalysisAgent
from agents.research_agent import ResearchAgent
from agents.critic_agent import CriticAgent
from agents.monitor_agent import MonitorAgent
from typing import Dict, AsyncGenerator
import httpx
import os

class AgentWorkflow:
    def __init__(self, db_client=None):
        self.analysis_agent = AnalysisAgent()
        self.research_agent = ResearchAgent()
        self.critic_agent = CriticAgent()
        self.monitor_agent = MonitorAgent()
        self.db_client = db_client
        # Get backend URL from environment or default to localhost
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    async def _check_kernel(self) -> bool:
        """
        Check kernel endpoint to see if analysis should continue
        Returns True if should continue, False if should stop
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/kernel", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "ok"
                return True  # Default to continue on error
        except Exception as e:
            print(f"Error checking kernel: {e}")
            return True  # Default to continue on error
    
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
            "message": "Starting analysis...",
            "kernel_decision": None  # Not determined yet
        }
            
        # Stage 1: Analysis Agent - send thinking message immediately
        yield {
            "agent": "analysis",
            "stage": 1,
            "iteration": iteration,
            "status": "thinking",
            "message": "Analyzing the problem and breaking it down into sub-problems...",
            "kernel_decision": None  # Not determined yet
        }
        # Wait for analysis to complete before proceeding
        print("Starting Analysis Agent...")
        analysis = await self.analysis_agent.process(context)
        print(f"Analysis Agent completed (response length: {len(analysis) if analysis else 0})")
        
        # Check kernel IMMEDIATELY after agent completes - stop right away if requested
        should_continue = await self._check_kernel()
        if not should_continue:
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after Analysis Agent",
                "stopped_agent": "analysis",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return
        
        # Only yield agent response if we're continuing
        context["analysis"] = analysis
        context["all_responses"]["analysis"] = analysis
        yield {
            "agent": "analysis",
            "stage": 1,
            "iteration": iteration,
            "status": "complete",
            "response": analysis,
            "kernel_decision": None  # Still in progress, not final
        }
        
        # Stage 2: Research Agent - only starts after analysis is complete
        yield {
            "agent": "research",
            "stage": 2,
            "iteration": iteration,
            "status": "thinking",
            "message": "Gathering relevant knowledge, existing information, and professional assumptions...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for research to complete before proceeding
        print("Starting Research Agent...")
        research = await self.research_agent.process(context)
        print(f"Research Agent completed (response length: {len(research) if research else 0})")
        
        # Check kernel IMMEDIATELY after agent completes - stop right away if requested
        should_continue = await self._check_kernel()
        if not should_continue:
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after {self.research_agent.name}",
                "stopped_agent": "research",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return
        
        # Only yield agent response if we're continuing
        context["research"] = research
        context["all_responses"]["research"] = research
        yield {
            "agent": "research",
            "stage": 2,
            "iteration": iteration,
            "status": "complete",
            "response": research,
            "kernel_decision": None  # Still in progress
        }
        
        # Stage 3: Critic Agent - only starts after research is complete
        yield {
            "agent": "critic",
            "stage": 3,
            "iteration": iteration,
            "status": "thinking",
            "message": "Critically evaluating the solution, identifying weaknesses and contradictions...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for critic to complete before proceeding
        print("Starting Critic Agent...")
        critique = await self.critic_agent.process(context)
        print(f"Critic Agent completed (response length: {len(critique) if critique else 0})")
        
        # Check kernel IMMEDIATELY after agent completes - stop right away if requested
        should_continue = await self._check_kernel()
        if not should_continue:
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after Critic Agent",
                "stopped_agent": "critic",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return
        
        # Only yield agent response if we're continuing
        context["critique"] = critique
        context["all_responses"]["critique"] = critique
        yield {
            "agent": "critic",
            "stage": 3,
            "iteration": iteration,
            "status": "complete",
            "response": critique,
            "kernel_decision": None  # Still in progress
        }
        
        # Stage 4: Monitor Agent - only starts after critic is complete
        yield {
            "agent": "monitor",
            "stage": 4,
            "iteration": iteration,
            "status": "thinking",
            "message": "Supervising the thinking process...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for monitor to complete before proceeding
        print("Starting Monitor Agent...")
        monitor_response = await self.monitor_agent.process(context)
        print(f"Monitor Agent completed (response length: {len(monitor_response) if monitor_response else 0})")
        
        # Check kernel IMMEDIATELY after agent completes - stop right away if requested
        should_continue = await self._check_kernel()
        if not should_continue:
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after Monitor Agent",
                "stopped_agent": "monitor",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return
        
        # Only yield agent response if we're continuing
        context["monitor"] = monitor_response
        context["all_responses"]["monitor"] = monitor_response
        yield {
            "agent": "monitor",
            "stage": 4,
            "iteration": iteration,
            "status": "complete",
            "response": monitor_response,
            "kernel_decision": None  # Still in progress
        }
        
        # Generate final summary using AI - only starts after monitor is complete
        yield {
            "agent": "summary",
            "stage": 5,
            "status": "thinking",
            "message": "Summarizing all agent responses into final answer...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for summary to complete
        print("Starting Summary generation...")
        final_summary = await self._generate_ai_summary(context)
        print(f"Summary completed (response length: {len(final_summary) if final_summary else 0})")
        
        # Check kernel IMMEDIATELY after summary completes - stop right away if requested
        should_continue = await self._check_kernel()
        if not should_continue:
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after Summary",
                "stopped_agent": "summary",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return
        
        # Only yield summary if we're continuing - this is the final successful completion
        yield {
            "agent": "summary",
            "stage": 5,
            "status": "complete",
            "response": final_summary,
            "done": True,
            "kernel_decision": "N"  # N = Normal completion (no hard stop)
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
