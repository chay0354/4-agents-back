from agents.analysis_agent import AnalysisAgent
from agents.research_agent import ResearchAgent
from agents.critic_agent import CriticAgent
from agents.monitor_agent import MonitorAgent
from typing import Dict, AsyncGenerator
import httpx
import os
from datetime import datetime

class AgentWorkflow:
    def __init__(self, db_client=None, kernel_check_func=None):
        self.analysis_agent = AnalysisAgent()
        self.research_agent = ResearchAgent()
        self.critic_agent = CriticAgent()
        self.monitor_agent = MonitorAgent()
        self.db_client = db_client
        # Use provided kernel check function, or fallback to HTTP request
        self.kernel_check_func = kernel_check_func
        # Get backend URL from environment or default to localhost (for fallback)
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    async def _check_kernel(self) -> bool:
        """
        Check kernel endpoint to see if analysis should continue
        Returns True if should continue, False if should stop
        """
        # If kernel check function is provided, use it directly (faster, works in production)
        if self.kernel_check_func:
            try:
                return not self.kernel_check_func()  # Return True if should continue (not stopped)
            except Exception as e:
                print(f"Error checking kernel via function: {e}")
                return True  # Default to continue on error
        
        # Fallback to HTTP request (for backwards compatibility or testing)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/kernel", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "ok"
                return True  # Default to continue on error
        except Exception as e:
            print(f"Error checking kernel via HTTP: {e}")
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
        agent_name = "Analysis Agent"
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] 🟢 STARTING: {agent_name}")
        yield {
            "agent": "analysis",
            "stage": 1,
            "iteration": iteration,
            "status": "thinking",
            "message": "Analyzing the problem and breaking it down into sub-problems...",
            "kernel_decision": None  # Not determined yet
        }
        # Wait for analysis to complete before proceeding
        analysis = await self.analysis_agent.process(context)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ✅ FINISHED: {agent_name} (duration: {duration:.2f}s, response length: {len(analysis) if analysis else 0})")
        
        # Yield agent response IMMEDIATELY - show it right away, don't wait for kernel check
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
        
        # Check kernel AFTER yielding response - if hard stop, prevent next agent from starting
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
        
        # Stage 2: Research Agent - only starts after analysis is complete
        agent_name = "Research Agent"
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] 🟢 STARTING: {agent_name}")
        yield {
            "agent": "research",
            "stage": 2,
            "iteration": iteration,
            "status": "thinking",
            "message": "Gathering relevant knowledge, existing information, and professional assumptions...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for research to complete before proceeding
        research = await self.research_agent.process(context)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ✅ FINISHED: {agent_name} (duration: {duration:.2f}s, response length: {len(research) if research else 0})")
        
        # Yield agent response IMMEDIATELY - show it right away, don't wait for kernel check
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
        
        # Check kernel AFTER yielding response - if hard stop, prevent next agent from starting
        should_continue = await self._check_kernel()
        if not should_continue:
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after Research Agent",
                "stopped_agent": "research",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return
        
        # Stage 3: Critic Agent - only starts after research is complete
        agent_name = "Critic Agent"
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] 🟢 STARTING: {agent_name}")
        yield {
            "agent": "critic",
            "stage": 3,
            "iteration": iteration,
            "status": "thinking",
            "message": "Critically evaluating the solution, identifying weaknesses and contradictions...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for critic to complete before proceeding
        critique = await self.critic_agent.process(context)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ✅ FINISHED: {agent_name} (duration: {duration:.2f}s, response length: {len(critique) if critique else 0})")
        
        # Yield agent response IMMEDIATELY - show it right away, don't wait for kernel check
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
        
        # Check kernel AFTER yielding response - if hard stop, prevent next agent from starting
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
        
        # Stage 4: Monitor Agent - only starts after critic is complete
        agent_name = "Monitor Agent"
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] 🟢 STARTING: {agent_name}")
        yield {
            "agent": "monitor",
            "stage": 4,
            "iteration": iteration,
            "status": "thinking",
            "message": "Supervising the thinking process...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for monitor to complete before proceeding
        monitor_response = await self.monitor_agent.process(context)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ✅ FINISHED: {agent_name} (duration: {duration:.2f}s, response length: {len(monitor_response) if monitor_response else 0})")
        
        # Yield agent response IMMEDIATELY - show it right away, don't wait for kernel check
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
        
        # Check kernel AFTER yielding response - if hard stop, prevent next agent from starting
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
        
        # Generate final summary using AI - only starts after monitor is complete
        agent_name = "Summary Agent"
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] 🟢 STARTING: {agent_name}")
        yield {
            "agent": "summary",
            "stage": 5,
            "status": "thinking",
            "message": "Summarizing all agent responses into final answer...",
            "kernel_decision": None  # Still in progress
        }
        # Wait for summary to complete
        final_summary = await self._generate_ai_summary(context)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ✅ FINISHED: {agent_name} (duration: {duration:.2f}s, response length: {len(final_summary) if final_summary else 0})")
        
        # Yield summary IMMEDIATELY - show it right away, don't wait for kernel check
        yield {
            "agent": "summary",
            "stage": 5,
            "status": "complete",
            "response": final_summary,
            "done": True,
            "kernel_decision": "N"  # N = Normal completion (no hard stop)
        }
        
        # Check kernel AFTER yielding summary - if hard stop was activated, mark it
        # (Note: summary is already shown, but we check for consistency)
        should_continue = await self._check_kernel()
        if not should_continue:
            # Summary was already shown, but mark that it was stopped
            yield {
                "agent": "system",
                "status": "stopped",
                "message": f"Analysis stopped by kernel after Summary",
                "stopped_agent": "summary",
                "kernel_decision": "L"  # L = Limited/Stopped by kernel
            }
            return

    async def _generate_ai_summary(self, context: Dict) -> str:
        """Generate final AI summary from all agent responses"""
        all_responses = context.get("all_responses", {})
        problem = context.get("problem", "")
        
        # Use the Analysis Agent to generate the summary (it has access to _call_llm)
        summary_prompt = f"""Based on the complete analysis by all agents, provide a comprehensive final answer.

Original Problem: {problem}

Analysis Agent Response:
{all_responses.get('analysis', 'N/A')}

Research Agent Response:
{all_responses.get('research', 'N/A')}

Critic Agent Response:
{all_responses.get('critique', 'N/A')}

Monitor Agent Response:
{all_responses.get('monitor', 'N/A')}

Final Ratings Agent Response:
{all_responses.get('ratings', 'N/A')}

Provide a final, complete answer that:
1. Synthesizes all insights from the agents into a coherent response
2. Directly answers the original problem with clear conclusions
3. Highlights key findings and actionable recommendations
4. Incorporates insights from the Final Ratings Agent about agent performance
5. Is well-structured with clear sections using ## for main headers and ### for sub-headers
6. Uses **bold** for emphasis on important points
7. Provides a complete answer - do NOT ask for additional information or clarification
8. Treat this as the final response - be definitive and comprehensive

Format your response like a professional analysis with proper markdown headers (## and ###) and bold text (**text**)."""
        
        return await self.analysis_agent._call_llm(summary_prompt, context)
