from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from agents.workflow import AgentWorkflow
from database.mongodb import MongoDBClient

load_dotenv()

app = FastAPI(title="4-Agents MOP System", version="1.0.0")

# CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB client
db_client = MongoDBClient()

class ProblemRequest(BaseModel):
    problem: str

@app.get("/")
async def root():
    return {"message": "4-Agents MOP System API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "database": db_client.is_connected()}

@app.post("/analyze")
async def analyze_problem(request: ProblemRequest):
    """
    Main endpoint to analyze a problem using the 4-agent system
    Streams agent responses in real-time
    """
    async def generate():
        try:
            workflow = AgentWorkflow(db_client)
            all_responses = {}
            
            async for update in workflow.process_problem_stream(request.problem):
                # Collect all responses for final save
                if update.get("status") == "complete" and "response" in update:
                    all_responses[update["agent"]] = update["response"]
                
                # Stream the update
                yield f"data: {json.dumps(update)}\n\n"
            
            # Save to database
            result = {
                "problem": request.problem,
                "responses": all_responses,
                "final_insights": all_responses.get("final", ""),
                "status": "completed",
                "created_at": datetime.utcnow().isoformat()
            }
            db_client.save_analysis(result)
            
        except Exception as e:
            error_update = {
                "agent": "error",
                "status": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_update)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/analyses")
async def get_analyses():
    """
    Get all previous analyses
    """
    try:
        analyses = db_client.get_all_analyses()
        return {"analyses": analyses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Get a specific analysis by ID
    """
    try:
        analysis = db_client.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

