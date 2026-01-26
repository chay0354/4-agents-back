# Backend - 4-Agents MOP System

Python FastAPI backend for the multi-agent problem-solving system.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
MONGODB_URI=mongodb+srv://Vercel-Admin-4-agents-proj:ZLNIJ3TO4H0a067t@4-agents-proj.ih3tbvw.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_NAME=4_agents_db
```

3. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Kernel Decision Field (`kernel_decision`)

### Overview

The `kernel_decision` field is a **binary indicator** that explicitly represents the Kernel's decision about whether the analysis completed normally or was stopped by a hard stop. This field is designed for automated accuracy testing without needing to parse text or understand internal logic.

### Field Values

- **`"N"`** - Normal completion: The analysis completed successfully without any hard stop intervention
- **`"L"`** - Limited/Stopped: The analysis was stopped by the Kernel (hard stop activated)
- **`null`** - In progress: The analysis is still running, decision not yet determined

### Where It Appears

#### 1. Streaming Response (`POST /analyze`)

The `kernel_decision` field appears in **every update** streamed via Server-Sent Events (SSE):

```json
{
  "agent": "system",
  "status": "stopped",
  "message": "Analysis stopped by kernel after Analysis Agent",
  "stopped_agent": "analysis",
  "kernel_decision": "L"
}
```

**During analysis:**
- Updates have `"kernel_decision": null` while the analysis is in progress
- When an agent completes successfully: `"kernel_decision": null` (still in progress)
- When hard stop occurs: `"kernel_decision": "L"` immediately
- When analysis completes successfully: `"kernel_decision": "N"` in the final summary update

**Example - Successful completion:**
```json
{
  "agent": "summary",
  "stage": 5,
  "status": "complete",
  "response": "...",
  "done": true,
  "kernel_decision": "N"
}
```

**Example - Hard stop:**
```json
{
  "agent": "system",
  "status": "stopped",
  "message": "Analysis stopped by kernel after Research Agent",
  "stopped_agent": "research",
  "kernel_decision": "L"
}
```

#### 2. Saved Analysis (`GET /analyses/{analysis_id}`)

The `kernel_decision` field is **permanently stored** in the database with each analysis:

```json
{
  "_id": "...",
  "problem": "What is the capital of France?",
  "responses": { ... },
  "final_insights": "...",
  "status": "completed",
  "kernel_decision": "N",
  "created_at": "2026-01-25T10:30:00Z"
}
```

**Field values in saved analyses:**
- `"kernel_decision": "N"` - Analysis completed successfully
- `"kernel_decision": "L"` - Analysis was stopped by kernel
- `"status"` field correlates: `"completed"` when `"N"`, `"stopped"` when `"L"`

### How It Works

1. **Kernel Check**: After each agent completes, the system checks the `/kernel` endpoint
2. **Decision Logic**:
   - If kernel returns `{"status": "stop"}` → `kernel_decision = "L"` (hard stop)
   - If kernel returns `{"status": "ok"}` → Continue, decision remains `null`
   - If all agents complete successfully → Final update has `kernel_decision = "N"`

3. **Tracking**: The final `kernel_decision` value is tracked through all streaming updates and saved to the database

### Usage for Automated Testing

You can use this field directly in your test automation:

```python
# Example: Check if analysis completed normally
response = await client.get(f"/analyses/{analysis_id}")
analysis = response.json()

if analysis["kernel_decision"] == "N":
    print("✅ Analysis completed successfully")
elif analysis["kernel_decision"] == "L":
    print("❌ Analysis was stopped by kernel")
```

```javascript
// Example: Check during streaming
eventSource.onmessage = (event) => {
  const update = JSON.parse(event.data);
  
  if (update.kernel_decision === "L") {
    console.log("Analysis stopped by kernel");
    // Handle stop case
  } else if (update.kernel_decision === "N") {
    console.log("Analysis completed successfully");
    // Handle success case
  }
};
```

### Benefits

- ✅ **No text parsing required** - Direct binary value
- ✅ **No internal logic knowledge needed** - Just check the field
- ✅ **Available in both streaming and saved results** - Consistent across endpoints
- ✅ **Explicit and clear** - "N" or "L" is unambiguous




