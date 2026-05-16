from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse
from agent import process_chat

app = FastAPI(title="SHL Assessment Recommender API")

@app.get("/health")
async def health_check():
    """Returns status ok for automated evaluation readiness."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Takes conversation history and returns the agent's next action."""
    try:
        response = await process_chat(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))