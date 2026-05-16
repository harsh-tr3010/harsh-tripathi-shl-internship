__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse
from agent import process_chat

app = FastAPI(title="SHL Assessment Recommender API")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await process_chat(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))