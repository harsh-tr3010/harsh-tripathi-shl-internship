from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    end_of_conversation: bool