import os
import json
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from models import ChatRequest, ChatResponse

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile",
    temperature=0.0
)

embeddings_engine = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

def load_catalog_and_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    
    if not os.path.exists(catalog_path):
        return []
        
    with open(catalog_path, "r") as f:
        catalog_data = json.load(f)
        
    items = catalog_data.get("items", [])
    valid_items = [item for item in items if "embedding" in item and item["embedding"]]
    
    if not valid_items:
        return []
        
    query_vector = np.array(embeddings_engine.embed_query(query))
    
    results = []
    for item in valid_items:
        item_vector = np.array(item["embedding"])
        
        dot_prod = np.dot(query_vector, item_vector)
        norm_q = np.linalg.norm(query_vector)
        norm_i = np.linalg.norm(item_vector)
        
        similarity = dot_prod / (norm_q * norm_i) if (norm_q * norm_i) > 0 else 0.0
        results.append((similarity, item))
        
    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:top_k]]

async def process_chat(request: ChatRequest) -> ChatResponse:
    langchain_messages = []
    user_query = ""
    
    for msg in request.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
            user_query = msg.content
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    matched_assessments = load_catalog_and_search(user_query, top_k=3)
    
    context_str = "\n".join([
        f"- Name: {item['name']} | URL: {item.get('url', 'N/A')} | Info: {item.get('description', '')}"
        for item in matched_assessments
    ])
    
    system_prompt = (
        "You are an expert HR Assessment Assistant matching candidates to valid solutions from our Filtered Catalog.\n"
        "CRITICAL RULES:\n"
        "1. Only suggest specific individual tests listed explicitly in the Catalog Context below.\n"
        "2. If the user asks for general hiring advice, pricing structure, or regulatory/legal/HIPAA compliance metrics, "
        "you MUST cleanly refuse by stating you can only recommend catalog solutions. Return an empty recommendations array.\n"
        "3. You must respond ONLY with a raw, valid JSON object following this exact schema structure:\n"
        "{\n"
        '  "reply": "Your conversational text response here.",\n'
        '  "recommendations": [{"name": "Exact Test Name", "url": "Exact URL Link"}],\n'
        '  "end_of_conversation": false\n'
        "}\n\n"
        f"FILTERED CATALOG KNOWLEDGE BASE CONTEXT:\n{context_str}"
    )
    
    langchain_messages.insert(0, SystemMessage(content=system_prompt))
    
    llm_output = await llm.ainvoke(langchain_messages)
    raw_content = str(llm_output.content).strip()
    
    if raw_content.startswith("```"):
        raw_content = raw_content.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    if raw_content.startswith("json"):
        raw_content = raw_content.split("json", 1)[1].strip()

    try:
        parsed_response = json.loads(raw_content)
        return ChatResponse(
            reply=parsed_response.get("reply", ""),
            recommendations=parsed_response.get("recommendations", []),
            end_of_conversation=parsed_response.get("end_of_conversation", False)
        )
    except Exception:
        return ChatResponse(
            reply="I encountered an issue processing that recommendation request. Could you specify the job role again?",
            recommendations=[],
            end_of_conversation=False
        )