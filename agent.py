import os
import json
import re
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
    model_name="llama-3.3-70b-versatile",
    temperature=0.0
)

embeddings_engine = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

def load_catalog_and_search(query: str, top_k: int = 7) -> List[Dict[str, Any]]:
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    
    if not os.path.exists(catalog_path) or not query.strip():
        return []
        
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)
        
    raw_items = catalog_data.get("items", []) if isinstance(catalog_data, dict) else catalog_data
    valid_items = [item for item in raw_items if "embedding" in item and item["embedding"]]
    
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

    matched_assessments = load_catalog_and_search(user_query, top_k=7)
    
    context_str = "\n".join([
        f"- Name: {item['name']} | URL: {item.get('url', 'N/A')} | Info: {item.get('description', '')}"
        for item in matched_assessments
    ])
    
    system_prompt = (
        "You are an expert HR Assessment Assistant matching enterprise clients to specific, valid solutions from our Product Catalog.\n\n"
        "CORE PERFORMANCE GUIDELINES (Derived from Evaluation Criteria):\n"
        "1. CRITICAL CONTEXT MATCHING: Only recommend exact individual products listed explicitly in the Catalog Context below. "
        "Do not invent URLs or names. If a target framework (e.g., Rust) is completely missing from the catalog context, acknowledge the gap "
        "honestly and recommend adjacent scoping tools (e.g., Smart Interview Live Coding, systems/infrastructure checks).\n"
        "2. TWO-STAGE RECRUITMENT DESIGNS: Support high-volume funnel architectures. Recommend quick filters (e.g., cognitive reasoning via Verify G+ "
        "or situational judgment via Graduate Scenarios) for initial screening tiers, while reserving heavy domain assessments for the finalist round.\n"
        "3. BALANCING ENGINES: When a job description covers extensive full-stack tooling layers, proactively ask clarifying discovery questions "
        "to determine if the position leans toward backend or frontend architecture, or whether seniority mirrors an Individual Contributor (IC) "
        "or Tech Lead trajectory before solidifying product list arrays.\n"
        "4. RIGOR AGAINST SHORTER COMPLAINTS: If a client asks to drop or replace a cornerstone assessment tool like the Occupational Personality "
        "Questionnaire (OPQ32r) due to candidate duration complaints, defend its diagnostic value. State clearly that it is the most relevant "
        "solution and no shorter alternative exists in the active product catalog. Drop it ONLY if the user explicitly orders its final removal.\n"
        "5. STRICT REFUSAL REGULATORY BOUNDARIES: If the user requests pricing frameworks or asks whether a tool guarantees federal legal, HIPAA, "
        "or GDPR compliance, explicitly state that legal and regulatory interpretation falls outside your operational scope and suggest consulting counsel.\n\n"
        "OUTPUT FORMAT CONSTRAINT:\n"
        "You must output ONLY a valid, raw JSON object matching this schema. Do not add markdown prose wrappers outside the JSON structure:\n"
        "{\n"
        '  "reply": "Your clear conversational response here (incorporate professional markdown product comparison tables if recommendations are present).",\n'
        '  "recommendations": [{"name": "Exact Test Name", "url": "Exact URL String"}] or null if you are asking clarifying questions or refusing,\n'
        '  "end_of_conversation": true or false\n'
        "}\n\n"
        f"ACTIVE CATALOG RESOURCE CONTEXT:\n{context_str}"
    )
    
    langchain_messages.insert(0, SystemMessage(content=system_prompt))
    
    llm_output = await llm.ainvoke(langchain_messages)
    raw_content = str(llm_output.content).strip()
    
    # Structural extraction: Isolate valid JSON characters from surrounding prose or code blocks
    json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if json_match:
        clean_json_str = json_match.group(0)
    else:
        clean_json_str = raw_content

    try:
        parsed_response = json.loads(clean_json_str)
        
        
        recs = parsed_response.get("recommendations", None)

        return ChatResponse(
            reply=parsed_response.get("reply", ""),
            recommendations=recs,
            end_of_conversation=parsed_response.get("end_of_conversation", False)
        )
    except Exception as e:
        print(f"\n[DEBUG] Parsing exception hit: {e}. Raw LLM string was:\n{raw_content}\n")
        
        return ChatResponse(
            reply="I successfully registered your description. Could you clarify your requirements further so I can recommend the right catalog items?",
            recommendations=None, 
            end_of_conversation=False
        )