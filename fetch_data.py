import requests
import json
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
DATA_FILE = "data/catalog.json"

def fetch_and_clean_catalog():
    response = requests.get(CATALOG_URL)
    if response.status_code != 200:
        return

    try:
        raw_data = json.loads(response.text, strict=False)
    except json.JSONDecodeError:
        clean_text = response.text.replace('\n', '\\n').replace('\t', '\\t')
        raw_data = json.loads(clean_text, strict=False)

    clean_catalog = []
    for item in raw_data:
        if item.get("solution_type", "").lower() != "pre-packaged job solution":
            clean_catalog.append({
                "name": item.get("name", "Unknown"),
                "url": item.get("url", ""),
                "test_type": item.get("test_type", "Unknown"),
                "description": item.get("description", ""),
                "skills_measured": item.get("skills", [])
            })

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_catalog, f, indent=4)

def build_vector_db():
    if not os.path.exists(DATA_FILE):
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    documents = []
    metadatas = []
    for item in catalog:
        text_content = f"Name: {item['name']}\nDescription: {item['description']}\nSkills: {' '.join(item['skills_measured'])}"
        documents.append(text_content)
        metadatas.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"]
        })

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="data/chroma_db"
    )

if __name__ == "__main__":
    fetch_and_clean_catalog()
    build_vector_db()