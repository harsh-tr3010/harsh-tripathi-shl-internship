import os
import json
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

embeddings_engine = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

def fetch_and_vectorize_catalog():
    os.makedirs("data", exist_ok=True)
    catalog_path = "data/catalog.json"

    try:
        response = requests.get(CATALOG_URL, timeout=15, verify=False)
        response.raise_for_status()
        raw_text = response.text
    except Exception as e:
        print(f"Error downloading source catalog: {e}")
        return

    try:
        decoder = json.JSONDecoder(strict=False)
        raw_data = decoder.decode(raw_text)
    except Exception as e:
        print(f"Error decoding JSON with permissive parser: {e}")
        return

    if isinstance(raw_data, list):
        raw_items = raw_data
    elif isinstance(raw_data, dict):
        raw_items = raw_data.get("items", raw_data.get("products", []))
    else:
        print("Error: Loaded JSON structure is neither a list nor a dictionary.")
        return

    cleaned_items = []

    print(f"Processing and embedding {len(raw_items)} catalog items...")

    for item in raw_items:
        name = item.get("name", "")
        description = item.get("description", "")
        url = item.get("url", "")
        
        if "Pre-packaged" in name or "Solution" in name:
            continue

        if not name or not description:
            continue

        text_to_embed = f"Name: {name} | Description: {description}"
        
        try:
            embedding = embeddings_engine.embed_query(text_to_embed)
        except Exception as e:
            print(f"Error generating embedding for {name}: {e}")
            embedding = []

        cleaned_items.append({
            "name": name,
            "description": description,
            "url": url,
            "embedding": embedding
        })

    output_payload = {"items": cleaned_items}

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(f"Successfully compiled and saved {len(cleaned_items)} vector-embedded items to {catalog_path}")

if __name__ == "__main__":
    fetch_and_vectorize_catalog()
