import requests
import json
import os

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
DATA_FILE = "data/catalog.json"

def fetch_and_clean_catalog():
    print(f"Fetching raw catalog from {CATALOG_URL}...")
    response = requests.get(CATALOG_URL)
    
    if response.status_code != 200:
        print(f"Failed to fetch. Status code: {response.status_code}")
        return

    
    try:
        raw_data = json.loads(response.text, strict=False)
    except json.JSONDecodeError:
        # Fallback just in case the data is extremely messy: manually clean the text string first
        clean_text = response.text.replace('\n', '\\n').replace('\t', '\\t')
        raw_data = json.loads(clean_text, strict=False)

    clean_catalog = []

    for item in raw_data:
        # Filter strictly for Individual Test Solutions as mandated by the assignment
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
        
    print(f"Successfully saved {len(clean_catalog)} Individual Test Solutions to {DATA_FILE}")

if __name__ == "__main__":
    fetch_and_clean_catalog()