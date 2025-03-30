import requests
import os

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

def parse_intent(query):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Given the user's query below, extract the following in JSON format:
    {{
        "allergens": ["list any allergens the user mentioned or wants to avoid"],
        "price_range": {{"min": float, "max": float}} or null,
        "intent": "brief description of user's intent"
    }}

    User query: "{query}"
    JSON output:
    """

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, json=data).json()
    return response["choices"][0]["message"]["content"]
