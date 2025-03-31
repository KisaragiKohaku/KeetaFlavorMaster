import requests
import json
import re
import logging
import os


logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def parse_intent(query: str) -> str:
    if not DEEPSEEK_API_KEY:
        print("[ERROR] DeepSeek API key is missing or empty!")
    else:
        print("[DEBUG] DeepSeek API key is loaded (length:", len(DEEPSEEK_API_KEY), ")")

    headers = {
        "Authorization": f"Bearer " + DEEPSEEK_API_KEY,
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a backend intent extractor. Analyze the user query and extract structured JSON.

Return only a JSON object. No markdown, no explanations.

Expected format:
{{
  "allergens": [list of allergens, lowercase, or empty list],
  "price_range": {{"min": float, "max": float}} or null,
  "intent": brief summary of user's need
}}

Examples:

User: "I want something vegetarian and cheap"
→ {{
  "allergens": [],
  "price_range": {{"max": 20}},
  "intent": "cheap vegetarian food"
}}

User: "Give me something not too pricey, no dairy"
→ {{
  "allergens": ["dairy"],
  "price_range": {{"max": 20}},
  "intent": "avoid dairy, budget food"
}}

User: "I want a luxurious dinner"
→ {{
  "allergens": [],
  "price_range": {{"min": 40}},
  "intent": "luxury food"
}}

User: "{query}"
→
"""

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
        "stream": False
    }

    fallback = {
        "allergens": [],
        "price_range": None,
        "intent": "unknown"
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=data)

        if response.status_code != 200:
            logger.error(f"DeepSeek API error {response.status_code}: {response.text}")
            return json.dumps(fallback)

        try:
            result = response.json()
        except Exception:
            logger.error(f"Failed to parse response as JSON: {response.text}")
            return json.dumps(fallback)

        if "choices" not in result:
            logger.error(f"Missing 'choices' in DeepSeek response: {result}")
            return json.dumps(fallback)

        # Step 1: 提取 content 字段
        content = result["choices"][0]["message"]["content"]

        # Step 2: 使用正则提取 JSON 块
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match:
            content = match.group(0)

        # Step 3: 验证 JSON 合法性
        try:
            json.loads(content)
            logger.info(f"Extracted raw content:\n{content}")
            return content  # 返回字符串 JSON，供调用方再加载
        except Exception:
            logger.error(f"Extracted raw content:\n{content}") 
            logger.error("Extracted content is invalid JSON")
            return json.dumps(fallback)

    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return json.dumps(fallback)
