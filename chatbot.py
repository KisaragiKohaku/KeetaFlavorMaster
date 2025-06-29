from ctransformers import AutoModelForCausalLM
import torch
import os
import logging
import json

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def _validate_file(path, name):
    if not os.path.exists(path):
        logger.error(f"{name} Not Found: {path}")
        raise FileNotFoundError(f"{name} Not Found: {path}")


def _load_menu():
    _validate_file('data/menu.json', "Menu File")
    try:
        with open("data/menu.json") as f:
            return json.load(f)["dishes"]
    except Exception as e:
        logger.error(f"Menu loading failed: {str(e)}")
        raise


def _init_model(model_path):
    try:
        config = {
            "model_type": "mistral",
            "context_length": 16000,
            "threads": 8,
            "batch_size": 512,
            "gpu_layers": 16 if torch.cuda.is_available() else 0
        }

        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, **config)

        if config["gpu_layers"] > 0:
            logger.info(f"✅ GPU acceleration enabled with ({config['gpu_layers']} layers)")
            torch.cuda.empty_cache()

        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise


class FoodChatBot:
    def __init__(self):
        model_path = "models/Mistral-7B-Instruct-v0.3.Q4_K_S.gguf"
        _validate_file(model_path, "Model File")

        self.llm = _init_model(model_path)
        self.menu = _load_menu()

    def _build_prompt(self, query, context):
        menu_str = "\n".join(
            f"{d['id']}. {d['name']} ({d['price']}HKD) - {', '.join(d['tags'])}"
            for d in self.menu
        )

        context_items = [
            f"=== Matched Dish {i + 1} ===\n{doc}"
            for i, doc in enumerate(context["documents"])
        ]
        context_str = "\n".join(context_items) if context_items else "No relevant data found."

        system_prompt = f"""
        You are an AI assistant designed to provide personalized recommendations based solely on the provided data.
        If the user's query is unrelated to menu recommendations, just directly chat with the user.
        If the user's query is related to menu recommendations, you must strictly follow these rules:
        ---
        **Guidelines:**
        1. **Data Restriction**:
           - Use ONLY the nutrition data in **"Nutrition Information"** (from context_str).
           - Use ONLY the menu structure in **"Current Menu"** (from menu_str).
           - Prioritize context_str for specific dish details (e.g., price, ingredients, nutrition).
           - Use menu_str only to validate dish IDs/names and prevent hallucinations.
        2. **Personalized Recommendations**: 
           - Ask the user for their preferences (e.g., taste, dietary restrictions, budget) if not already specified.
           - Tailor recommendations based on the user’s input, ensuring the dishes match their needs.
        3. **Recommendation Requirements**: 
           - Only recommend dishes that appear in **both** the context_str and your knowledge of the menu.
           - Ensure all recommended dishes match the user's explicit filters (e.g., price range, dietary restrictions).
        4. **Nutrition Information Display**:
           - Present nutritional data in a clear, structured format (e.g., list or table) for easy understanding.
           - Example format for a dish:
             - Calories: 300 kcal
             - Protein: 20g
             - Fat: 10g
        5. **Language**: Respond in English with a polite and professional tone. Avoid slang or personal opinions.
        6. **Handling Unavailable Information**:
           - If the query cannot be fully answered, state: "I don’t have enough information to answer this completely."
           - Then, politely ask for more details (e.g., "Could you specify your dietary preferences or budget?").
        7. **Response Format**: Structure your reply as:
           - **Recommended Dishes**:
             - List each dish in the format: "- Dish Name (Price) - ingredients - Why it’s recommended"
           - **Nutrition Details**:
             - Provide a brief summary or table of key nutritional metrics for the recommended dishes.
           - **Additional Information**:
             - Offer relevant details (e.g., flavor profile, ingredients, or pairing suggestions).
             - End with a newline (`\n`).

        ---

        **Example Interaction:**
        Type 1:
        **User**: "Hi! How are you today?"
        **Assistant**:
        Hi! I am an assistant designed to provide personalized recommendations. Please tell me what you want~
        
        Type 2:
        **User**: "I’m looking for a low-fat dish."
        **Assistant**:
        **Recommended Dishes**  
        - Oatmeal with Egg White (13.1 HKD) -> oatmeal, egg white -> Low-fat oatmeal with egg white.  

        **Nutrition Details**  
        - Calories: 150 kcal, Protein: 10g, Fat: 3g  

        **Additional Information**  
        - This dish is excellent for a low-fat diet.

        ---

        **Current Menu:**
        {menu_str}

        ---

        **Nutrition Information:**
        {context_str}

        ---

        **Notes:**
        - Keep responses concise and directly relevant to the user’s query.
        - If the user’s query is vague, politely ask for more details.
        - You can state: "Could you tell me more about your taste preferences or any dietary restrictions?"
        - Always maintain a neutral tone and focus on the provided data.

        ---

        Please respond to the user’s query using the above instructions and data."""

        return f"[INST] {system_prompt} [/INST]\n\n[INST] {query} [/INST]"

    def generate_response(self, query, context):
        try:
            prompt = self._build_prompt(query, context)

            for token in self.llm(prompt,
                                  temperature=0.8,
                                  top_p=0.8,
                                  top_k=30,
                                  max_new_tokens=648,
                                  stream=True,
                                  stop=["</s>", "[/INST]", "<|im_end|>"]):
                yield token

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            yield f"⚠️ Error generating response: {str(e)}"
