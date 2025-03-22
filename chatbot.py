from ctransformers import AutoModelForCausalLM
import torch
import os
import logging
import json

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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
            "context_length": 15000,
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

        self._warmup_model()

    def _warmup_model(self):
        """模型预加载"""
        logger.info("Warm up the model...")
        try:
            fake_prompt = "[INST] Warm-up Test [/INST]\n\n[INST] Query Test [/INST]"
            for _ in self.llm(fake_prompt, stream=True, max_new_tokens=1):
                pass
        except Exception as e:
            logger.warning(f"Error during warm-up: {str(e)}")

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

        system_prompt = f"""You are a professional dining assistant and must strictly follow these rules:
1. Answer only using the provided menu and nutrition data.
2. Recommendations must include price and at least two nutritional metrics.
3. Reply in English, keeping it friendly yet professional.
4. If the user's query involves unavailable info, clearly state it cannot be answered.
5. Please format your response as follows:

**Recommended Dishes**
- Dish Name (Price) - Nutritional Metrics
**Additional Information**
- Other relevant information
- Notice that you should use '\n' in the end

=== Current Menu ===
{menu_str}

=== Relevant Nutrition Data ===
{context_str}"""

        return f"[INST] {system_prompt} [/INST]\n\n[INST] {query} [/INST]"

    def generate_response(self, query, context):
        prompt = self._build_prompt(query, context)

        try:
            for token in self.llm(prompt,
                                  temperature=0.85,
                                  top_p=0.9,
                                  stream=True,
                                  stop=["</s>", "[/INST]", "<|im_end|>"]):
                yield token

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            yield f"⚠️ Error generating response: {str(e)}"
