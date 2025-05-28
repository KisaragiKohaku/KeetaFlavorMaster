from chromadb.api.types import IncludeEnum
from sentence_transformers import SentenceTransformer
from intent_parser import parse_intent
import chromadb
import logging
import torch
import json
import os

logger = logging.getLogger(__name__)


class NutritionRetriever:
    def __init__(self):
        for file in ["data/menu.json"]:
            if not os.path.exists(file):
                raise FileNotFoundError(f"File Not Found: {file}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using Device: {self.device.upper()}")

        self._load_menu_data()

        self.encoder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device,
            cache_folder="models/sentence-transformers"
        )
        self.client = chromadb.PersistentClient(path="./chroma_db")

        self._init_collection()
        self._build_allergen_lookup()

    def _init_collection(self):
        try:
            existing_collections = self.client.list_collections()
            if "menu" in existing_collections:
                logger.info("Using existing Menu collection")
                self.collection = self.client.get_collection("menu")
            else:
                logger.info("Initializing New Menu collection")
                self.collection = self.client.create_collection(
                    name="menu",
                    metadata={"hnsw:space": "cosine"}
                )
                self._embed_menu_data()

        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    def _embed_menu_data(self):
        logger.info("Embedding Menu data...")
        try:
            documents, metadatas, ids = [], [], []
            for dish in self.menu_dict.values():
                doc = self.format_dish_info(dish)
                documents.append(doc)
                metadatas.append({"dish_id": dish["id"]})
                ids.append(str(dish["id"]))

            if documents:
                embeddings = self.encoder.encode(documents).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully embedded {len(documents)} dishes")
        except Exception as e:
            logger.error(f"Failed to embed data: {str(e)}")
            raise

    def format_dish_info(self, dish):
        nutrition = dish.get("nutrition", {})
        return (
            f"Dish: {dish['name']}\n"
            f"Price: {dish['price']} HKD\n"
            f"Ingredients: {', '.join(dish['ingredients'])}\n"
            f"Tags: {', '.join(dish['tags'])}\n"
            f"Serving Time: {dish['serving_time']}\n"
            f"Calories: {nutrition.get('calories', 'N/A')} kcal\n"
            f"Protein: {nutrition.get('protein', 'N/A')} g\n"
            f"Carbs: {nutrition.get('carbs', 'N/A')} g\n"
            f"Fat: {nutrition.get('fat', 'N/A')} g\n"
            f"Allergens: {', '.join(dish.get('allergens', []))}\n"
            f"Description: {dish.get('description', 'No description available')}"
        )

    def _load_menu_data(self):
        logger.info("Loading Menu data...")
        try:
            with open("data/menu.json") as f:
                data = json.load(f)["dishes"]
                self.menu_dict = {d["id"]: d for d in data}
                self.nutrition_dict = {d["id"]: d["nutrition"] for d in data}
            logger.info(f"Successfully loaded {len(self.menu_dict)} menu records")
        except Exception as e:
            logger.error(f"Failed to load menu: {str(e)}")
            raise

    def _build_allergen_lookup(self):
        self.dish_allergen_map = {
            dish_id: [a for a in dish.get("allergens", [])]
            for dish_id, dish in self.menu_dict.items()
        }

    def parse_price(self, value):
        try:
            if value is None:
                return float('inf')
            return float(str(value).replace('HKD', '').strip())
        except:
            return float('inf')

    def get_nutrition_data(self, dish_id):
        return self.nutrition_dict.get(dish_id, {
            "calories": "N/A",
            "protein": "N/A",
            "carbs": "N/A",
            "fat": "N/A",
            "allergens": [],
            "description": "No description available"
        })

    def search(self, query: str, k: int = 10) -> dict:
        try:
            intent_data = json.loads(parse_intent(query))
            excluded_info = [item for item in intent_data.get("avoid", []) if item is not None]
            price_info = [item if item is not None else None for item in intent_data.get("price", [None, None])]

            # Step 1: 向量检索
            query_embed = self.encoder.encode(query, normalize_embeddings=True)
            results = self.collection.query(
                query_embeddings=[query_embed.tolist()],
                n_results=k,
                include=[IncludeEnum.documents, IncludeEnum.metadatas]
            )

            matched_dishes = [
                self.menu_dict[meta["dish_id"]]
                for meta in results["metadatas"][0]
                if meta["dish_id"] in self.menu_dict
            ]
            logger.info(f"{len(matched_dishes)} dishes matched vector search")

            # Step 2: 过敏原过滤
            if excluded_info:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if all(
                        allergen not in self.dish_allergen_map.get(dish["id"], [])
                        for allergen in excluded_info
                    ) and dish["id"] in self.dish_allergen_map
                ]
                logger.info(f"{len(matched_dishes)} dishes remain after allergen filtering: {excluded_info}")

            # Step 3: 价格过滤
            if price_info and matched_dishes:
                min_price = price_info[0] if price_info[0] is not None else 0
                max_price = price_info[1] if price_info[1] is not None else float("inf")
                matched_dishes = [
                    dish for dish in matched_dishes
                    if min_price <= self.parse_price(dish.get("price", 0)) <= max_price
                ]
                logger.info(f"{len(matched_dishes)} dishes remain after price filtering: {price_info}")

            # Step 4: 构建文档（只基于最终保留的 matched_dishes）
            if matched_dishes:
                documents = [self.format_dish_info(d) for d in matched_dishes]
            else:
                documents = ["No matching dishes found based on your preferences."]

            return {
                "documents": documents,
                "dishes": matched_dishes
            }

        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            return {"documents": ["Search Function is not available"], "dishes": []}
