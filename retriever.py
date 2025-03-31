from chromadb.api.types import IncludeEnum
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from intent_parser import parse_intent
import chromadb
import logging
import torch
import json
import os
import re

logger = logging.getLogger(__name__)


class NutritionRetriever:
    def __init__(self):
        for file in ["data/menu.json", "data/nutrition.json"]:
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
        self._load_nutrition_dict()

    def _init_collection(self):
        try:
            existing_collections = self.client.list_collections()
            if "nutrition" in existing_collections:
                logger.info("Using existing Nutrition collection")
                self.collection = self.client.get_collection("nutrition")
            else:
                logger.info("Initializing New Nutrition collection")
                self.collection = self.client.create_collection(
                    name="nutrition",
                    metadata={"hnsw:space": "cosine"}
                )
                self._load_nutrition_data()

        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    def _load_nutrition_dict(self):
        self.nutrition_dict = {}
        try:
            with open("data/nutrition.json") as f:
                nutrition_data = json.load(f)["nutrition_data"]
                self.nutrition_dict = {item["dish_id"]: item for item in nutrition_data}
            logger.info(f"Successfully loaded {len(self.nutrition_dict)} nutrition records into dictionary.")
        except Exception as e:
            logger.error(f"Failed to load nutrition data dictionary: {str(e)}")

    def format_dish_info(self, dish, nutrition):
        return (
            f"Dish: {dish['name']}\n"
            f"Price: {dish['price']} HKD\n"
            f"Ingredients: {', '.join(dish['ingredients'])}\n"
            f"Tags: {', '.join(dish['tags'])}\n"
            f"Serving Time: {dish['serving_time']}\n"
            f"Calories: {nutrition['calories']} kcal\n"
            f"Protein: {nutrition['protein']} g\n"
            f"Carbs: {nutrition['carbs']} g\n"
            f"Fat: {nutrition['fat']} g\n"
            f"Allergens: {', '.join(nutrition['allergens'])}\n"
            f"Description: {nutrition['description']}"
        )

    def _load_menu_data(self):
        logger.info("Loading Menu data...")
        try:
            with open("data/menu.json") as f:
                self.menu_dict = {d["id"]: d for d in json.load(f)["dishes"]}
            logger.info(f"Successfully loaded {len(self.menu_dict)} menu records")
        except Exception as e:
            logger.error(f"Failed to load menu: {str(e)}")
            raise

    def _load_nutrition_data(self):
        logger.info("Loading Nutrition data...")
        try:
            with open("data/nutrition.json") as f:
                nutrition_data = json.load(f)["nutrition_data"]

            documents, metadatas, ids = [], [], []
            for item in nutrition_data:
                if dish := self.menu_dict.get(item["dish_id"]):
                    documents.append(self.format_dish_info(dish, item))
                    metadatas.append({"dish_id": item["dish_id"]})
                    ids.append(str(item["dish_id"]))

            if documents:
                embeddings = self.encoder.encode(documents).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully loaded {len(documents)} nutrition records")

        except Exception as e:
            logger.error(f"Failed to load Nutrition data: {str(e)}")
            raise

    def _build_allergen_lookup(self):
        self.dish_allergen_map = {}
        try:
            with open("data/nutrition.json") as f:
                nutrition_data = json.load(f)["nutrition_data"]
                for item in nutrition_data:
                    self.dish_allergen_map[item["dish_id"]] = [a.lower() for a in item.get("allergens", [])]
        except Exception as e:
            logger.error(f"Failed to load allergen info: {str(e)}")

    def parse_price(self, value):
        try:
            return float(re.sub(r"[^\d.]", "", str(value)))
        except:
            return float("inf")

    def get_nutrition_data(self, dish_id):
        return self.nutrition_dict.get(dish_id, {
            "calories": "N/A",
            "protein": "N/A",
            "carbs": "N/A",
            "fat": "N/A",
            "allergens": [],
            "description": "No description available"
        })

    def search(self, query: str, k: int = 10) -> Dict[str, Union[List[str], List[dict]]]:
        try:
            intent_data = json.loads(parse_intent(query))
            excluded_allergens = intent_data.get('allergens', [])
            price_range = intent_data.get('price_range', None)

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
            if excluded_allergens:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if all(
                        allergen.lower() not in self.dish_allergen_map.get(dish["id"], [])
                        for allergen in excluded_allergens
                    )
                ]
                logger.info(f"{len(matched_dishes)} dishes remain after allergen filtering: {excluded_allergens}")

            # Step 3: 价格过滤
            if price_range and matched_dishes:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if price_range.get("min", 0) <= self.parse_price(dish.get("price")) <= price_range.get("max",
                                                                                                           float("inf"))
                ]
                logger.info(f"{len(matched_dishes)} dishes remain after price filtering: {price_range}")

            # Step 4: 构建文档（只基于最终保留的 matched_dishes）
            if matched_dishes:
                documents = [self.format_dish_info(d, self.get_nutrition_data(d["id"])) for d in matched_dishes]
            else:
                documents = ["No matching dishes found based on your preferences."]

            return {
                "documents": documents,
                "dishes": matched_dishes
            }

        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            return {"documents": ["Search Function is not available"], "dishes": []}
