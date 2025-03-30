from chromadb.api.types import IncludeEnum
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
from intent_parser import parse_intent
import chromadb
import logging
import torch
import json
import os
import re

logger = logging.getLogger(__name__)

def _format_document(dish, item):
    return (
        f"Dish: {dish['name']}\n"
        f"Price: {dish['price']}HKD\n"
        f"Ingredients: {dish['ingredients']}\n"
        f"Tags: {dish['tags']}\n"
        f"Serving Time: {dish['serving_time']}\n"
        f"Calories: {item['calories']}kcal\n"
        f"Protein: {item['protein']}g\n"
        f"Carbs: {item['carbs']}g\n"
        f"Fat: {item['fat']}\n"
        f"Allergens: {item['allergens']}\n"
        f"Description: {item['description']}"
    )


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
                    documents.append(_format_document(dish, item))
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

    def search(self, query: str, k: int = 2) -> Dict[str, Union[List[str], List[dict]]]:
        try:
            intent_data = json.loads(parse_intent(query))
        
            # 提取过敏原
            excluded_allergens = intent_data.get('allergens', [])
        
            # 提取价格过滤信息
            price_range = intent_data.get('price_range', None)
        
            # 向量检索
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
        
            # 根据DeepSeek的过敏原过滤
            if excluded_allergens:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if all(
                        allergen.lower() not in self.dish_allergen_map.get(dish["id"], [])
                        for allergen in excluded_allergens
                    )
                ]
        
            # 根据DeepSeek的价格过滤
            if price_range:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if price_range["min"] <= self.parse_price(dish.get("price")) <= price_range["max"]
                ]
        
            documents = [self.format_dish_info(d) for d in matched_dishes]
        
            return {
                "documents": documents if documents else ["No matching dishes found based on your query."],
                "dishes": matched_dishes
            }


        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            return {"documents": ["Search Function is not available"], "dishes": []}
