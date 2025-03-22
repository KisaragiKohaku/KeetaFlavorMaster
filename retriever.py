from chromadb.api.types import IncludeEnum
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
import chromadb
import logging
import torch
import json
import os
import re
import numpy as np

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
            cache_folder="models/sentence-transformers")
        self.client = chromadb.PersistentClient(path="./chroma_db")

        self._init_collection()

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

            self._validate_data_consistency(nutrition_data)

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

    def _validate_data_consistency(self, nutrition_data: List[dict]):
        menu_ids = set(self.menu_dict.keys())
        nutrition_ids = {n["dish_id"] for n in nutrition_data}

        if missing := menu_ids - nutrition_ids:
            logger.warning(f"Dish ID without nutritional information: {missing}")
        if extra := nutrition_ids - menu_ids:
            logger.warning(f"Undefined Dish ID: {extra}")

    def parse_price(self, value):
        try:
            return float(re.sub(r"[^\d.]", "", str(value)))
        except:
            return float("inf")

    def search(self, query: str, k: int = 5) -> Dict[str, Union[List[str], List[dict]]]:
        try:
            query_embed = self.encoder.encode(query, normalize_embeddings=True)

            # 改进后的价格意图识别逻辑（包含最便宜/最贵意图）
            intent_phrases = {
                "low_price": ["cheap", "affordable", "budget-friendly", "not expensive",
                              "low cost", "economical", "cost-effective"],
                "high_price": ["luxury", "high-end", "expensive", "premium",
                               "fine dining", "most luxurious", "top-tier"],
                "lowest_price": ["cheapest", "lowest price", "least expensive",
                                 "absolutely cheapest", "give me the cheapest one"],
                "highest_price": ["most expensive", "absolutely most expensive",
                                  "give me the priciest one", "top priced", "highest price"]
            }

            def has_intent(query_embed, phrases):
                phrase_embeds = self.encoder.encode(phrases, normalize_embeddings=True)
                scores = np.dot(phrase_embeds, query_embed)
                return np.max(scores) > 0.6

            is_low_price = has_intent(query_embed, intent_phrases["low_price"])
            is_high_price = has_intent(query_embed, intent_phrases["high_price"])
            is_lowest_price = has_intent(query_embed, intent_phrases["lowest_price"])
            is_highest_price = has_intent(query_embed, intent_phrases["highest_price"])
            has_price_request = is_low_price or is_high_price or is_lowest_price or is_highest_price

            # 获取向量并搜索
            results = self.collection.query(
                query_embeddings=[query_embed.tolist()],
                n_results=k,
                include=[IncludeEnum.documents, IncludeEnum.metadatas]
            )

            documents = results["documents"][0] if results["documents"] else []
            matched_dishes = [
                self.menu_dict[meta["dish_id"]]
                for meta in results["metadatas"][0]
                if meta["dish_id"] in self.menu_dict
            ]

            if has_price_request and matched_dishes:
                if is_lowest_price:
                    matched_dishes = [min(matched_dishes, key=lambda dish: self.parse_price(dish.get("price")))]
                elif is_highest_price:
                    matched_dishes = [max(matched_dishes, key=lambda dish: self.parse_price(dish.get("price")))]
                else:
                    matched_dishes = sorted(
                        matched_dishes,
                        key=lambda dish: self.parse_price(dish.get("price")),
                        reverse=is_high_price
                    )[:5]
                documents = ["Here are some dishes selected based on your price preference:"]

            return {
                "documents": documents or ["No matching information exists"],
                "dishes": matched_dishes or []
            }

        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            return {"documents": ["Search Function is not available"], "dishes": []}
