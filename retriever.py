from chromadb.api.types import IncludeEnum
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
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

    # nutrition related function
    def _build_allergen_lookup(self):
        self.dish_allergen_map = {}
        try:
            with open("data/nutrition.json") as f:
                nutrition_data = json.load(f)["nutrition_data"]
                for item in nutrition_data:
                    self.dish_allergen_map[item["dish_id"]] = [a.lower() for a in item.get("allergens", [])]
        except Exception as e:
            logger.error(f"Failed to load allergen info: {str(e)}")

    def detect_allergen_intent(self, query: str) -> List[str]:
        known_allergens = [
            "chicken", "dairy", "egg", "fish", "gluten", "milk", "mushroom",
            "peanut", "pork", "shellfish", "shrimp", "soy", "wheat"
        ]
        found = []
        for allergen in known_allergens:
            if re.search(rf"\b(allergic to|no|avoid|don’t eat|cannot eat|can’t eat)\s+{allergen}\b", query):
                found.append(allergen)
            if allergen.endswith("y"):
                plural = allergen[:-1] + "ies"
            else:
                plural = allergen + "s"
            if re.search(rf"\b(allergic to|no|avoid|don’t eat|cannot eat|can’t eat)\s+{plural}\b", query):
                found.append(allergen)
        return list(set(found))

    # price related function
    def parse_price(self, value):
        try:
            return float(re.sub(r"[^\d.]", "", str(value)))
        except:
            return float("inf")

    def get_price_filter_type(self, query: str) -> Optional[Dict[str, Union[str, float]]]:
        query_lower = query.lower()
        range_patterns = [
            r'\bbetween\s+(\d+\.?\d*)\s+and\s+(\d+\.?\d*)',
            r'\bfrom\s+(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*[~～]\s*(\d+\.?\d*)'
        ]
        for pattern in range_patterns:
            if match := re.search(pattern, query_lower):
                try:
                    low_val = float(match.group(1))
                    high_val = float(match.group(2))
                except:
                    break
                if low_val > high_val:
                    low_val, high_val = high_val, low_val
                return {"type": "range", "min": low_val, "max": high_val}

        lt_patterns = [
            r'less than\s+(\d+\.?\d*)',
            r'under\s+(\d+\.?\d*)',
            r'below\s+(\d+\.?\d*)',
            r'at most\s+(\d+\.?\d*)',
        ]
        for pattern in lt_patterns:
            if match := re.search(pattern, query_lower):
                try:
                    value = float(match.group(1))
                except:
                    break
                return {"type": "lt", "value": value}

        gt_patterns = [
            r'more than\s+(\d+\.?\d*)',
            r'over\s+(\d+\.?\d*)',
            r'above\s+(\d+\.?\d*)',
            r'at least\s+(\d+\.?\d*)',
        ]
        for pattern in gt_patterns:
            if match := re.search(pattern, query_lower):
                try:
                    value = float(match.group(1))
                except:
                    break
                return {"type": "gt", "value": value}

        if re.search(r'cheapest|least expensive|lowest price|lowest priced',
                     query_lower):
            return {"type": "min"}
        if re.search(r'most expensive|priciest|costliest',
                     query_lower):
            return {"type": "max"}

        low_keywords = ["cheap", "affordable", "economical", "budget", "inexpensive", "low price", "most affordable"]
        high_keywords = ["expensive", "premium", "luxurious", "luxury", "pricey", "most luxurious"]
        if any(k in query_lower for k in low_keywords):
            return {"type": "low"}
        if any(k in query_lower for k in high_keywords):
            return {"type": "high"}
        return None

    def filter_by_price(self, dishes: List[dict], filter_spec: Dict[str, Union[str, float]]) -> List[dict]:
        if not dishes or not filter_spec:
            return dishes

        ftype = filter_spec.get("type")
        if ftype == "range":
            min_val = filter_spec.get("min", float("-inf"))
            max_val = filter_spec.get("max", float("inf"))
            return [d for d in dishes if min_val <= self.parse_price(d.get("price")) <= max_val][:5]

        if ftype == "lt":
            threshold = filter_spec.get("value", float("inf"))
            return [d for d in dishes if self.parse_price(d.get("price")) <= threshold][:5]

        if ftype == "gt":
            threshold = filter_spec.get("value", float("-inf"))
            return [d for d in dishes if self.parse_price(d.get("price")) >= threshold][:5]

        valid_dishes = [d for d in dishes if self.parse_price(d.get("price")) != float("inf")]
        if not valid_dishes:
            return []

        if ftype == "min":
            min_price = min(self.parse_price(d.get("price")) for d in valid_dishes)
            return [d for d in valid_dishes if self.parse_price(d.get("price")) == min_price][:5]

        if ftype == "max":
            max_price = max(self.parse_price(d.get("price")) for d in valid_dishes)
            return [d for d in valid_dishes if self.parse_price(d.get("price")) == max_price][:5]

        if ftype == "low":
            return sorted(valid_dishes, key=lambda d: self.parse_price(d.get("price")))[:5]

        if ftype == "high":
            return sorted(valid_dishes, key=lambda d: self.parse_price(d.get("price")), reverse=True)[:5]

        return dishes

    def search(self, query: str, k: int = 2) -> Dict[str, Union[List[str], List[dict]]]:
        try:
            query_embed = self.encoder.encode(query, normalize_embeddings=True)

            # Step 1: detect allergen
            excluded_allergens = self.detect_allergen_intent(query.lower())

            # Step 2: price filter detection
            filter_info = self.get_price_filter_type(query.lower())

            # Step 3: vector search
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

            # Step 4: allergen filtering
            if excluded_allergens:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if all(
                        allergen not in self.dish_allergen_map.get(dish["id"], [])
                        for allergen in excluded_allergens
                    )
                ]
                logger.info(f"{len(matched_dishes)} dishes remain after excluding allergens: {excluded_allergens}")

            # Step 5: price filtering
            if filter_info and matched_dishes:
                matched_dishes = self.filter_by_price(matched_dishes, filter_info)
                documents = ["Here are some dishes selected based on your preferences."]

            return {
                "documents": documents or ["No matching information exists"],
                "dishes": matched_dishes or []
            }

        except Exception as e:
            logger.error(f"Failed to search: {str(e)}")
            return {"documents": ["Search Function is not available"], "dishes": []}
