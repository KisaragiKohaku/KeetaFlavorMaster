from chromadb.api.types import IncludeEnum
from sentence_transformers import SentenceTransformer
from intent_parser import parse_intent
import chromadb
import logging
import torch
import json
import os

logger = logging.getLogger(__name__)


def dynamic_chunking(text, max_chars=500):
    if len(text) <= max_chars:
        return [text]

    def smart_split(content, delimiter, max_size):
        segments = content.split(delimiter)
        chunks_list = []
        current = ""

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            if len(current) + len(seg) <= max_size:
                current += seg + delimiter
            else:
                if current:
                    chunks_list.append(current.strip())
                current = seg + delimiter

        if current:
            chunks_list.append(current.strip())
        return chunks_list

    chunks = smart_split(text, '\n', max_chars)

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            sentences = smart_split(chunk, '.', max_chars)
            final_chunks.extend(sentences)

    return final_chunks


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
        logger.info("Embedding Menu data with dynamic chunking...")
        try:
            documents, metadatas, ids = [], [], []
            chunk_count = 0
            long_text_dishes = 0

            for dish in self.menu_dict.values():
                dish_text = self.format_dish_info(dish)
                chunks = dynamic_chunking(dish_text)

                if len(chunks) > 1:
                    long_text_dishes += 1
                    chunk_count += len(chunks)

                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "dish_id": dish["id"],
                        "is_chunked": len(chunks) > 1,
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    })
                    ids.append(f"{dish['id']}_{idx}")

            if documents:
                embeddings = self.encoder.encode(documents).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(
                    f"Embedded {len(documents)} chunks"
                    f"({chunk_count} from {long_text_dishes} long dishes)"
                )
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

    def search(self, query: str, k: int = 10) -> dict:
        # noinspection PyBroadException
        try:
            intent_data = json.loads(parse_intent(query))
            excluded_info = [item for item in (intent_data.get("avoid") or []) if item is not None]
            price_data = intent_data.get("price") or [None, None]
            price_info = [item if item is not None else None for item in price_data]

            # Step 1: 向量检索
            query_embed = self.encoder.encode(query, normalize_embeddings=True)

            n_results = min(100, max(k * 2, len(query) // 10 + k))
            results = self.collection.query(
                query_embeddings=[query_embed.tolist()],
                n_results=n_results,
                include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances]
            )

            dish_scores = {}
            for i, meta in enumerate(results["metadatas"][0]):
                dish_id = meta["dish_id"]
                distance = results["distances"][0][i]

                # 计算相关性分数（距离转相似度）
                score = 1 / (1 + distance) if distance > 0 else 1.0

                # 更新菜品分数（取最高分）
                if dish_id not in dish_scores or score > dish_scores[dish_id]:
                    dish_scores[dish_id] = score

            # 按相关性排序
            sorted_dish_ids = sorted(
                dish_scores.keys(),
                key=lambda x: dish_scores[x],
                reverse=True
            )[:k]

            valid_dish_ids = [int(did) for did in sorted_dish_ids if int(did) in self.menu_dict]
            matched_dishes = [self.menu_dict[did] for did in valid_dish_ids]

            logger.info(f"{len(matched_dishes)} dishes matched vector search")

            # Step 2: 过敏原过滤
            if excluded_info:
                matched_dishes = [
                    dish for dish in matched_dishes
                    if all(
                        allergen not in (self.dish_allergen_map.get(dish["id"]) or [])
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
                    if min_price <= dish.get("price", 0) <= max_price
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

        except json.JSONDecodeError:
            logger.error("Intent parsing returned invalid JSON")
            return {"documents": ["Failed to understand the request"], "dishes": []}

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Search processing error: {str(e)}")
            return {"documents": ["Error processing the request"], "dishes": []}

        except Exception:
            logger.exception("Unexpected error during search")
            return {"documents": ["Search function is currently unavailable"], "dishes": []}
