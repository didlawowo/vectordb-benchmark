import json
import os
import random
import time
from locust import User, task, between, events
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config.common import COLLECTION_NAME
import openlit

logger = logging.getLogger(__name__)

openlit.init(otlp_endpoint="http://127.0.0.1:4318")


@events.request.add_listener
def on_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    **kwargs,
):
    if exception:
        logger.error(
            f"Request failed: {request_type} {name} took {response_time}ms - {str(exception)}"
        )


# Chargement des données de test
testset = []
with open("data/testset.json", "r") as file:
    testset = json.load(file)

def _calculate_recall(self, results, ground_truth, k=10):
    """
    📊 Calcule le recall@k
    """
    retrieved_ids = set(r["id"] for r in results[:k])
    relevant_ids = set(ground_truth)
    
    if not relevant_ids:
        return 0.0
        
    return len(retrieved_ids.intersection(relevant_ids)) / len(relevant_ids)

def _record_metrics(self, recall, latency):
    """
    📈 Enregistre les métriques pour analyse ultérieure
    """
    self.metrics["recall_at_10"].append(recall)
    self.metrics["latency"].append(latency)
    
    
def calculate_sparse_score(query_sparse, doc_sparse):
    try:
        # Vérification de la structure du vecteur sparse
        if (
            isinstance(query_sparse, dict)
            and "indices" in query_sparse
            and "data" in query_sparse
        ):
            query_dict = {
                str(idx): val
                for idx, val in zip(query_sparse["indices"], query_sparse["data"])
            }
        else:
            # Si le format n'est pas celui attendu, on considère que query_sparse est déjà un dictionnaire
            query_dict = query_sparse

        return sum(query_dict.get(str(k), 0) * v for k, v in doc_sparse.items())
    except Exception as e:
        logger.error(f"Error in calculate_sparse_score: {e}")
        return 0


class SearchQdrantUser(User):
    wait_time = between(1, 3)

    def on_start(self):
        try:
            self.client = QdrantClient(
                host=os.environ.get("QDRANT_HOST", "localhost"),
                port=os.environ.get("QDRANT_PORT", 6333),
                timeout=60,
                prefer_grpc=True,
            )
            logger.info(f"Qdrant collection initialized: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant connection: {e}")
            raise Exception(f"Qdrant initialization failed: {e}")

    @task(weight=3)
    def test_dense_search(self):
        start_time = time.time()
        
        try:
            test = random.choice(testset)
            dense_vector = test["dense_vector"]
            ground_truth = test.get("ground_truth", [])  # 📝 Add ground truth to testset

            response = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=dense_vector,
                limit=50,
                with_payload=["id", "name"],  # 🔍 Only fetch needed fields
                search_params=models.SearchParams(hnsw_ef=128)
            )
            
            # 📊 Calculate metrics
            recall_at_k = calculate_recall(response, ground_truth, k=10)
            
        except Exception as e:
            logger.error(f"❌ Dense search error: {e}")
            raise
        finally:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="Qdrant",
                name="dense_search",
                response_time=total_time,
                response_length=len(response),
                context={"recall": recall_at_k}
            )

    @task
    def test_hybrid_search_on_qdrant(self):
        start_time = time.time()
        formatted_results = []

        try:
            test = random.choice(testset)
            sparse_vector = test["sparse_vector"]
            dense_vector = test["dense_vector"]

            # Recherche par vecteur dense
            response = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=dense_vector,
                limit=50,
                with_payload=["name", "description", "sparse_vector"],
                search_params=models.SearchParams(hnsw_ef=128),
            )

            # Calcul des scores hybrides
            hybrid_results = []
            for hit in response:
                doc_sparse = hit.payload.get("sparse_vector", {})
                sparse_score = calculate_sparse_score(sparse_vector, doc_sparse)
                hybrid_score = 0.45 * sparse_score + 0.55 * hit.score

                hybrid_results.append(
                    {
                        "name": hit.payload.get("name"),
                        "description": hit.payload.get("description"),
                        "dense_score": hit.score,
                        "sparse_score": sparse_score,
                        "hybrid_score": hybrid_score,
                    }
                )

            # Tri et sélection des meilleurs résultats
            hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            formatted_results = [
                {
                    "name": res["name"],
                    "description": res["description"],
                    "score": res["hybrid_score"],
                }
                for res in hybrid_results[:10]
            ]

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")

        finally:
            # Calcul du temps total et envoi des événements
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="Qdrant",
                name="hybrid_search",
                response_time=total_time,
                response_length=len(formatted_results),
                response=formatted_results,
                context={},
                exception=None if formatted_results else Exception("No results found"),
            )
