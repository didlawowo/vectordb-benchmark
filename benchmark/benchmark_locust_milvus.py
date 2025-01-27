import json
import os
import random
import time
from locust import User, task, between, events
import logging
from pymilvus import AnnSearchRequest, Collection, WeightedRanker, connections, db
from config.common import COLLECTION_NAME, DATABASE_NAME
import grpc.experimental.gevent as grpc_gevent

grpc_gevent.init_gevent()
logger = logging.getLogger(__name__)


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
    # else:
    #     logger.info(f"Request success: {request_type} {name} took {response_time}ms")


testset = []

with open("data/testset.json", "r") as file:
    testset = json.load(file)


class SearchMilvusUser(User):
    wait_time = between(1, 3)

    def on_start(self):
        try:
            connections.connect(
                host=os.environ.get("MILVUS_HOST"), port=19530
            )
            db.using_database(DATABASE_NAME)
            logger.info(f"Milvus database initialized with {DATABASE_NAME}")
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
        except Exception as e:
            logger.error(f"Failed to initialize Milvus connection: {e}")
            events.request.fire(
                request_type="Milvus",
                name="initialization",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
                response=None,
            )
            raise Exception(f"Milvus initialization failed: {e}")

    @task
    def test_search_on_milvus(self):
        start_time = time.time()

        try:
            test = random.choice(testset)
            sparse_vector = test["sparse_vector"]
            dense_vector = test["dense_vector"]

            # Start timing the request
            start_time = time.time()

            # Hybrid search
            sparse_search_params = {"metric_type": "IP"}
            sparse_req = AnnSearchRequest(
                [sparse_vector], "sparse_vector", sparse_search_params, limit=50
            )
            dense_search_params = {"metric_type": "COSINE"}
            dense_req = AnnSearchRequest(
                [dense_vector], "dense_vector", dense_search_params, limit=50
            )
            # logger.info("Hybrid search with sparse vector and dense vector")

            response = self.collection.hybrid_search(
                [sparse_req, dense_req],
                rerank=WeightedRanker(0.45, 0.55),
                output_fields=["name", "description"],
                limit=10,
            )
            formatted_results = []
            for hits in response:
                for hit in hits:
                    formatted_results.append(
                        {
                            "name": hit.entity.get("name"),
                            "description": hit.entity.get("description"),
                            "score": hit.score,
                        }
                    )
            # Calculate response time
            total_time = int((time.time() - start_time) * 1000)

            # Report success event
            if len(formatted_results) == 0:
                logger.error("No results found")
                events.request.fire()
                events.request.fire(
                    request_type="Milvus",
                    name="hybrid_search",
                    response_time=total_time,
                    response_length=10,
                    response=formatted_results,
                    context={},
                    exception=Exception("No results found"),
                )
            else:
                # logger.info(f"Results found: {formatted_results[0]}")
                events.request.fire(
                    request_type="Milvus",
                    name="hybrid_search",
                    response_time=total_time,
                    response_length=10,
                    response=formatted_results,
                    context={},
                    exception=None,
                )

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            # Report failure event
            events.request.fire(
                request_type="Milvus",
                name="hybrid_search",
                response_time=total_time,
                response_length=0,
                exception=e,
                context={},
                response=None,
            )
