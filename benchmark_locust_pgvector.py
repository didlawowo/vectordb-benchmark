import asyncio
import json
import logging
import os
import random
import threading
import time

import asyncpg
import openlit
from locust import User, between, events, task
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

openlit.init(otlp_endpoint="http://127.0.0.1:4318")

TABLE_NAME = "benchmark"


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


# Chargement des donnees de test
testset = []
with open("data/testset.json", "r") as file:
    testset = json.load(file)


def calculate_sparse_score(query_sparse, doc_sparse):
    try:
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
            query_dict = query_sparse

        return sum(query_dict.get(str(k), 0) * v for k, v in doc_sparse.items())
    except Exception as e:
        logger.error(f"Error in calculate_sparse_score: {e}")
        return 0


class SearchPgvectorUser(User):
    wait_time = between(1, 3)

    def on_start(self):
        try:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()

            future = asyncio.run_coroutine_threadsafe(self._init_pool(), self._loop)
            future.result(timeout=30)
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection: {e}")
            raise Exception(f"PostgreSQL initialization failed: {e}")

    async def _init_pool(self):
        self.pool = await asyncpg.create_pool(
            host=os.environ.get("PG_HOST", "localhost"),
            port=int(os.environ.get("PG_PORT", 5432)),
            user=os.environ.get("PG_USER", "benchmark"),
            password=os.environ.get("PG_PASSWORD", "benchmark"),
            database=os.environ.get("PG_DATABASE", "benchmark"),
            min_size=2,
            max_size=10,
            init=register_vector,
        )

    def _run_async(self, coro):
        """Run an async coroutine from the gevent greenlet"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60)

    async def _hybrid_search(self, dense_vector, sparse_vector):
        """Execute dense search via pgvector, then rerank with sparse scores"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, name, description, sparse_vector,
                       1 - (dense_vector <=> $1::vector) AS dense_score
                FROM {TABLE_NAME}
                ORDER BY dense_vector <=> $1::vector
                LIMIT 50
                """,
                str(dense_vector),
            )

        hybrid_results = []
        for row in rows:
            doc_sparse = json.loads(row["sparse_vector"]) if isinstance(row["sparse_vector"], str) else row["sparse_vector"]
            sparse_score = calculate_sparse_score(sparse_vector, doc_sparse)
            hybrid_score = 0.45 * sparse_score + 0.55 * row["dense_score"]

            hybrid_results.append(
                {
                    "name": row["name"],
                    "description": row["description"],
                    "dense_score": row["dense_score"],
                    "sparse_score": sparse_score,
                    "hybrid_score": hybrid_score,
                }
            )

        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return [
            {
                "name": r["name"],
                "description": r["description"],
                "score": r["hybrid_score"],
            }
            for r in hybrid_results[:10]
        ]

    @task
    def test_hybrid_search_on_pgvector(self):
        start_time = time.time()
        formatted_results = []

        try:
            test = random.choice(testset)
            sparse_vector = test["sparse_vector"]
            dense_vector = test["dense_vector"]

            formatted_results = self._run_async(
                self._hybrid_search(dense_vector, sparse_vector)
            )

        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")

        finally:
            total_time = int((time.time() - start_time) * 1000)
            events.request.fire(
                request_type="pgvector",
                name="hybrid_search",
                response_time=total_time,
                response_length=len(formatted_results),
                response=formatted_results,
                context={},
                exception=None if formatted_results else Exception("No results found"),
            )
