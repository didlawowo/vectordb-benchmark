import json
import os
from loguru import logger
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
    connections,
    db,
    Collection,
)
from FlagEmbedding import BGEM3FlagModel
from common import COLLECTION_NAME, DATABASE_NAME, TESTSET_FILE, TESTSET_SOURCE_FILE

logger.level("INFO")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")

connections.connect(
    host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT")
)


def get_embeddings(query_text: str):
    # Generate embeddings for query
    embeddings = model.encode([query_text], return_dense=True, return_sparse=True)

    # Prepare dense vector
    dense_vector = embeddings["dense_vecs"][0].tolist()
    sparse_vector = {
        int(idx): float(weight)
        for idx, weight in embeddings["lexical_weights"][0].items()
    }
    return dense_vector, sparse_vector


def search_games(query_text: str, top_k: int = 10) -> list[dict]:
    """Search for games using hybrid search"""
    db.using_database(DATABASE_NAME)
    logger.info(f"Milvus database initialized with {DATABASE_NAME}")

    dense_vector, sparse_vector = get_embeddings(query_text)
    # Connect to collection
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # Hybrid search
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        [sparse_vector], "sparse_vector", sparse_search_params, limit=50
    )
    dense_search_params = {"metric_type": "COSINE"}
    dense_req = AnnSearchRequest(
        [dense_vector], "dense_vector", dense_search_params, limit=50
    )

    results = collection.hybrid_search(
        [sparse_req, dense_req],
        rerank=WeightedRanker(0.45, 0.55),
        output_fields=["name", "description"],
        limit=top_k,
    )

    # Format results
    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append(
                {
                    "name": hit.entity.get("name"),
                    "description": hit.entity.get("description"),
                    "score": hit.score,
                }
            )

    return formatted_results


def generate_testset(source_file: str) -> list[dict]:
    testset = []
    with open(source_file, "r") as file:
        queries = file.readlines()

    for query in queries:
        dense_vector, sparse_vector = get_embeddings(query)
        testset.append(
            {
                "query": query,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
        )
    return testset


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Milvus database operations")
    parser.add_argument(
        "action",
        choices=["test", "generate_testset"],
        help="test: run search query, generate_testset: generate testset",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query to use with test action",
        default="A game with Solid Snake where you need to rescue hostages",
    )
    args = parser.parse_args()

    if args.action == "test":
        if not args.query:
            parser.error("--query is required when using 'test' action")

        logger.info(f"Running test search for: '{args.query}'")
        results = search_games(args.query)

        print(f"\nSearch results for: '{args.query}'")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']} (Score: {result['score']:.3f})")
            print(f"   {result['description'][:200]}...")

    elif args.action == "generate_testset":
        logger.info(f"Generating testset from {TESTSET_SOURCE_FILE}")
        testset = generate_testset(TESTSET_SOURCE_FILE)

        with open(TESTSET_FILE, "w") as file:
            json.dump(testset, file, indent=4)
        logger.info(f"Testset generated and saved to {TESTSET_FILE}")
