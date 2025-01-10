import os
import json
from loguru import logger
from typing import List, Dict
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from fastembed import TextEmbedding
from FlagEmbedding import BGEM3FlagModel

logger.level("INFO")

# Initialisation des modèles d'embedding
dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
sparse_model = TextEmbedding("BAAI/bge-m3")

# Configuration des clients Qdrant
client = QdrantClient(url=os.environ["DB_URL"])


# Configuration du vector store
vector_store = QdrantVectorStore(
    collection_name=os.environ["COLLECTION_NAME"],
    client=client,
    aclient=client,
    enable_hybrid=True,
    batch_size=50,
)


def get_embeddings(query_text: str):
    """Génère les embeddings denses et sparses"""
    # BGE-M3 pour les embeddings denses et sparses
    dense_embeddings = dense_model.encode(
        [query_text], return_dense=True, return_sparse=True
    )
    dense_vector = dense_embeddings["dense_vecs"][0].tolist()
    sparse_vector = {
        int(idx): float(weight)
        for idx, weight in dense_embeddings["lexical_weights"][0].items()
    }

    return dense_vector, sparse_vector


def search_games(query_text: str, top_k: int = 10) -> List[Dict]:
    """Recherche hybride asynchrone dans Qdrant"""
    logger.info(f"Searching for: {query_text}")

    try:
        dense_vector, sparse_vector = get_embeddings(query_text)

        # Recherche hybride avec le vector store
        response = client.search(
            collection_name=os.environ["COLLECTION_NAME"],
            query_vector=dense_vector,
            query_filter=None,
            limit=50,
            with_payload=True,
            search_params={"exact": False, "hnsw_ef": 128},
        )

        # Calcul des scores hybrides
        hybrid_results = []
        for hit in response:
            doc_sparse = hit.payload.get("sparse_vector", {})
            sparse_score = sum(
                sparse_vector.get(k, 0) * v for k, v in doc_sparse.items()
            )
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
        return hybrid_results[:top_k]

    except Exception as e:
        logger.error(f"Error during search: {e}")
        return []


async def generate_testset(source_file: str) -> List[Dict]:
    """Génère un ensemble de test avec les embeddings"""
    testset = []

    try:
        with open(source_file, "r") as file:
            queries = file.readlines()

        for query in queries:
            query = query.strip()
            dense_vector, sparse_vector = get_embeddings(query)

            testset.append(
                {
                    "query": query,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                }
            )

        return testset

    except Exception as e:
        logger.error(f"Error generating testset: {e}")
        return []


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qdrant vector store operations")
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
        results = await search_games(args.query)

        print(f"\nSearch results for: '{args.query}'")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
            print(f"   Dense Score: {result['dense_score']:.3f}")
            print(f"   Sparse Score: {result['sparse_score']:.3f}")
            print(f"   Description: {result['description'][:200]}...")

    elif args.action == "generate_testset":
        logger.info(f"Generating testset from {os.environ.get('TESTSET_SOURCE_FILE')}")
        testset = await generate_testset(os.environ.get("TESTSET_SOURCE_FILE"))

        if testset:
            with open(os.environ.get("TESTSET_FILE"), "w") as file:
                json.dump(testset, file, indent=4)
            logger.info(
                f"Testset generated and saved to {os.environ.get('TESTSET_FILE')}"
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
