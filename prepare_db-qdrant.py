import os
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from common import DIMENSION, COLLECTION_NAME
import openlit

openlit.init(otlp_endpoint="http://127.0.0.1:4318")


def init_qdrant():
    """Initialize Qdrant connection"""
    try:
        # Augmenter le timeout pour le client
        global client
        client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=os.environ.get("QDRANT_PORT", 6333),
            timeout=300,  # 5 minutes timeout
        )

        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if COLLECTION_NAME in collection_names:
            logger.info(f"Collection {COLLECTION_NAME} already exists")
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection {COLLECTION_NAME}")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=DIMENSION, distance=models.Distance.COSINE
            ),
        )
        logger.info(f"Created new collection: {COLLECTION_NAME}")

    except Exception as e:
        logger.error(f"Error with Qdrant initialization: {e}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def insert_batch(points):
    """Insert a batch of points with retry logic"""
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def load_and_insert_data(jsonl_path: str):
    """Load data from JSONL file and insert into Qdrant"""
    batch_size = 1000  # Réduit de 10000 à 1000
    points = []

    total_inserted = 0
    skipped_entries = 0
    corrupted_lines = 0
    failed_batches = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                entry = json.loads(line)

                sparse_dict = entry["sparse_vector"]
                if not sparse_dict["data"] or not sparse_dict["indices"]:
                    logger.warning(
                        f"Skipping entry at line {line_number} due to empty sparse vector"
                    )
                    skipped_entries += 1
                    continue

                sparse_vector = {
                    str(x[0]): float(x[1])
                    for x in zip(sparse_dict["indices"], sparse_dict["data"])
                }

                point = models.PointStruct(
                    id=int(entry["id"]),
                    vector=entry["dense_vector"],
                    payload={
                        "name": entry["name"],
                        "description": entry["description"][:19900],
                        "sparse_vector": sparse_vector,
                    },
                )
                points.append(point)

            except json.JSONDecodeError as e:
                corrupted_lines += 1
                logger.error(f"Corrupted JSON at line {line_number}: {str(e)}")
                with open("corrupted_lines.txt", "a") as f_corrupt:
                    f_corrupt.write(f"Line {line_number}: {line}\n")
                    f_corrupt.write(f"Error: {str(e)}\n\n")
                continue

            # Insert in batches
            if len(points) >= batch_size:
                logger.info(f"Inserting batch of {len(points)} entities...")
                try:
                    insert_batch(points)  # Utilise la fonction avec retry
                    total_inserted += len(points)
                    logger.info(f"Progress: {total_inserted} entities inserted so far")
                except Exception as e:
                    logger.error(f"Error inserting batch after retries: {e}")
                    logger.debug(f"Sample data that failed: ID={points[0].id}")
                    failed_batches += 1
                points = []

    # Insert remaining entries
    if points:
        logger.info(f"Inserting remaining {len(points)} entities...")
        try:
            insert_batch(points)
            total_inserted += len(points)
        except Exception as e:
            logger.error(f"Error inserting final batch: {e}")
            failed_batches += 1

    logger.info("Insertion complete!")
    logger.info(f"Total entities inserted: {total_inserted}")
    logger.info(f"Total entries skipped due to empty vectors: {skipped_entries}")
    logger.info(f"Total corrupted lines found: {corrupted_lines}")
    logger.info(f"Total failed batches: {failed_batches}")

    # Vérification finale
    collection_info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Collection count: {collection_info.points_count}")

    return total_inserted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and insert data into Qdrant")
    parser.add_argument("jsonl_path", help="Path to the JSONL file to load")
    args = parser.parse_args()

    init_qdrant()
    load_and_insert_data(args.jsonl_path)
