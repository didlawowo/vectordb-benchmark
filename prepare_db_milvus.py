import os
from loguru import logger
from pymilvus import (
    connections,
    db,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
import json
from scipy.sparse import csr_array

from common import DATASET_EMBEDDING_FILE, DIMENSION, COLLECTION_NAME, DATABASE_NAME

logger.level("INFO")

connections.connect(
    host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT")
)


def init_milvus():
    """Initialize Milvus connection and create database"""
    try:
        # Vérifier si la base de données existe déjà
        existing_dbs = db.list_database()
        if DATABASE_NAME in existing_dbs:
            logger.info(f"Database {DATABASE_NAME} already exists, using it")
        else:
            db.create_database(DATABASE_NAME)
            logger.info(f"Created new database: {DATABASE_NAME}")

        db.using_database(DATABASE_NAME)
        logger.info(f"Using database: {DATABASE_NAME}")

    except Exception as e:
        logger.error(f"Error with database initialization: {e}")
        raise


def dict_to_csr(d):
    return csr_array((d["data"], d["indices"], d["indptr"]), shape=d["shape"])


def create_collection():
    """Create Milvus collection with schema for both dense and sparse vectors"""
    logger.info(f"Creating collection {COLLECTION_NAME}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=250),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=20000),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(
        fields=fields, description="Game collection with hybrid search"
    )

    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"Dropping collection {COLLECTION_NAME} because it already exists")
        utility.drop_collection(COLLECTION_NAME)

    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create indexes
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "FLAT", "metric_type": "COSINE"}
    collection.create_index("dense_vector", dense_index)

    return collection


def load_and_insert_data(collection: Collection, jsonl_path: str):
    """Load data from JSONL file and insert into Milvus"""
    entities = {
        "id": [],
        "name": [],
        "description": [],
        "dense_vector": [],
        "sparse_vector": [],
    }

    total_inserted = 0
    skipped_entries = 0
    corrupted_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                # logger.debug("Populating entities...")
                entry = json.loads(line)

                # Vérifier si le vecteur sparse est vide
                sparse_dict = entry["sparse_vector"]
                if not sparse_dict["data"] or not sparse_dict["indices"]:
                    logger.warning(
                        f"Skipping entry at line {line_number} due to empty sparse vector"
                    )
                    skipped_entries += 1
                    continue

                sparse_vector = {
                    x[0]: x[1] for x in zip(sparse_dict["indices"], sparse_dict["data"])
                }

                entities["id"].append(int(entry["id"]))
                entities["name"].append(entry["name"])
                entities["description"].append(entry["description"][:9900])
                entities["dense_vector"].append(entry["dense_vector"])
                entities["sparse_vector"].append(sparse_vector)

            except json.JSONDecodeError as e:
                corrupted_lines += 1
                logger.error(f"Corrupted JSON at line {line_number}: {str(e)}")
                # Optionnel : sauvegarder les lignes corrompues pour analyse
                with open("corrupted_lines.txt", "a") as f_corrupt:
                    f_corrupt.write(f"Line {line_number}: {line}\n")
                    f_corrupt.write(f"Error: {str(e)}\n\n")
                continue

            # Insert in batches of 10000
            if len(entities["id"]) >= 10000:
                logger.info(f"Inserting batch of {len(entities['id'])} entities...")
                try:
                    insert_data = [
                        entities["id"],
                        entities["name"],
                        entities["description"],
                        entities["dense_vector"],
                        entities["sparse_vector"],
                    ]
                    collection.insert(insert_data)
                    total_inserted += len(entities["id"])
                    logger.info(f"Progress: {total_inserted} entities inserted so far")
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    logger.debug(
                        f"Sample data that failed: ID={entities['id'][0]}, Name={entities['name'][0]}"
                    )
                entities = {k: [] for k in entities}

    # Insert remaining entries
    if entities["id"]:
        logger.info(f"Inserting remaining {len(entities['id'])} entities...")
        try:
            insert_data = [
                entities["id"],
                entities["name"],
                entities["description"],
                entities["dense_vector"],
                entities["sparse_vector"],
            ]
            collection.insert(insert_data)
            total_inserted += len(entities["id"])
        except Exception as e:
            logger.error(f"Error inserting final batch: {e}")

    collection.flush()
    logger.info(f"Insertion complete!")
    logger.info(f"Total entities inserted: {total_inserted}")
    logger.info(f"Total entries skipped due to empty vectors: {skipped_entries}")
    logger.info(f"Total corrupted lines found: {corrupted_lines}")

    # Vérification finale
    logger.info(f"Verifying collection count...")
    collection_count = collection.num_entities
    logger.info(f"Collection count: {collection_count}")

    return total_inserted


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Milvus database operations")
    parser.add_argument(
        "action",
        choices=["init"],
        help="init: initialize and populate database, test: run search query",
    )
    args = parser.parse_args()

    if args.action == "init":
        # Initialize and setup
        logger.info("Initializing Milvus database...")
        init_milvus()
        collection = create_collection()

        # Load and insert data
        logger.info("Loading and inserting data...")
        load_and_insert_data(collection, DATASET_EMBEDDING_FILE)
        logger.info("Database initialization complete!")
