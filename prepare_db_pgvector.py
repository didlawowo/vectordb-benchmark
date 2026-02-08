import asyncio
import json
import os

import asyncpg
import openlit
from loguru import logger
from pgvector.asyncpg import register_vector
from tenacity import retry, stop_after_attempt, wait_exponential

from common import DIMENSION

openlit.init(otlp_endpoint="http://127.0.0.1:4318")

TABLE_NAME = "benchmark"

PG_CONFIG = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "user": os.environ.get("PG_USER", "benchmark"),
    "password": os.environ.get("PG_PASSWORD", "benchmark"),
    "database": os.environ.get("PG_DATABASE", "benchmark"),
}

pool = None


async def init_pgvector():
    """Initialize PostgreSQL connection and create table with pgvector extension"""
    global pool
    try:
        pool = await asyncpg.create_pool(
            **PG_CONFIG, min_size=2, max_size=10, init=register_vector
        )

        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

            await conn.execute(f"""
                CREATE TABLE {TABLE_NAME} (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    dense_vector vector({DIMENSION}),
                    sparse_vector JSONB
                )
            """)
            logger.info(f"Created table: {TABLE_NAME}")

    except Exception as e:
        logger.error(f"Error with PostgreSQL initialization: {e}")
        raise


async def create_index():
    """Create HNSW index after data insertion for faster build"""
    async with pool.acquire() as conn:
        logger.info("Creating HNSW index on dense_vector (this may take a while)...")
        await conn.execute(f"""
            CREATE INDEX ON {TABLE_NAME}
            USING hnsw (dense_vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 128)
        """)
        logger.info("HNSW index created successfully")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def insert_batch(records):
    """Insert a batch of records with retry logic"""
    async with pool.acquire() as conn:
        await conn.executemany(
            f"""INSERT INTO {TABLE_NAME} (id, name, description, dense_vector, sparse_vector)
                VALUES ($1, $2, $3, $4, $5)""",
            records,
        )


async def load_and_insert_data(jsonl_path: str):
    """Load data from JSONL file and insert into PostgreSQL"""
    batch_size = 1000
    records = []
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

                # Convert CSR format to {str_index: weight} dict for JSONB storage
                sparse_vector = {
                    str(x[0]): float(x[1])
                    for x in zip(sparse_dict["indices"], sparse_dict["data"])
                }

                record = (
                    int(entry["id"]),
                    entry["name"],
                    entry["description"][:19900],
                    str(entry["dense_vector"]),
                    json.dumps(sparse_vector),
                )
                records.append(record)

            except json.JSONDecodeError as e:
                corrupted_lines += 1
                logger.error(f"Corrupted JSON at line {line_number}: {str(e)}")
                with open("corrupted_lines.txt", "a") as f_corrupt:
                    f_corrupt.write(f"Line {line_number}: {line}\n")
                    f_corrupt.write(f"Error: {str(e)}\n\n")
                continue

            if len(records) >= batch_size:
                logger.info(f"Inserting batch of {len(records)} entities...")
                try:
                    await insert_batch(records)
                    total_inserted += len(records)
                    logger.info(f"Progress: {total_inserted} entities inserted so far")
                except Exception as e:
                    logger.error(f"Error inserting batch after retries: {e}")
                    failed_batches += 1
                records = []

    # Insert remaining
    if records:
        logger.info(f"Inserting remaining {len(records)} entities...")
        try:
            await insert_batch(records)
            total_inserted += len(records)
        except Exception as e:
            logger.error(f"Error inserting final batch: {e}")
            failed_batches += 1

    logger.info("Insertion complete!")
    logger.info(f"Total entities inserted: {total_inserted}")
    logger.info(f"Total entries skipped due to empty vectors: {skipped_entries}")
    logger.info(f"Total corrupted lines found: {corrupted_lines}")
    logger.info(f"Total failed batches: {failed_batches}")

    async with pool.acquire() as conn:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        logger.info(f"Table row count: {count}")

    return total_inserted


async def main(jsonl_path: str):
    await init_pgvector()
    await load_and_insert_data(jsonl_path)
    await create_index()
    await pool.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and insert data into PostgreSQL with pgvector"
    )
    parser.add_argument("jsonl_path", help="Path to the JSONL file to load")
    args = parser.parse_args()

    asyncio.run(main(args.jsonl_path))
