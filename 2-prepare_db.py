import os
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from icecream import ic
import chromadb
from pymilvus import (
    connections,
    db,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


# 🔧 Configuration class
class Config:
    def __init__(self, backend: str = "milvus"):
        self.backend = backend
        # 🔌 Connection settings common
        self.host = os.getenv("VECTOR_HOST", "localhost")

        self.store_name = os.getenv("STORE_NAME", "vector_store")
        self.batch_size = int(os.getenv("BATCH_SIZE", "10000"))

        ic(self.__dict__)  # 🍦 Debug all config


# 🏭 Abstract base class for vector stores
class VectorStore:
    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        raise NotImplementedError()

    def insert_data(self, data_file: str) -> int:
        # ic(data_file)
        raise NotImplementedError()


# 🎯 Milvus implementation
class MilvusStore(VectorStore):
    def __init__(self, config: Config):
        super().__init__(config)
        self.collection: Optional[Collection] = None
        self.id_map = {}  # 🔄 Pour mapper les hash IDs aux int IDs
        from config.milvus import DIMENSION

        self.dimension = DIMENSION

    def setup(self):
        logger.info("🔌 Setting up Milvus connection...")
        try:
            connections.connect(host=self.config.host, port=self.config.port)

            # Initialize database
            existing = db.list_database()
            if self.config.store_name not in existing:
                db.create_database(self.config.store_name)
            db.using_database(self.config.store_name)

            # Setup schema
            fields = [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("hash_id", DataType.VARCHAR, max_length=128),
                FieldSchema("title", DataType.VARCHAR, max_length=300),
                FieldSchema("content", DataType.VARCHAR, max_length=20000),
                FieldSchema("dense", DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
            ]
            schema = CollectionSchema(
                fields=fields, description="Hybrid search collection"
            )

            if utility.has_collection(self.config.store_name):
                utility.drop_collection(self.config.store_name)

            self.collection = Collection(name=self.config.store_name, schema=schema)

            # Create indexes
            self.collection.create_index(
                "sparse", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            )
            self.collection.create_index(
                "dense", {"index_type": "FLAT", "metric_type": "COSINE"}
            )

            logger.success("✅ Milvus setup complete!")

        except Exception as e:
            logger.error(f"❌ Milvus setup failed: {e}")
            raise

    def insert_data(self, data_file: str) -> int:
        if not self.collection:
            raise ValueError("❌ Collection not initialized!")

        batch: Dict[str, List[Any]] = {
            "pk": [],
            "hash_id": [],
            "title": [],
            "content": [],
            "dense": [],
            "sparse": [],
        }
        stats = {"inserted": 0, "skipped": 0, "errors": 0}

        def process_batch() -> None:
            if not batch["pk"]:
                return
            try:
                ic(f"Processing batch of {len(batch['pk'])} items")  # 🍦 Debug
                self.collection.insert(
                    [
                        batch["pk"],
                        batch["hash_id"],
                        batch["title"],
                        batch["content"],
                        batch["dense"],
                        batch["sparse"],
                    ]
                )
                stats["inserted"] += len(batch["pk"])
                logger.info(f"📊 Progress: {stats['inserted']} records")
                for k in batch:
                    batch[k] = []
            except Exception as e:
                logger.error(f"❌ Batch insert failed: {e}")
                stats["errors"] += 1

        counter = 1  # 🔢 Pour générer des IDs numériques uniques
        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    hash_id = record["id"]
                    vectors = record["vectors"]
                    texts = record["texts"]

                    sparse_data = vectors["sparse"]
                    if not sparse_data["data"] or not sparse_data["indices"]:
                        stats["skipped"] += 1
                        continue

                    sparse_vec = dict(zip(sparse_data["indices"], sparse_data["data"]))

                    batch["pk"].append(counter)  # ID numérique auto-incrémenté
                    batch["hash_id"].append(hash_id)  # 🆕 Stockage du hash ID original
                    batch["title"].append(texts["prompt"][:250])
                    batch["content"].append(texts["full_context"][:9900])
                    batch["dense"].append(vectors["dense"])
                    batch["sparse"].append(sparse_vec)

                    self.id_map[hash_id] = counter  # 🔄 Garder la trace du mapping
                    counter += 1

                    if len(batch["pk"]) >= self.config.batch_size:
                        process_batch()

                except json.JSONDecodeError as e:
                    stats["errors"] += 1
                    logger.error(f"❌ Invalid JSON at line {idx}: {e}")
                    continue
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"❌ Error processing line {idx}: {e}")
                    continue

        process_batch()
        self.collection.flush()
        return stats["inserted"]


# 🎨 ChromaDB implementation
class ChromaStore(VectorStore):
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = None
        self.collection = None
        self.port = os.getenv("CHROMADB_PORT", 8000)  # 🆕 Port par défaut 8000
        self.processed_ids = set()
        from config.chroma import DIMENSION

        self.dimension = DIMENSION

    def setup(self):
        logger.info("🔌 Setting up ChromaDB connection to Docker...")
        try:
            # Connexion au serveur ChromaDB via HTTP
            self.client = chromadb.HttpClient(host=self.config.host, port=self.port)

            # Tentative de suppression de la collection existante
            try:
                self.client.delete_collection(self.config.store_name)
                logger.info("🗑️ Collection existante supprimée")
            except Exception as e:
                logger.debug(f"⚠️ No collection to delete: {e}")

            # Création de la nouvelle collection
            self.collection = self.client.create_collection(
                name=self.config.store_name, metadata={"dimension": self.dimension}
            )

            logger.success("✅ ChromaDB setup complete!")

        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise

        logger.info("🔌 Setting up ChromaDB...")
        try:
            # Reset collection if exists
            try:
                self.client.delete_collection(self.config.store_name)
            except Exception as e:
                logger.debug(f"⚠️ No collection to delete: {e}")
                pass

            self.collection = self.client.create_collection(
                name=self.config.store_name, metadata={"dimension": self.dimension}
            )

            logger.success("✅ ChromaDB setup complete!")

        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise

    # Dans la classe ChromaStore, modifiez la méthode insert_data :

    def insert_data(self, data_file: str) -> int:
        if not self.collection:
            raise ValueError("❌ Collection not initialized!")

        stats = {"inserted": 0, "skipped": 0, "errors": 0}
        batch_ids = []
        batch_embeddings = []
        batch_metadata = []
        batch_documents = []

        def process_batch() -> None:
            if not batch_ids:
                return
            try:
                ic(f"Processing ChromaDB batch of {len(batch_ids)} items")

                # Conversion des métadonnées en format compatible
                formatted_metadata = []
                for meta in batch_metadata:
                    # Convertir le vecteur sparse en format sérialisable
                    sparse_vec = meta["sparse_vector"]
                    formatted_meta = {
                        "title": meta["title"],
                        "sparse_indices": ",".join(map(str, sparse_vec["indices"])),
                        "sparse_data": ",".join(map(str, sparse_vec["data"])),
                        "sparse_shape": str(sparse_vec["shape"][1]),
                    }
                    formatted_metadata.append(formatted_meta)

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=formatted_metadata,
                    documents=batch_documents,
                )
                stats["inserted"] += len(batch_ids)
                logger.info(f"📊 Progress: {stats['inserted']} records")
                batch_ids.clear()
                batch_embeddings.clear()
                batch_metadata.clear()
                batch_documents.clear()
            except Exception as e:
                logger.error(f"❌ ChromaDB batch insert failed: {e}")
                stats["errors"] += 1

        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    record_id = record["id"]

                    # 🔍 Skip si l'ID a déjà été traité
                    if record_id in self.processed_ids:
                        stats["skipped"] += 1
                        logger.debug(f"⚠️ Skipping duplicate ID: {record_id}")
                        continue

                    self.processed_ids.add(record_id)  # ✅ Marquer l'ID comme traité

                    batch_ids.append(record_id)
                    batch_embeddings.append(record["vectors"]["dense"])
                    batch_metadata.append(
                        {
                            "title": record["texts"]["prompt"],
                            "sparse_vector": record["vectors"]["sparse"],
                            "metadata": record["metadata"],
                        }
                    )
                    batch_documents.append(record["texts"]["full_context"][:9900])

                    if len(batch_ids) >= self.config.batch_size:
                        process_batch()

                except json.JSONDecodeError as e:
                    stats["errors"] += 1
                    logger.error(f"❌ Invalid JSON at line {idx}: {e}")
                    continue
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"❌ Error processing line {idx}: {str(e)}")
                    continue

        if batch_ids:  # Traiter le dernier batch
            process_batch()

        # 📊 Log final des statistiques
        logger.info(f"📈 Final stats: {stats}")
        return stats["inserted"]

    def __init__(self, config: Config):
        super().__init__(config)
        self.client = None
        self.batch_size = 10  # 🔧 Taille de batch réduite pour plus de stabilité
        self.stats = {
            "total_inserted": 0,
            "skipped_entries": 0,
            "corrupted_lines": 0,
            "failed_batches": 0,
        }
        from config.qdrant import DIMENSION

        self.dimension = DIMENSION

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _insert_batch(self, points):
        """Insert a batch of points with retry logic"""
        if not points:
            return
        self.client.upsert(collection_name=self.config.store_name, points=points)

    def setup(self):
        """Initialize Qdrant connection and create collection"""
        logger.info("🔌 Setting up Qdrant connection...")
        try:
            # 🔗 Create client with increased timeout
            self.client = QdrantClient(
                host=self.config.host, port=os.getenv("QDRANT_PORT"), timeout=30
            )

            # 📋 Check and cleanup existing collection
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.config.store_name in collection_names:
                logger.info(f"🗑️ Deleting existing collection {self.config.store_name}")
                self.client.delete_collection(collection_name=self.config.store_name)

            # 📦 Create new collection
            self.client.create_collection(
                collection_name=self.config.store_name,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
            )
            logger.success(f"✅ Created new collection: {self.config.store_name}")

        except Exception as e:
            logger.error(f"❌ Qdrant setup failed: {e}")
            raise

    def insert_data(self, data_file: str) -> int:
        """Load and insert data from JSONL file into Qdrant"""
        if not self.client:
            raise ValueError("❌ Client not initialized!")

        points = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    record = json.loads(line)

                    # 🔍 Validate sparse vector data
                    sparse_data = record["vectors"]["sparse"]
                    if not sparse_data["data"] or not sparse_data["indices"]:
                        logger.warning(
                            f"⚠️ Skipping line {line_number}: empty sparse vector"
                        )
                        self.stats["skipped_entries"] += 1
                        continue

                    # 🔄 Convert sparse vector to dictionary format
                    sparse_vector = {
                        str(x[0]): float(x[1])
                        for x in zip(sparse_data["indices"], sparse_data["data"])
                    }

                    # 📝 Create point with proper structure
                    point = models.PointStruct(
                        id=record["id"],
                        vector=record["vectors"]["dense"],
                        payload={
                            "title": record["texts"]["prompt"][:250],
                            "content": record["texts"]["full_context"][:19900],
                            "sparse_vector": sparse_vector,
                            "metadata": record.get("metadata", {}),
                        },
                    )
                    points.append(point)

                except json.JSONDecodeError as e:
                    self.stats["corrupted_lines"] += 1
                    logger.error(f"❌ Corrupted JSON at line {line_number}: {str(e)}")
                    continue

                # 📦 Process batch when size limit reached
                if len(points) >= self.batch_size:
                    try:
                        self._insert_batch(points)
                        self.stats["total_inserted"] += len(points)
                        logger.info(
                            f"📊 Progress: {self.stats['total_inserted']} entities inserted"
                        )
                    except Exception as e:
                        logger.error(f"❌ Batch insertion failed: {e}")
                        self.stats["failed_batches"] += 1
                    points = []

        # 📤 Process remaining points
        if points:
            try:
                self._insert_batch(points)
                self.stats["total_inserted"] += len(points)
            except Exception as e:
                logger.error(f"❌ Final batch insertion failed: {e}")
                self.stats["failed_batches"] += 1

        # 📊 Log final statistics
        logger.info("📈 Insertion complete!")
        logger.info(f"Total inserted: {self.stats['total_inserted']}")
        logger.info(f"Skipped: {self.stats['skipped_entries']}")
        logger.info(f"Corrupted: {self.stats['corrupted_lines']}")
        logger.info(f"Failed batches: {self.stats['failed_batches']}")

        # 🔍 Final verification
        collection_info = self.client.get_collection(self.config.store_name)
        logger.info(f"Final collection count: {collection_info.points_count}")

        return self.stats["total_inserted"]


class QdrantStore(VectorStore):
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = None
        self.batch_size = 10
        from config.qdrant import DIMENSION

        self.dimension = DIMENSION
        self.stats = {
            "total_inserted": 0,
            "skipped_entries": 0,
            "corrupted_lines": 0,
            "failed_batches": 0,
        }
        self.id_counter = 1

    def setup(self):
        """Initialize Qdrant connection and create collection with both vector types"""
        logger.info("🔌 Setting up Qdrant connection...")
        try:
            self.client = QdrantClient(
                host=self.config.host,
                port=os.getenv("QDRANT_PORT", 6333),
                # prefer_grpc=True,
                timeout=60,
            )

            # Nettoyage de la collection existante
            collections = self.client.get_collections().collections
            if self.config.store_name in [c.name for c in collections]:
                self.client.delete_collection(collection_name=self.config.store_name)

            # Configuration avec un nom de vecteur explicite
            vectors_config = {
                "default": models.VectorParams(  # Nom explicite du vecteur
                    size=self.dimension, distance=models.Distance.COSINE
                )
            }

            self.client.create_collection(
                collection_name=self.config.store_name,
                vectors_config=vectors_config,  # Configuration avec nom de vecteur
            )
            logger.success(f"✅ Created collection: {self.config.store_name}")

        except Exception as e:
            logger.error(f"❌ Qdrant setup failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _insert_batch(self, points):
        """Insert a batch of points with retry logic"""
        if not points:
            return
        sample_point = points[0]
        logger.debug("Sample point structure:")
        logger.debug(f"- ID: {sample_point.id}")
        logger.debug(f"- Vector type: {type(sample_point.vector)}")
        logger.debug(f"- Vector length: {len(sample_point.vector)}")
        logger.debug(f"- Payload keys: {sample_point.payload.keys()}")

        self.client.upsert(
            collection_name=self.config.store_name, points=points, wait=True
        )

    def insert_data(self, data_file: str) -> int:
        """Load and insert data with both dense and sparse vectors"""
        if not self.client:
            raise ValueError("❌ Client not initialized!")

        points = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    record = json.loads(line)

                    sparse_data = record["vectors"]["sparse"]

                    # Validation
                    if not sparse_data["data"] or not sparse_data["indices"]:
                        logger.warning(
                            f"⚠️ Skipping line {line_number}: empty sparse vector"
                        )
                        self.stats["skipped_entries"] += 1
                        continue
                    sparse_vector = {
                        str(x[0]): float(x[1])
                        for x in zip(sparse_data["indices"], sparse_data["data"])
                    }

                    # Création du point avec les deux types de vecteurs
                    point = models.PointStruct(
                        id=self.id_counter,
                        vector={
                            "default": record["vectors"][
                                "dense"
                            ]  # Nom correspondant à la config
                        },
                        payload={
                            "title": record["texts"]["prompt"][:250],
                            "content": record["texts"]["full_context"][:19900],
                            "metadata": record.get("metadata", {}),
                            # Stockage du vecteur sparse dans le payload
                            "sparse_vector": sparse_vector,
                            "original_hash_id": record[
                                "id"
                            ],  # Sauvegarde du hash original
                        },
                    )
                    points.append(point)
                    self.id_counter += 1

                    if len(points) >= self.batch_size:
                        try:
                            self._insert_batch(points)
                            self.stats["total_inserted"] += len(points)
                            logger.info(
                                f"📊 Progress: {self.stats['total_inserted']} entities inserted"
                            )
                        except Exception as e:
                            logger.error(f"❌ Batch insertion failed: {e}")
                            logger.error(f"Error details: {str(e)}")
                            self.stats["failed_batches"] += 1
                        points = []

                except json.JSONDecodeError as e:
                    self.stats["corrupted_lines"] += 1
                    logger.error(f"❌ Corrupted JSON at line {line_number}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error processing line {line_number}: {str(e)}")
                    self.stats["corrupted_lines"] += 1
                    continue

        # Traitement du dernier lot
        if points:
            try:
                self._insert_batch(points)
                self.stats["total_inserted"] += len(points)
            except Exception as e:
                logger.error(f"❌ Final batch insertion failed: {e}")
                self.stats["failed_batches"] += 1

        logger.info("📈 Insertion complete!")
        for key, value in self.stats.items():
            logger.info(f"{key}: {value}")

        # Vérification finale
        try:
            collection_info = self.client.get_collection(self.config.store_name)
            logger.info(f"Final collection size: {collection_info.points_count}")
        except Exception as e:
            logger.error(f"❌ Failed to get final collection info: {e}")

        return self.stats["total_inserted"]


# 🏭 Factory for creating vector stores
def create_vector_store(backend: str = "milvus") -> VectorStore:
    config = Config(backend)
    if backend == "milvus":
        return MilvusStore(config)
    elif backend == "chroma":
        return ChromaStore(config)
    elif backend == "qdrant":
        return QdrantStore(config)
    else:
        raise ValueError(f"❌ Unsupported backend: {backend}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["setup"], default="setup")
    parser.add_argument(
        "--backend", choices=["milvus", "chroma", "qdrant"], default="qdrant"
    )
    parser.add_argument("--data-file", required=True, help="Path to data file")
    args = parser.parse_args()

    if args.mode == "setup":
        # Create and setup store
        store = create_vector_store(args.backend)
        store.setup()
        inserted = store.insert_data(args.data_file)
        logger.success(f"✨ Successfully inserted {inserted} records!")
