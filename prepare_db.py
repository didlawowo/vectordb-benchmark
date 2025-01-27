import os
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from icecream import ic   
import chromadb   
from pymilvus import (
    connections, db, utility,
    Collection, CollectionSchema, FieldSchema, DataType
)
from config.milvus import DIMENSION

# 🔧 Configuration class
class Config:
    def __init__(self, backend: str = "milvus"):
        self.backend = backend
        # 🔌 Connection settings
        self.host = os.getenv("VECTOR_HOST", "localhost")
        self.port = 19530
        self.dim = DIMENSION
        
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
        ic(data_file)
        raise NotImplementedError()

# 🎯 Milvus implementation
class MilvusStore(VectorStore):
    def __init__(self, config: Config):
        super().__init__(config)
        self.collection: Optional[Collection] = None
        self.id_map = {}  # 🔄 Pour mapper les hash IDs aux int IDs
        
    def setup(self):
        logger.info("🔌 Setting up Milvus connection...")
        try:
            connections.connect(
                host=self.config.host,
                port=self.config.port
            )
            
            # Initialize database
            existing = db.list_database()
            if self.config.store_name not in existing:
                db.create_database(self.config.store_name)
            db.using_database(self.config.store_name)
            
            # Setup schema avec le hash ID en plus
            fields = [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("hash_id", DataType.VARCHAR, max_length=128),  # 🆕 Ajout du hash ID
                FieldSchema("title", DataType.VARCHAR, max_length=300),
                FieldSchema("content", DataType.VARCHAR, max_length=20000),
                FieldSchema("dense", DataType.FLOAT_VECTOR, dim=self.config.dim),
                FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR)
            ]
            schema = CollectionSchema(fields=fields, description="Hybrid search collection")
            
            if utility.has_collection(self.config.store_name):
                utility.drop_collection(self.config.store_name)
            
            self.collection = Collection(name=self.config.store_name, schema=schema)
            
            # Create indexes
            self.collection.create_index("sparse", {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP"
            })
            self.collection.create_index("dense", {
                "index_type": "FLAT",
                "metric_type": "COSINE"
            })
            
            logger.success("✅ Milvus setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Milvus setup failed: {e}")
            raise

    def insert_data(self, data_file: str) -> int:
        if not self.collection:
            raise ValueError("❌ Collection not initialized!")
            
        batch: Dict[str, List[Any]] = {
            "pk": [], "hash_id": [], "title": [], "content": [],
            "dense": [], "sparse": []
        }
        stats = {"inserted": 0, "skipped": 0, "errors": 0}
        
        def process_batch() -> None:
            if not batch["pk"]:
                return
            try:
                ic(f"Processing batch of {len(batch['pk'])} items")  # 🍦 Debug
                self.collection.insert([
                    batch["pk"], batch["hash_id"], batch["title"], 
                    batch["content"], batch["dense"], batch["sparse"]
                ])
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
        self.port = 8080
        self.processed_ids = set()
 
    def setup(self):
        logger.info("🔌 Setting up ChromaDB...")
        try:
            # 🔄 Choix du client en fonction de l'environnement
            if self.config.host != "localhost":
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port
                )
            else:
                # 💾 Utilisation du client persistant pour le local
                self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Reset collection if exists
            try:
                self.client.delete_collection(self.config.store_name)
            except Exception as e:
                logger.debug(f"⚠️ No collection to delete: {e}")
                pass
                
            self.collection = self.client.create_collection(
                name=self.config.store_name,
                metadata={"dimension": self.config.dim}
            )
            
            logger.success("✅ ChromaDB setup complete!")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise            

    
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
                ic(f"Processing ChromaDB batch of {len(batch_ids)} items")  # 🍦 Debug
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    documents=batch_documents
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
                    batch_metadata.append({
                        "title": record["texts"]["prompt"],
                        "sparse_vector": record["vectors"]["sparse"],
                        "metadata": record["metadata"]
                    })
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

class QdrantStore(VectorStore):
    def __init__(self, config: Config):
        super().__init__(config)
        self.client = None
        self.collection = None
        
    def setup(self):
        logger.info("🔌 Setting up ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=self.config.chroma_dir)
            
            # Reset collection if exists
            try:
                self.client.delete_collection(self.config.store_name)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete existing collection: {e}")
                pass
                
            self.collection = self.client.create_collection(
                name=self.config.store_name,
                metadata={"dimension": self.config.dim}
            )
            
            logger.success("✅ ChromaDB setup complete!")
            
        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            raise
            
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
                ic(f"Processing ChromaDB batch of {len(batch_ids)} items")  # 🍦 Debug
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    documents=batch_documents
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
                    
                    batch_ids.append(str(record["id"]))
                    batch_embeddings.append(record["dense_vector"])
                    batch_metadata.append({
                        "title": record["name"],
                        "sparse_vector": record["sparse_vector"]
                    })
                    batch_documents.append(record["description"][:9900])
                    
                    if len(batch_ids) >= self.config.batch_size:
                        process_batch()
                        
                except json.JSONDecodeError as e:
                    stats["errors"] += 1
                    logger.error(f"❌ Invalid JSON at line {idx}: {e}")
                    continue

        process_batch()
        return stats["inserted"]

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
    parser.add_argument("--backend", choices=["milvus", "chroma"], default="milvus")
    parser.add_argument("--data-file", required=True, help="Path to data file")
    args = parser.parse_args()

    if args.mode == "setup":
        # Create and setup store
        store = create_vector_store(args.backend)
        store.setup()
        inserted = store.insert_data(args.data_file)
        logger.success(f"✨ Successfully inserted {inserted} records!")