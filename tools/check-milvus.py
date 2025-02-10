from pymilvus import connections, utility
from loguru import logger

def check_milvus():
    try:
        # Vérifier la connexion
        connections.connect(host='localhost', port=19530)
        logger.info("✅ Connection à Milvus établie")
        
        # Vérifier la collection
        if utility.has_collection("vector_store"):
            collection_info = utility.get_collection_info("vector_store")
            logger.info(f"✅ Collection trouvée: {collection_info}")
            
            # Compter les entités
            from pymilvus import Collection
            collection = Collection("vector_store")
            count = collection.num_entities
            logger.info(f"📊 Nombre d'entités: {count}")
            
            # Vérifier les index
            index_info = collection.indexes
            logger.info(f"📑 Index: {index_info}")
            
            return True
        else:
            logger.warning("⚠️ Collection non trouvée")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur Milvus: {e}")
        return False

check_milvus()