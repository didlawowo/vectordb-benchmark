 
import json
import numpy as np

from loguru import logger
from FlagEmbedding import BGEM3FlagModel
import torch
from datasets import load_dataset
from icecream import ic
import pydantic
from scipy.sparse import csr_array
from typing import Any, List
from common import DATASET_EMBEDDING_FILE

def divide_chunks(chunking_list: list[Any], chunk_size: int):
    # looping till length l
    for i in range(0, len(chunking_list), chunk_size): 
        yield chunking_list[i:i + chunk_size]


def csr_to_dict(csr):
    """
    📊 Convertit une matrice CSR en dictionnaire pour la sérialisation JSON
    """
    return {
        'data': csr.data.tolist(),
        'indices': csr.indices.tolist(),
        'indptr': csr.indptr.tolist(),
        'shape': csr.shape
    }

CHUNK_SIZE = 500

    
# 1. 🔄 Détection automatique du meilleur device disponible
def get_optimal_device():
    ic(torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        return "mps"  # 🍎 Apple Silicon
    elif torch.cuda.is_available():
        return "cuda"  # 🎮 NVIDIA GPU
    else:
        return "cpu"   # 💻 CPU par défaut

# 2. 🚀 Initialisation du modèle avec le bon device
def init_model(device=None):
    if device is None:
        device = get_optimal_device()
    
    logger.info(f"Using device: {device}")
    
    return BGEM3FlagModel(
        'BAAI/bge-m3', 
        use_fp16=device != "cpu",  # FP16 seulement si GPU/MPS
        device=device
    )


class MessageData(pydantic.BaseModel):
    prompt: str
    prompt_id: str
    messages: List[dict]  # Liste de messages avec au moins 'content'

def load_and_process_data(max_documents=None):
    """
    💾 Prépare les données UltraChat pour la base vectorielle
    """
    logger.info("Initializing data loading...")
    
    try:
        dataset_train = load_dataset(
            "HuggingFaceH4/ultrachat_200k", 
            split="train_sft"
        )
        
        logger.info(f"Dataset loaded. Size: {len(dataset_train)}")
        
        if max_documents:
            dataset_train = dataset_train.select(range(max_documents))
            
        model = init_model(get_optimal_device())
        
        # Traitement par batch
        for start_idx in range(0, len(dataset_train), CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, len(dataset_train))
            texts = []
            entries = []
            
            # On récupère le batch complet avec toutes les colonnes
            current_batch = dataset_train[start_idx:end_idx]
            
            # Debug pour voir la structure
            # ic("Batch structure:", current_batch)
            # ic("First item:", current_batch[0])
            
            # On itère sur les indices du batch
            for idx in range(len(current_batch)):
                try:
                    # On récupère l'item complet
                    item = {
                        'prompt': current_batch['prompt'][idx],
                        'prompt_id': current_batch['prompt_id'][idx],
                        'messages': current_batch['messages'][idx]
                    }
                    
                    # Extraction des messages utilisateur
                    user_messages = [
                        msg['content'] 
                        for msg in item['messages'] 
                        if msg['role'] == 'user'
                    ]
                    
                    # Construction du texte
                    full_text = f"{item['prompt']} " + " ".join(user_messages)
                    
                    texts.append(full_text)
                    entries.append({
                        'prompt_id': item['prompt_id'],
                        'text': full_text[:1000]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing item at index {idx}: {e}")
                    continue
            
            if texts:
                logger.info(f"Processing batch {start_idx//CHUNK_SIZE + 1}, texts: {len(texts)}")
                process_embeddings(texts, model, entries)
            
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise
def process_embeddings(texts: List[str], model, entries: List[dict]):
    """
    🔄 Traite un lot de textes et sauvegarde leurs embeddings
    """
    try:
        embeddings = model.encode(
            texts, 
            return_dense=True,
            return_sparse=True,
            batch_size=32
        )
        
        with open(DATASET_EMBEDDING_FILE, 'a', encoding='utf-8') as f:
            for i, entry in enumerate(entries):
                try:
                    # Construction du vecteur sparse
                    indices = []
                    sparse_data = []
                    for idx, weight in embeddings['lexical_weights'][i].items():
                        indices.append(int(idx))
                        sparse_data.append(float(weight))
                    
                    sparse_vector = csr_array(
                        (sparse_data, (np.zeros_like(indices), indices)),
                        shape=(1, model.model.config.vocab_size)
                    )
                    
                    # Combinaison avec les métadonnées
                    output = {
                        'prompt_id': entry['prompt_id'],
                        'text': entry['text'],
                        'dense_vector': embeddings['dense_vecs'][i].tolist(),
                        'sparse_vector': csr_to_dict(sparse_vector)
                    }
                    
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    logger.error(f"Error processing text {i}: {e}")
                    continue
                
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise
if __name__ == "__main__":


    # Test
    # examine_data_structure()
    logger.info("Starting data processing...")
    load_and_process_data(max_documents=80_000)  # Commencer avec 100K documents
    logger.info("Processing complete!")