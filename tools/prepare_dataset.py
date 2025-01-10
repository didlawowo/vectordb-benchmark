import csv
from typing import Any
import json
import numpy as np
import pydantic
from loguru import logger
from FlagEmbedding import BGEM3FlagModel
from scipy.sparse import csr_array

from common import CSV_FILE, DATASET_EMBEDDING_FILE


def divide_chunks(chunking_list: list[Any], chunk_size: int):
    # looping till length l
    for i in range(0, len(chunking_list), chunk_size): 
        yield chunking_list[i:i + chunk_size]

def csr_to_dict(csr):
    return {
        'data': csr.data.tolist(),
        'indices': csr.indices.tolist(),
        'indptr': csr.indptr.tolist(),
        'shape': csr.shape
    }

# Read CSV file
class GameData(pydantic.BaseModel):
    name: str
    description: str


CHUNK_SIZE = 500
logger.level("INFO")
data: list[GameData] = []

# Initialize the BGE-M3 model using FlagEmbedding
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')

with open(CSV_FILE, 'r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        data.append(GameData(name=row['name'], description=row['description']))

dict_index = 0
for chunk in divide_chunks(data, CHUNK_SIZE):
    logger.info(f"Processing chunk of size {len(chunk)}")
    texts = [f"{game.name} {game.description}" for game in chunk]
    
    # Generate embeddings using BGE-M3
    # FlagModel.encode() returns both dense and sparse embeddings by default
    embeddings = model.encode(texts, return_dense=True, return_sparse=True)
    
    # Process each text in the chunk
    with open(DATASET_EMBEDDING_FILE, 'a', encoding='utf-8') as f:
        logger.info("Writing to file...")
        for i, game in enumerate(chunk):
            indices = []
            sparse_data = []
            for idx, weight in embeddings['lexical_weights'][i].items():
                indices.append(int(idx))
                sparse_data.append(float(weight))
            
            sparse_vector = csr_array(
                (sparse_data, (np.zeros_like(indices), indices)),
                shape=(1, model.model.config.vocab_size)
            )
            entry = {
                'id': dict_index,
                'name': game.name,
                'description': game.description,
                'dense_vector': embeddings['dense_vecs'][i].tolist(),
                'sparse_vector': csr_to_dict(sparse_vector),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            dict_index += 1