import json
import numpy as np
from loguru import logger
from FlagEmbedding import BGEM3FlagModel
import torch
from datasets import load_dataset

import pydantic
from scipy.sparse import csr_array
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
from config.common import MODEL_NAME


# 🔧 Configuration
@dataclass
class Config:
    chunk_size: int = 500
    model_name: str = MODEL_NAME
    batch_size: int = 32
    output_file: str = "data/processed_dataset.jsonl"
    max_documents: Optional[int] = None
    splits: List[str] = field(
        default_factory=lambda: ["test_sft", "test_gen"]
    )  # Splits à utiliser


# 📊 Modèles de données
class ConversationData(pydantic.BaseModel):
    prompt: str
    prompt_id: str
    messages: List[Dict[str, str]]

    def get_full_context(self) -> str:
        """Obtient tout le contexte de la conversation"""
        return " ".join([msg["content"] for msg in self.messages])

    def get_user_messages(self) -> str:
        """Obtient uniquement les messages utilisateur"""
        return " ".join(
            [msg["content"] for msg in self.messages if msg["role"] == "user"]
        )

    def get_message_counts(self) -> Dict[str, int]:
        """Calcule les statistiques des messages"""
        return {
            "messages_count": len(self.messages),
            "user_messages_count": len(
                [m for m in self.messages if m["role"] == "user"]
            ),
            "assistant_messages_count": len(
                [m for m in self.messages if m["role"] == "assistant"]
            ),
        }


# 🛠️ Fonctions utilitaires
def get_optimal_device() -> str:
    """Détecte le meilleur device disponible"""
    if torch.backends.mps.is_available():
        return "mps"  # 🍎 Apple Silicon
    elif torch.cuda.is_available():
        return "cuda"  # 🎮 NVIDIA GPU
    return "cpu"  # 💻 CPU par défaut


@contextmanager
def init_model(device: Optional[str] = None):
    """Initialise le modèle avec gestion du contexte"""
    if device is None:
        device = get_optimal_device()

    # logger.info(f"🚀 Using device: {device} for model {config.model_name}")
    model = BGEM3FlagModel( 'BAAI/bge-m3', use_fp16=device != "cpu", device=device, )
    try:
        yield model
    finally:
        if hasattr(model, "close"):
            model.close()
        if device == "cuda":
            torch.cuda.empty_cache()


def csr_to_dict(csr):
    """Convertit une matrice CSR en dictionnaire pour JSON"""
    return {
        "data": csr.data.tolist(),
        "indices": csr.indices.tolist(),
        "indptr": csr.indptr.tolist(),
        "shape": csr.shape,
    }


class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.device = get_optimal_device()
        Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)

    def process_embeddings(
        self, texts: List[str], entries: List[ConversationData], model
    ):
        """Traite un lot de textes et sauvegarde leurs embeddings"""
        try:
            embeddings = model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                batch_size=self.config.batch_size,
            )

            with open(self.config.output_file, "a", encoding="utf-8") as f:
                for i, entry in enumerate(entries):
                    try:
                        # Construction du vecteur sparse
                        indices = []
                        sparse_data = []
                        for idx, weight in embeddings["lexical_weights"][i].items():
                            indices.append(int(idx))
                            sparse_data.append(float(weight))

                        sparse_vector = csr_array(
                            (sparse_data, (np.zeros_like(indices), indices)),
                            shape=(1, model.model.config.vocab_size),
                        )

                        # Préparation des données pour vectordb
                        output = {
                            "id": entry.prompt_id,
                            "vectors": {
                                "dense": embeddings["dense_vecs"][i].tolist(),
                                "sparse": csr_to_dict(sparse_vector),
                            },
                            "texts": {
                                "prompt": entry.prompt,
                                "prompt_id": entry.prompt_id,
                                "full_context": entry.get_full_context(),
                                "user_messages": entry.get_user_messages(),
                            },
                            "metadata": entry.get_message_counts(),
                        }

                        f.write(json.dumps(output, ensure_ascii=False) + "\n")

                    except Exception as e:
                        logger.error(f"❌ Error processing text {i}: {e}")
                        # logger.debug(f"Entry data: {entry}")
                        continue

        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            raise

    def process_batch(self, batch):
        """Traite un lot de données"""
        texts = []
        entries = []

        # 🔄 Traitement des données comme un dictionnaire de listes
        try:
            for i in range(len(batch["prompt"])):  # Itération sur les indices
                conversation_data = ConversationData(
                    prompt=batch["prompt"][i],
                    prompt_id=batch["prompt_id"][i],
                    messages=batch["messages"][i],
                )

                texts.append(conversation_data.prompt)
                entries.append(conversation_data)

        except Exception as e:
            logger.error(f"❌ Error processing batch: {e}")
            logger.debug(f"Batch structure: {batch.keys()}")

        return texts, entries

    def run(self):
        """Point d'entrée principal du traitement"""
        logger.info("🚀 Starting data processing...")

        try:
            # Charger et concaténer les splits demandés
            datasets = []
            for split in self.config.splits:
                logger.info(f"📚 Loading split: {split}")
                dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
                datasets.append(dataset)

            # Concatenate datasets if multiple splits
            # if len(datasets) > 1:
            #     dataset_train = datasets[0].concatenate_datasets(datasets[1:])
            # else:
            dataset_train = datasets[0]

            if self.config.max_documents:
                dataset_train = dataset_train.select(range(self.config.max_documents))

            logger.info(f"📚 Dataset loaded. Total size: {len(dataset_train)}")

            with init_model(self.device) as model:
                for start_idx in range(0, len(dataset_train), self.config.chunk_size):
                    end_idx = min(
                        start_idx + self.config.chunk_size, len(dataset_train)
                    )
                    batch = dataset_train[start_idx:end_idx]

                    logger.info(
                        f"⚙️ Processing batch {start_idx // self.config.chunk_size + 1}"
                    )

                    # Traitement du batch
                    texts, entries = self.process_batch(batch)
                    if texts and entries:
                        self.process_embeddings(texts, entries, model)

            logger.success("✨ Processing complete!")

        except Exception as e:
            logger.error(f"❌ Error in processing: {e}")
            raise


if __name__ == "__main__":
    config = Config(max_documents=10000, splits=["test_sft"])
    processor = DatasetProcessor(config)
    processor.run()
