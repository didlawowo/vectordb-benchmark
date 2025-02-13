from typing import List, Dict
import hashlib
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from loguru import logger
from icecream import ic


class TestQueriesGenerator:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize with BGE model"""
        self.model = SentenceTransformer(model_name)
        self.instruction = "Represent this sentence for searching relevant passages:"
        logger.info(f"🔧 Initialized with model: {model_name}")

    def generate_hash(self, text: str) -> str:
        """Generate a unique hash for a text"""
        return hashlib.sha256(text.encode()).hexdigest()

    def encode_text(self, text: str) -> List[float]:
        """Encode text using BGE model"""
        return self.model.encode(
            self.instruction + text, normalize_embeddings=True
        ).tolist()

    def prepare_test_queries(
        self,
        num_samples: int = 1000,
        min_length: int = 50,
        max_length: int = 500,
        use_test_split: bool = True,
    ) -> List[Dict]:
        """Prepare test queries from UltraChat dataset"""
        try:
            logger.info("🔄 Loading UltraChat dataset...")
            dataset = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
            )

            # Display dataset info
            logger.info(f"Dataset splits: {dataset.keys()}")
            split = "test_sft" if use_test_split and "test" in dataset else "train_sft"
            logger.info(f"Using {split} split")

            testset = []

            # Sample random conversations
            total_convs = len(dataset[split])
            logger.info(f"Total conversations in {split}: {total_convs}")

            sampled_indices = np.random.choice(
                total_convs, min(num_samples, total_convs), replace=False
            )
            logger.info(f"Sampled {len(sampled_indices)} conversations")
            for idx in tqdm(sampled_indices, desc="Processing conversations"):
                conversation = dataset[split][int(idx)]
                messages = conversation["messages"]
                logger.info(
                    f"Processing conversation {idx} with {len(messages)} messages"
                )

                # Process messages
                for i in range(0, len(messages)):
                    message = messages[i]
                    # ic(message)

                    # Extract content from message dictionary
                    message_content = message.get("content", "")

                    # Skip empty messages
                    if not message_content:
                        continue

                    # Get the corresponding assistant response
                    assistant_response = (
                        messages[i + 1]["content"] if i + 1 < len(messages) else None
                    )
                    previous_message = messages[i - 2]["content"] if i >= 2 else None

                    try:
                        # Create test query document
                        document = {
                            "title": message_content[:100],
                            "content": message_content,
                            "vector": self.encode_text(message_content),
                            "metadata": {
                                "split": split,
                                "conversation_length": len(messages),
                                "message_position": i,
                                "message_length": len(message_content),
                                "role": message.get("role", ""),
                                "has_response": assistant_response is not None,
                                "response_length": len(assistant_response)
                                if assistant_response
                                else 0,
                            },
                            "context": {
                                "previous_message": previous_message,
                                "next_message": assistant_response,
                            },
                            "hash_id": self.generate_hash(message_content),
                        }
                        testset.append(document)

                    except Exception as e:
                        logger.error(f"❌ Error processing message: {str(e)}")
                        continue

            logger.info(f"✅ Generated {len(testset)} test queries")
            return testset

        except Exception as e:
            logger.error(f"❌ Failed to prepare test queries: {str(e)}")
            raise

    def save_queries(self, queries: List[Dict], output_file: str):
        """Save test queries to file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(queries, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved queries to {output_file}")

            # Print some statistics
            logger.info("\n📊 Query Statistics:")
            splits = set(q["metadata"]["split"] for q in queries)
            for split in splits:
                split_queries = [q for q in queries if q["metadata"]["split"] == split]
                logger.info(f"- {split} split: {len(split_queries)} queries")
                avg_length = sum(len(q["content"]) for q in split_queries) / len(
                    split_queries
                )
                logger.info(f"  Average query length: {avg_length:.2f} characters")

        except Exception as e:
            logger.error(f"❌ Failed to save queries: {str(e)}")
            raise


def main():
    # Initialize generator
    generator = TestQueriesGenerator()

    # Generate test queries from both splits
    test_queries = generator.prepare_test_queries(
        num_samples=500,
        min_length=2,
        max_length=500,
        use_test_split=True,  # Use test split if available
    )

    # Save queries
    generator.save_queries(test_queries, "data/test_queries.json")


if __name__ == "__main__":
    main()
