from datasets import load_dataset
from loguru import logger
import json

logger.info("🔍 Debugging dataset structure")

try:
    # Charger un seul élément
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(1))
    
    # Examinons la structure
    first_item = dataset[0]
    logger.info("📊 First item type: " + str(type(first_item)))
    logger.info("📊 First item keys: " + str(first_item.keys()))
    logger.info("📊 First item content:")
    logger.info(json.dumps(first_item, indent=2))
    
except Exception as e:
    logger.error(f"❌ Error: {e}")
