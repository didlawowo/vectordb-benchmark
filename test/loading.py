from datasets import load_dataset
from loguru import logger

logger.info("🔄 Testing dataset loading...")

try:
    # Charger seulement 2 items
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft"
    ).select(range(2))
    
    # Afficher la structure du premier item
    first_item = dataset[0]
    logger.info("📊 First item structure:")
    for key, value in first_item.items():
        logger.info(f"- {key}: {type(value)}")
        if isinstance(value, list):
            logger.info(f"  First message: {value[0]}")
            
    logger.success("✅ Test completed successfully!")
    
except Exception as e:
    logger.error(f"❌ Test failed: {e}")