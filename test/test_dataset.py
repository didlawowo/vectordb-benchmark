from loguru import logger
from datasets import load_dataset
import pydantic
import tempfile
from dataclasses import dataclass
from typing import  List, Dict
from icecream import ic

# Configuration pour le test
@dataclass
class TestConfig:
    chunk_size: int = 2  # Petit chunk pour le test
    model_name: str = 'BAAI/bge-m3'
    batch_size: int = 1
    max_documents: int = 2
    
    def __post_init__(self):
        self.output_file = tempfile.mktemp(suffix='.jsonl')  # Fichier temporaire pour le test

class Message(pydantic.BaseModel):
    role: str
    content: str

class MessageData(pydantic.BaseModel):
    prompt: str
    prompt_id: str
    messages: List[Dict[str, str]]

    def get_full_text(self) -> str:
        """Combine le prompt et les messages utilisateur"""
        user_messages = [
            msg['content'] 
            for msg in self.messages 
            if msg['role'] == 'user'
        ]
        return f"{self.prompt} " + " ".join(user_messages)

def test_message_data():
    """Test de la classe MessageData"""
    # Test avec l'exemple fourni
    test_data = {
        'prompt': "These instructions apply to section-based themes...",
        'prompt_id': "f0e37e9f7800261167ce91143f98f511f768847236f133f2d0aed60b444ebe57",
        'messages': [
            {
                'content': "These instructions apply to section-based themes...",
                'role': "user"
            },
            {
                'content': "This feature only applies to Collection pages...",
                'role': "assistant"
            }
        ]
    }
    
    try:
        # Test de création de l'objet
        message_data = MessageData(**test_data)
        ic("✅ MessageData creation: OK")
        
        # Test de get_full_text
        full_text = message_data.get_full_text()
        ic("✅ get_full_text: OK")
        ic(f"Full text length: {len(full_text)}")
        ic(f"First 100 chars: {full_text[:100]}")
        
    except Exception as e:
        ic(f"❌ Test failed: {e}")
        raise

def test_dataset_loading():
    """Test du chargement du dataset"""
    try:
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft"
        )
        sample = dataset.select(range(2))
        ic("✅ Dataset loading: OK")
        
        # Affichage de la structure
        ic("\nDataset structure:")
        first_item = sample[0]
        for key in first_item:
            ic(f"- {key}: {type(first_item[key])}")
            
        return sample
        
    except Exception as e:
        ic(f"❌ Dataset loading failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🧪 Starting tests...")
    
    # Test 1: MessageData
    ic("\n1️⃣ Testing MessageData")
    test_message_data()
    
    # Test 2: Dataset Loading
    ic("\n2️⃣ Testing Dataset Loading")
    test_dataset_loading()