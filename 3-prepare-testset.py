from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def preparer_jeu_test():
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    testset = []
    
    # On utilise le modèle de Sentence Transformers pour les vecteurs denses
    modele_dense = SentenceTransformer('all-MiniLM-L6-v2')
    
 
    
    for conversation in dataset['train'].select(range(1000)):
        # Pour chaque tour de conversation
        for i in range(0, len(conversation['messages']), 2):
            message = conversation['messages'][i]
            
            # Création du document à indexer
            document = {
                "title": message[:100],  # On prend les 100 premiers caractères comme titre
                "content": message,
                "metadata": {
                    "messages_count": len(conversation['messages']),
                    "message_index": i
                }
            }
            
            # Création des vecteurs
            document["dense_vector"] = modele_dense.encode(message).tolist()
            document["sparse_vector"] = vectoriseur.transform(message)
            
            # Génération d'un hash unique
            document["original_hash_id"] = generer_hash(message)
            
            testset.append(document)
    
    return testset