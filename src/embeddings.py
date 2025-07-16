from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Convert text to vector embeddings"""
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(text)
        return embeddings
    
    def similarity_search(self, query_embedding: np.ndarray, 
                         embeddings: np.ndarray, top_k: int = 5) -> List[int]:
        """Find most similar embeddings"""
        similarities = np.dot(embeddings, query_embedding.T)
        top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]
        return top_indices.flatten().tolist()