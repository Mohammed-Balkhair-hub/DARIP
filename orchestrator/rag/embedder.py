"""
Embedding generation using sentence-transformers.

Provides consistent embedding generation for chunks and queries.
"""

import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wrapper for sentence-transformers embedding model.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedder with model.
        
        Args:
            model_name: HuggingFace model name
        """
        logger.info(f"[embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"[embedder] Model loaded, dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            show_progress: Show progress bar
            
        Returns:
            Array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        logger.debug(f"[embedder] Embedding {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query: Query string
            normalize: Whether to L2-normalize
            
        Returns:
            Array of shape (dimension,)
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        
        return embedding[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

