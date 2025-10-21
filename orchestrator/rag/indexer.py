"""
FAISS index management for fast similarity search.

Uses IndexFlatIP for exact cosine similarity search (via normalized vectors).
"""

import logging
import numpy as np
from typing import List, Tuple
import faiss

logger = logging.getLogger(__name__)


class FAISSIndexer:
    """
    FAISS index wrapper for similarity search.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        # IndexFlatIP: inner product (cosine for normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        logger.info(f"[indexer] Created FAISS IndexFlatIP with dimension {dimension}")
    
    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to index.
        
        Args:
            embeddings: Array of shape (n_vectors, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        self.index.add(embeddings)
        logger.info(f"[indexer] Added {len(embeddings)} vectors, total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
            - distances: Array of shape (1, k) with similarity scores
            - indices: Array of shape (1, k) with vector indices
        """
        # Ensure shape is (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32
        query_embedding = query_embedding.astype('float32')
        
        # Limit k to index size
        k = min(k, self.index.ntotal)
        
        if k == 0:
            return np.array([[]]), np.array([[]])
        
        distances, indices = self.index.search(query_embedding, k)
        
        return distances, indices
    
    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal
    
    def save(self, filepath: str):
        """
        Save index to file.
        
        Args:
            filepath: Path to save index
        """
        faiss.write_index(self.index, filepath)
        logger.info(f"[indexer] Saved index to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FAISSIndexer':
        """
        Load index from file.
        
        Args:
            filepath: Path to index file
            
        Returns:
            FAISSIndexer instance
        """
        index = faiss.read_index(filepath)
        dimension = index.d
        
        indexer = cls(dimension)
        indexer.index = index
        
        logger.info(f"[indexer] Loaded index from {filepath}, {index.ntotal} vectors")
        
        return indexer

