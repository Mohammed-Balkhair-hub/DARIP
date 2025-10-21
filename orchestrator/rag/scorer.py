"""
Hybrid scoring combining dense (semantic) and lexical (BM25) scores.

Provides unified scoring interface for retrieval.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridScorer:
    """
    Hybrid scorer combining dense (FAISS) and BM25 (lexical) scores.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], alpha: float = 0.35):
        """
        Initialize hybrid scorer.
        
        Args:
            chunks: List of chunk dicts with 'text' field
            alpha: Weight for lexical score (1-alpha for dense score)
        """
        self.chunks = chunks
        self.alpha = alpha
        
        # Build BM25 index
        logger.info(f"[scorer] Building BM25 index for {len(chunks)} chunks")
        corpus = [chunk['text'] for chunk in chunks]
        tokenized_corpus = [text.lower().split() for text in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"[scorer] BM25 index built")
    
    def score_dense(self, faiss_distances: np.ndarray, faiss_indices: np.ndarray) -> List[Tuple[int, float]]:
        """
        Convert FAISS results to (chunk_index, dense_score) pairs.
        
        Args:
            faiss_distances: Array of shape (1, k) with similarity scores
            faiss_indices: Array of shape (1, k) with chunk indices
            
        Returns:
            List of (chunk_index, score) tuples
        """
        results = []
        
        for i in range(len(faiss_indices[0])):
            chunk_idx = int(faiss_indices[0][i])
            score = float(faiss_distances[0][i])
            
            # IndexFlatIP returns inner product (cosine for normalized vectors)
            # Already in range [0, 1] for normalized vectors
            results.append((chunk_idx, score))
        
        return results
    
    def score_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Score chunks using BM25.
        
        Args:
            query: Query string
            k: Number of top results
            
        Returns:
            List of (chunk_index, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Normalize scores to [0, 1]
        max_score = scores.max() if len(scores) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0
        
        results = [(int(idx), float(scores[idx] / max_score)) for idx in top_k_indices]
        
        return results
    
    def hybrid_score(
        self,
        dense_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float, float, float]]:
        """
        Combine dense and BM25 scores.
        
        Args:
            dense_results: List of (chunk_index, dense_score)
            bm25_results: List of (chunk_index, bm25_score)
            
        Returns:
            List of (chunk_index, hybrid_score, dense_score, bm25_score)
        """
        # Create score dicts
        dense_scores = {idx: score for idx, score in dense_results}
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Get union of chunk indices
        all_indices = set(dense_scores.keys()) | set(bm25_scores.keys())
        
        # Compute hybrid scores
        results = []
        for chunk_idx in all_indices:
            dense_score = dense_scores.get(chunk_idx, 0.0)
            bm25_score = bm25_scores.get(chunk_idx, 0.0)
            
            # Hybrid: alpha * lexical + (1-alpha) * dense
            hybrid = self.alpha * bm25_score + (1 - self.alpha) * dense_score
            
            results.append((chunk_idx, hybrid, dense_score, bm25_score))
        
        # Sort by hybrid score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_chunk(self, chunk_idx: int) -> Dict[str, Any]:
        """Get chunk by index."""
        return self.chunks[chunk_idx]

