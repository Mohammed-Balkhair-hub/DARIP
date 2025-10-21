"""
RAG (Retrieval-Augmented Generation) utilities for query-based article retrieval.

Modules:
- chunker: Text chunking with overlap
- embedder: Embedding generation
- indexer: FAISS index management
- scorer: Hybrid scoring (dense + BM25)
- mmr: MMR diversity selection
"""

from .chunker import chunk_articles
from .embedder import Embedder
from .indexer import FAISSIndexer
from .scorer import HybridScorer
from .mmr import mmr_select, enforce_per_query_cap

__all__ = ['chunk_articles', 'Embedder', 'FAISSIndexer', 'HybridScorer', 'mmr_select', 'enforce_per_query_cap']

