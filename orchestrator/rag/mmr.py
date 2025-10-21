"""
Maximal Marginal Relevance (MMR) for diversity selection.

Selects diverse articles based on relevance and dissimilarity to already selected ones.
"""

import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def mmr_select(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    k: int,
    lambda_param: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Select k diverse articles using MMR.
    
    MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    
    Args:
        articles: List of article dicts with 'score' field
        embeddings: Array of article embeddings, shape (len(articles), dim)
        k: Number of articles to select
        lambda_param: Trade-off between relevance and diversity (0=max diversity, 1=max relevance)
        
    Returns:
        List of selected articles
    """
    if len(articles) <= k:
        return articles
    
    if len(articles) != len(embeddings):
        raise ValueError(f"Number of articles ({len(articles)}) doesn't match embeddings ({len(embeddings)})")
    
    logger.info(f"[mmr] Selecting {k} diverse articles from {len(articles)} using MMR (λ={lambda_param})")
    
    selected_indices = []
    remaining_indices = list(range(len(articles)))
    
    # Select first article (highest relevance score)
    relevance_scores = np.array([article['score']['hybrid'] for article in articles])
    first_idx = np.argmax(relevance_scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select remaining articles
    while len(selected_indices) < k and remaining_indices:
        max_mmr_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]
            
            # Diversity component: max similarity to already selected
            selected_embeddings = embeddings[selected_indices]
            candidate_embedding = embeddings[idx].reshape(1, -1)
            
            similarities = cosine_similarity(candidate_embedding, selected_embeddings)[0]
            max_sim = similarities.max()
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    # Return selected articles in order of selection
    selected_articles = [articles[idx] for idx in selected_indices]
    
    logger.info(f"[mmr] Selected {len(selected_articles)} diverse articles")
    
    return selected_articles


def enforce_per_query_cap(
    articles: List[Dict[str, Any]],
    per_query_cap: int
) -> List[Dict[str, Any]]:
    """
    Ensure no single query dominates the results.
    
    Strategy: Count each article against its PRIMARY query only (highest scoring).
    This prevents over-filtering when articles match multiple queries.
    
    Args:
        articles: List of article dicts with 'matched_queries' field
        per_query_cap: Max articles per query
        
    Returns:
        Filtered list of articles
        
    Note: For maximum recall (no filtering), set per_query_cap to a very high value
          or disable this function in the pipeline. MMR diversity will still apply.
    """
    query_counts = {}
    filtered = []
    
    for article in articles:
        matched_queries = article.get('matched_queries', [])
        
        if not matched_queries:
            # No queries matched, skip article
            continue
        
        # Strategy: Only count against PRIMARY query (first in list = highest scoring)
        # This is less restrictive than counting against ALL matched queries
        primary_query = matched_queries[0]
        
        # Check if primary query has room
        if query_counts.get(primary_query, 0) < per_query_cap:
            filtered.append(article)
            query_counts[primary_query] = query_counts.get(primary_query, 0) + 1
    
    logger.info(f"[mmr] After per-query cap ({per_query_cap}): {len(filtered)} articles (from {len(articles)} candidates)")
    logger.debug(f"[mmr] Query distribution: {dict(query_counts)}")
    
    return filtered

