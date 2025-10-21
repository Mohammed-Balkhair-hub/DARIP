"""
Step 3: RAG Query Retrieval Agent

Replaces clustering with query-based retrieval using fixed queries.
Outputs queried_news.json with top-ranked articles.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

from config import settings
from rag import chunk_articles, Embedder, FAISSIndexer, HybridScorer, mmr_select, enforce_per_query_cap

logger = logging.getLogger(__name__)


def retrieve_with_rag(date: str = None) -> Dict[str, Any]:
    """
    Main entry point for RAG query retrieval.
    
    Args:
        date: Date string (YYYY-MM-DD), defaults to today
        
    Returns:
        Dict with retrieval stats
    """
    if date is None:
        date = settings.TODAY
    
    logger.info(f"[rag_retriever] ===== STEP 3: RAG Query Retrieval =====")
    logger.info(f"[rag_retriever] Date: {date}")
    
    start_time = time.time()
    
    # 1. Load enriched raw_items.json
    raw_items_path = os.path.join(settings.OUTPUT_DIR, date, 'raw_items.json')
    logger.info(f"[rag_retriever] Loading {raw_items_path}")
    
    with open(raw_items_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    articles = raw_data.get('articles', [])
    logger.info(f"[rag_retriever] Loaded {len(articles)} articles")
    
    # 2. Load queries configuration
    logger.info(f"[rag_retriever] Loading queries from {settings.RAG_QUERIES_FILE}")
    
    with open(settings.RAG_QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries_config = json.load(f)
    
    queries = queries_config['queries']
    logger.info(f"[rag_retriever] Loaded {len(queries)} queries")
    
    # 3. Chunk articles
    logger.info(f"[rag_retriever] Chunking articles (size={settings.RAG_CHUNK_SIZE}, overlap={settings.RAG_CHUNK_OVERLAP})")
    chunks = chunk_articles(articles, settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP)
    logger.info(f"[rag_retriever] Created {len(chunks)} chunks")
    
    if len(chunks) == 0:
        logger.warning("[rag_retriever] No chunks created, cannot proceed")
        return {'error': 'No chunks created'}
    
    # 4. Embed chunks
    logger.info(f"[rag_retriever] Embedding chunks with {settings.RAG_EMBEDDING_MODEL}")
    embedder = Embedder(settings.RAG_EMBEDDING_MODEL)
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = embedder.embed_texts(chunk_texts, normalize=True, show_progress=True)
    
    # 5. Build FAISS index
    logger.info("[rag_retriever] Building FAISS index")
    indexer = FAISSIndexer(embedder.get_dimension())
    indexer.add(chunk_embeddings)
    
    # 6. Build BM25 scorer
    logger.info("[rag_retriever] Building hybrid scorer")
    scorer = HybridScorer(chunks, alpha=settings.RAG_ALPHA_LEXICAL)
    
    # 7. Retrieve for each query
    logger.info(f"[rag_retriever] Retrieving candidates for {len(queries)} queries")
    all_article_candidates = {}  # item_id -> article with scores
    
    for query_obj in queries:
        query_text = query_obj['query']
        expansions = query_obj.get('expansions', [])
        
        # Build expanded query
        expanded_query = f"{query_text}; " + "; ".join(expansions)
        logger.info(f"[rag_retriever] Query: '{query_text}' (expanded)")
        
        # Embed query
        query_embedding = embedder.embed_query(expanded_query, normalize=True)
        
        # Dense retrieval (FAISS)
        faiss_distances, faiss_indices = indexer.search(query_embedding, settings.RAG_K_ANN)
        dense_results = scorer.score_dense(faiss_distances, faiss_indices)
        
        # Lexical retrieval (BM25)
        bm25_results = scorer.score_bm25(expanded_query, settings.RAG_K_BM25)
        
        # Hybrid scoring
        hybrid_results = scorer.hybrid_score(dense_results, bm25_results)
        
        # Keep top k_merge
        hybrid_results = hybrid_results[:settings.RAG_K_MERGE]
        
        # Group by article (item_id)
        article_scores = _aggregate_chunks_to_articles(hybrid_results, chunks, scorer, articles)
        
        # Add to global candidates
        for item_id, article_data in article_scores.items():
            if item_id not in all_article_candidates:
                all_article_candidates[item_id] = article_data
                all_article_candidates[item_id]['matched_queries'] = []
            
            # Update score if higher
            if article_data['score']['hybrid'] > all_article_candidates[item_id]['score']['hybrid']:
                all_article_candidates[item_id]['score'] = article_data['score']
                all_article_candidates[item_id]['best_chunk'] = article_data['best_chunk']
            
            # Add matched query
            all_article_candidates[item_id]['matched_queries'].append(query_text)
    
    logger.info(f"[rag_retriever] Found {len(all_article_candidates)} unique article candidates")
    
    # 8. Convert to list and sort by score
    candidate_articles = list(all_article_candidates.values())
    candidate_articles.sort(key=lambda x: x['score']['hybrid'], reverse=True)
    
    # 9. Apply per-query cap
    candidate_articles = enforce_per_query_cap(candidate_articles, settings.RAG_PER_QUERY_CAP)
    
    # 10. Apply MMR diversity selection
    logger.info(f"[rag_retriever] Applying MMR diversity selection (λ={settings.RAG_MMR_LAMBDA})")
    
    # Get embeddings for articles (use best chunk embedding)
    article_embeddings = []
    for article in candidate_articles:
        chunk_idx = article['best_chunk']['chunk_index']
        # Find the actual chunk in the original chunks list
        matching_chunk = next((c for c in chunks if c['item_id'] == article['item_id'] and c['chunk_index'] == chunk_idx), None)
        if matching_chunk:
            chunk_idx_in_list = chunks.index(matching_chunk)
            article_embeddings.append(chunk_embeddings[chunk_idx_in_list])
        else:
            # Fallback: use zero embedding
            article_embeddings.append(np.zeros(embedder.get_dimension()))
    
    import numpy as np
    article_embeddings = np.array(article_embeddings)
    
    final_articles = mmr_select(
        candidate_articles,
        article_embeddings,
        k=settings.RAG_FINAL_LIMIT,
        lambda_param=settings.RAG_MMR_LAMBDA
    )
    
    # 11. Add rank and prepare output
    for rank, article in enumerate(final_articles, start=1):
        article['rank'] = rank
    
    # 12. Build output JSON
    output = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'input': {
            'date_window': date,
            'total_items': len(articles),
            'collected_at': raw_data.get('collected_at', '')
        },
        'retrieval_config': {
            'embedding_model': settings.RAG_EMBEDDING_MODEL,
            'k_ann': settings.RAG_K_ANN,
            'k_bm25': settings.RAG_K_BM25,
            'k_merge': settings.RAG_K_MERGE,
            'alpha_lexical': settings.RAG_ALPHA_LEXICAL,
            'mmr_lambda': settings.RAG_MMR_LAMBDA,
            'final_limit': settings.RAG_FINAL_LIMIT,
            'per_query_cap': settings.RAG_PER_QUERY_CAP,
            'queries': queries
        },
        'items': final_articles,
        'stats': {
            'chunks_indexed': len(chunks),
            'candidates_union_articles': len(all_article_candidates),
            'after_per_query_cap': len(candidate_articles),
            'after_mmr': len(final_articles)
        }
    }
    
    # 13. Write output
    output_path = os.path.join(settings.OUTPUT_DIR, date, 'queried_news.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    
    logger.info(f"[rag_retriever] Wrote {len(final_articles)} articles to {output_path}")
    logger.info(f"[rag_retriever] ===== RETRIEVAL COMPLETE ({elapsed:.1f}s) =====")
    
    return {
        'date': date,
        'articles_retrieved': len(final_articles),
        'elapsed_sec': round(elapsed, 2)
    }


def _aggregate_chunks_to_articles(
    hybrid_results: List[tuple],
    chunks: List[Dict],
    scorer: HybridScorer,
    articles: List[Dict]
) -> Dict[str, Dict]:
    """
    Aggregate chunk-level scores to article-level.
    
    Args:
        hybrid_results: List of (chunk_idx, hybrid_score, dense_score, bm25_score)
        chunks: List of all chunks
        scorer: HybridScorer instance
        articles: Original articles list (for full_text lookup)
        
    Returns:
        Dict mapping item_id to article data with scores
    """
    # Create lookup dict for original articles by item_id
    articles_by_id = {article.get('item_id'): article for article in articles}
    
    article_data = defaultdict(lambda: {
        'max_score': 0,
        'best_chunk_idx': None,
        'best_scores': {}
    })
    
    for chunk_idx, hybrid_score, dense_score, bm25_score in hybrid_results:
        chunk = scorer.get_chunk(chunk_idx)
        item_id = chunk['item_id']
        
        # Track best chunk for this article
        if hybrid_score > article_data[item_id]['max_score']:
            article_data[item_id]['max_score'] = hybrid_score
            article_data[item_id]['best_chunk_idx'] = chunk_idx
            article_data[item_id]['best_scores'] = {
                'hybrid': hybrid_score,
                'dense': dense_score,
                'bm25': bm25_score
            }
    
    # Build final article objects
    result = {}
    
    for item_id, data in article_data.items():
        best_chunk_idx = data['best_chunk_idx']
        
        # Skip if no chunks found (shouldn't happen but defensive)
        if best_chunk_idx is None:
            continue
        
        best_chunk = scorer.get_chunk(best_chunk_idx)
        best_scores = data['best_scores']
        
        # Extract snippet (first 40 words)
        snippet = ' '.join(best_chunk['text'].split()[:40]) + '...'
        
        # Get original article for full_text
        orig_article = articles_by_id.get(item_id, {})
        
        # Build article dict
        article = {
            'item_id': item_id,
            'title': best_chunk['title'],
            'source': best_chunk['source'],
            'url': best_chunk['url'],
            'published_at': best_chunk['published_at'],
            'topics': best_chunk['topics'],
            'score': {
                'dense': best_scores['dense'],
                'lexical': best_scores['bm25'],
                'hybrid': best_scores['hybrid']
            },
            'best_chunk': {
                'chunk_index': best_chunk['chunk_index'],
                'snippet': snippet,
                'char_range': [best_chunk.get('char_start', 0), best_chunk.get('char_end', 0)]
            },
            'has_fulltext': orig_article.get('has_fulltext', True),
            'full_text_words': orig_article.get('full_text_words', 0),
            'full_text': orig_article.get('full_text', '')
        }
        
        result[item_id] = article
    
    return result


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    result = retrieve_with_rag()
    print(json.dumps(result, indent=2))

