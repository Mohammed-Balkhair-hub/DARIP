"""
Text chunking with overlap for RAG retrieval.

Chunks articles into semantic pieces with overlap to avoid boundary loss.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from dateutil import parser as dateparser

logger = logging.getLogger(__name__)


def chunk_articles(articles: List[Dict[str, Any]], chunk_size: int = 400, overlap: float = 0.15) -> List[Dict[str, Any]]:
    """
    Chunk articles into smaller pieces with overlap.
    
    Args:
        articles: List of article dicts from raw_items.json
        chunk_size: Target chunk size in tokens (~4 chars per token)
        overlap: Fraction of overlap between chunks (0.15 = 15%)
        
    Returns:
        List of chunk dicts with metadata
    """
    # First, deduplicate and normalize
    articles = _deduplicate_articles(articles)
    
    chunks = []
    
    for article in articles:
        # Extract text for chunking
        text = _get_text_for_retrieval(article)
        
        if not text:
            logger.warning(f"[chunker] No text for article {article.get('item_id')}")
            continue
        
        # Chunk the text
        article_chunks = _chunk_text(text, chunk_size, overlap)
        
        # Add metadata to each chunk
        for chunk_index, chunk_text in enumerate(article_chunks):
            chunk = {
                'item_id': article.get('item_id', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'title': article.get('title', ''),
                'published_at': article.get('published_at', ''),
                'topics': article.get('topics', []),
                'chunk_index': chunk_index,
                'chunk_count': len(article_chunks),
                'text': chunk_text,
                'char_start': sum(len(c) for c in article_chunks[:chunk_index]),
                'char_end': sum(len(c) for c in article_chunks[:chunk_index + 1])
            }
            chunks.append(chunk)
    
    logger.info(f"[chunker] Created {len(chunks)} chunks from {len(articles)} articles")
    
    return chunks


def _deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate articles by URL and title.
    
    Args:
        articles: List of article dicts
        
    Returns:
        Deduplicated list
    """
    seen_urls = {}
    seen_titles = {}
    
    for article in articles:
        url = article.get('url', '').strip()
        title = article.get('title', '').strip().lower()
        
        # Deduplicate by URL - keep latest
        if url:
            if url in seen_urls:
                existing = seen_urls[url]
                if _is_newer(article, existing):
                    seen_urls[url] = article
            else:
                seen_urls[url] = article
        
        # Deduplicate by exact title - keep longest full_text
        if title:
            if title in seen_titles:
                existing = seen_titles[title]
                if _is_longer_text(article, existing):
                    seen_titles[title] = article
            else:
                seen_titles[title] = article
    
    # Merge deduplication results by item_id
    result = {}
    for article in seen_urls.values():
        result[article.get('item_id')] = article
    for article in seen_titles.values():
        result[article.get('item_id')] = article
    
    return list(result.values())


def _is_newer(article1: Dict, article2: Dict) -> bool:
    """Check if article1 is newer than article2."""
    try:
        date1 = dateparser.parse(article1.get('published_at', ''))
        date2 = dateparser.parse(article2.get('published_at', ''))
        return date1 > date2 if date1 and date2 else False
    except:
        return False


def _is_longer_text(article1: Dict, article2: Dict) -> bool:
    """Check if article1 has longer full text than article2."""
    words1 = article1.get('full_text_words', 0)
    words2 = article2.get('full_text_words', 0)
    return words1 > words2


def _get_text_for_retrieval(article: Dict[str, Any]) -> str:
    """
    Extract text from article for retrieval.
    
    Priority:
    1. full_text (if has_fulltext and full_text_words > 50)
    2. content
    3. title only
    
    Args:
        article: Article dict
        
    Returns:
        Text string for chunking
    """
    has_fulltext = article.get('has_fulltext', False)
    full_text_words = article.get('full_text_words', 0)
    full_text = article.get('full_text', '')
    
    if has_fulltext and full_text_words > 50 and full_text:
        return full_text
    
    content = article.get('content', '')
    if content:
        return content
    
    # Fallback to title only
    return article.get('title', '')


def _chunk_text(text: str, chunk_size: int, overlap: float) -> List[str]:
    """
    Chunk text into pieces with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Target size in tokens (~4 chars per token)
        overlap: Fraction of overlap
        
    Returns:
        List of text chunks
    """
    # Convert token size to character size (rough estimate: 4 chars/token)
    char_chunk_size = chunk_size * 4
    overlap_chars = int(char_chunk_size * overlap)
    
    # Handle short text - return single chunk
    if len(text) <= char_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + char_chunk_size
        
        # Get chunk
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end markers
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')
            last_break = max(last_period, last_newline)
            
            if last_break > char_chunk_size * 0.7:  # At least 70% of chunk
                chunk = chunk[:last_break + 1]
                end = start + last_break + 1
        
        chunks.append(chunk.strip())
        
        # Move start forward with overlap
        start = end - overlap_chars
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

