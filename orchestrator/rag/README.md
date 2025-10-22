# RAG Query Retrieval System

Semantic search using fixed queries to retrieve relevant articles.

---

## Files

```
rag/
  ├── chunker.py    - Split articles into chunks with overlap
  ├── embedder.py   - Generate embeddings (384-dim, all-MiniLM-L6-v2)
  ├── indexer.py    - FAISS index for similarity search
  ├── scorer.py     - Hybrid scoring (dense + BM25)
  └── mmr.py        - MMR diversity selection + per-query cap
```

**Used by:** `agents/rag_retriever.py`  
**Config:** `config/queries.json`, `config/settings.py`

---

## Flow

```
enriched raw_items.json (1300-1500 articles)
  ↓
Chunker: Split into ~3500-4200 chunks (350-550 tokens, 15% overlap)
  ↓
Embedder: Generate 384-dim embeddings (normalized)
  ↓
Indexer: Build FAISS IndexFlatIP
  ↓
Scorer: For each query (19 total):
  • Dense retrieve (FAISS, top 300 chunks)
  • Lexical retrieve (BM25, top 200 chunks)
  • Hybrid score: 0.25×BM25 + 0.75×cosine
  • Aggregate chunks → articles
  ↓
Merge: Union across queries (~1200-1300 articles)
  ↓
MMR: Per-query cap (max 8 per primary query) → ~150 articles
  ↓
MMR: Diversity selection (λ=0.3) → 30 articles
  ↓
queried_news.json (30 ranked articles with full_text)
```

**Time:** ~30-60 seconds (scales with article count)

---

## Components

### chunker.py
- `chunk_articles(articles, chunk_size, overlap)` → List of chunks
- Deduplicates by URL and title
- Uses `full_text` if available, else `content`

### embedder.py
- `Embedder(model_name)` → embedder instance
- `.embed_texts(texts)` → embeddings matrix
- `.embed_query(query)` → query embedding

### indexer.py
- `FAISSIndexer(dimension)` → indexer instance
- `.add(embeddings)` → add to index
- `.search(query_emb, k)` → (distances, indices)

### scorer.py
- `HybridScorer(chunks, alpha)` → scorer instance
- `.score_dense(distances, indices)` → dense results
- `.score_bm25(query, k)` → BM25 results
- `.hybrid_score(dense, bm25)` → combined scores (0.25×BM25 + 0.75×cosine)

### mmr.py
- `mmr_select(articles, embeddings, k, λ)` → k diverse articles
- `enforce_per_query_cap(articles, cap)` → filtered by cap

---

## Configuration

### `config/queries.json`
```json
{
  "queries": [
    {
      "query": "nvidia gpu hardware",
      "expansions": ["blackwell", "RTX", "B200", "GB200", ...]
    }
  ]
}
```

### `config/settings.py`
```python
RAG_CHUNK_SIZE = 400           # tokens
RAG_CHUNK_OVERLAP = 0.15       # 15%
RAG_K_ANN = 300                # Dense candidates
RAG_K_BM25 = 200               # Lexical candidates
RAG_ALPHA_LEXICAL = 0.25       # BM25 weight
RAG_MMR_LAMBDA = 0.30          # Diversity weight
RAG_FINAL_LIMIT = 30           # Final count
RAG_PER_QUERY_CAP = 8          # Max per query
```

---

## Output

### `queried_news.json`
```json
{
  "items": [
    {
      "rank": 1,
      "item_id": "...",
      "title": "...",
      "url": "...",
      "matched_queries": ["nvidia gpu hardware"],
      "score": {"dense": 0.84, "lexical": 0.42, "hybrid": 0.74},
      "best_chunk": {"snippet": "...", "char_range": [0, 550]},
      "full_text": "..."
    }
  ],
  "stats": {
    "chunks_indexed": 1242,
    "candidates_union_articles": 455,
    "after_per_query_cap": 85,
    "after_mmr": 30
  }
}
```
