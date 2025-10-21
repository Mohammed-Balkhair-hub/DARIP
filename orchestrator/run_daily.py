from agents.collector import collect_all
from agents.dedup_cluster import dedup_and_cluster
from agents.fulltext_enricher import enrich_fulltext
from agents.rag_retriever import retrieve_with_rag


def main():
    # Step 1: Collection (fetch articles from RSS)
    collect_all()
    
    # Step 2: Full-Text Enrichment (fetch full text for all raw items)
    enrich_fulltext()
    
    # Step 3: RAG Query Retrieval (NEW - replaces clustering)
    retrieve_with_rag()
    
    # Step 3 (OLD): Deduplication & Clustering - COMMENTED OUT
    # dedup_and_cluster()
    
    print(f"[run_daily] ===== PIPELINE COMPLETE =====")

if __name__ == "__main__":
    main()