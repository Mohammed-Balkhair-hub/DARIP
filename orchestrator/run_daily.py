from agents.collector import collect_all
from agents.fulltext_enricher import enrich_fulltext
from agents.rag_retriever import retrieve_with_rag
from agents.script_writer import script_writer


def main():
    # Step 1: Collection (fetch articles from RSS)
    #collect_all()
    
    # Step 2: Full-Text Enrichment (fetch full text for all raw items)
    #enrich_fulltext()
    
    # Step 3: RAG Query Retrieval (selects top 30 articles)
    #retrieve_with_rag()
    
    # Step 4: Generate Podcast Script
    script_writer()
    
    print(f"[run_daily] ===== PIPELINE COMPLETE =====")

if __name__ == "__main__":
    main()