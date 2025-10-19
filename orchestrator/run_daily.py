from agents.collector import collect_all
from agents.dedup_cluster import dedup_and_cluster


def main():
    # Step 1: Collection
    collect_all()
    
    # Step 2: Deduplication & Clustering
    dedup_and_cluster()
    
    print(f"[run_daily] ===== PIPELINE COMPLETE =====")

if __name__ == "__main__":
    main()