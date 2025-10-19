"""
LLM-Based Cluster Refinement Module

Uses LangGraph to orchestrate a 2-step LLM workflow:
1. Filter articles to top N most important (focused filtering)
2. Optimize clusters and generate labels (focused clustering)

Also handles:
- Moving misplaced articles to better clusters
- Merging similar/overlapping clusters
- Splitting confused mixed-topic clusters
- Creating new clusters for orphaned topics
"""

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.graph import StateGraph, END

from config import settings


class RefinementState(TypedDict):
    """State for LangGraph refinement workflow."""
    # Input
    items: List[Dict[str, Any]]
    embeddings: np.ndarray
    initial_labels: List[int]
    settings_obj: Any
    
    # Filtering step outputs
    filtered_items: List[Dict[str, Any]]
    filtered_embeddings: np.ndarray
    filtered_labels: List[int]
    filter_stats: Dict[str, Any]
    
    # Clustering step outputs
    refined_labels: List[int]
    label_map: Dict[int, str]
    clustering_stats: Dict[str, Any]
    
    # Final
    error: str
    failed: bool


def _build_cluster_summaries(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: List[int]
) -> List[Dict[str, Any]]:
    """
    Build compact summaries of each cluster for LLM review.
    Returns top 5 most central articles per cluster.
    """
    num_clusters = len(set(labels))
    summaries = []
    
    for cluster_id in range(num_clusters):
        # Get items and embeddings for this cluster
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        if not cluster_indices:
            continue
        
        cluster_items = [items[i] for i in cluster_indices]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Find most central articles (by cosine to centroid)
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        top_indices = np.argsort(similarities)[::-1][:5]  # Top 5
        
        # Build summary
        representatives = []
        for idx in top_indices:
            item = cluster_items[idx]
            global_idx = cluster_indices[idx]
            representatives.append({
                "index": global_idx,
                "title": item.get("title", ""),
                "snippet": item.get("abstract", "")[:100] + "..."
            })
        
        summaries.append({
            "cluster_id": cluster_id,
            "size": len(cluster_items),
            "representatives": representatives
        })
    
    return summaries


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_llm_for_filtering(
    items: List[Dict[str, Any]],
    settings_obj
) -> Dict[str, Any]:
    """
    Call LLM to filter articles down to top N most important.
    Separate focused call just for filtering.
    """
    client = OpenAI(api_key=settings_obj.OPENAI_API_KEY)
    
    num_items = len(items)
    num_to_keep = settings_obj.REFINEMENT_MAX_FINAL_ARTICLES
    num_to_remove = max(0, num_items - num_to_keep)
    
    # Build article list for LLM
    articles_text = ""
    for i, item in enumerate(items):
        title = item.get("title", "")
        snippet = item.get("abstract", "")[:100]
        articles_text += f"{i}. \"{title}\"\n   {snippet}...\n\n"
    
    prompt = f"""You are filtering AI/tech news for a daily podcast. Select the TOP {num_to_keep} most important articles.

Total articles: {num_items}
MUST keep: EXACTLY {num_to_keep} articles
MUST remove: {num_to_remove} articles

ARTICLES:
{articles_text}

SELECTION CRITERIA - Keep ONLY:
✅ Major AI model launches (GPT, Claude, Gemini, etc.)
✅ Significant hardware/chip announcements (NVIDIA, AMD, etc.)
✅ Major research breakthroughs and innovations
✅ Critical policy/regulation with real impact
✅ Important funding rounds or acquisitions
✅ Significant technical innovations

REMOVE:
❌ Opinion pieces and editorials
❌ General tech news unrelated to AI
❌ Minor updates or incremental changes
❌ Redundant coverage of same story
❌ Social media drama or controversies
❌ Commentary without news value

OUTPUT (JSON only):
{{
  "keep_indices": [0, 2, 5, 8, ...],  // Exactly {num_to_keep} indices
  "reasoning": "Kept major launches, hardware news, and policy updates. Removed opinion pieces and minor updates."
}}

You MUST return exactly {num_to_keep} indices in keep_indices array.
"""
    
    response = client.chat.completions.create(
        model=settings_obj.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a news curator that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings_obj.REFINEMENT_TEMPERATURE,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    
    result_text = response.choices[0].message.content
    result = json.loads(result_text)
    
    return result


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_llm_for_clustering(
    cluster_summaries: List[Dict[str, Any]],
    settings_obj
) -> Dict[str, Any]:
    """
    Call LLM to analyze clusters and suggest refinements + labels.
    Focused on clustering optimization only (filtering already done).
    """
    client = OpenAI(api_key=settings_obj.OPENAI_API_KEY)
    
    # Build prompt
    prompt = """You are an expert AI news clustering system. Review these clusters and optimize them for coherence and clarity.

Current clusters:

"""
    
    # Add cluster summaries
    for summary in cluster_summaries:
        prompt += f"\n[Cluster {summary['cluster_id']}] ({summary['size']} articles)\n"
        for i, rep in enumerate(summary['representatives'][:3], 1):
            prompt += f"  {i}. \"{rep['title']}\"\n"
            prompt += f"     {rep['snippet']}\n"
    
    prompt += f"""

YOUR TASK: Optimize these clusters for semantic coherence and generate clear, professional labels.

CLUSTERING:
Each cluster should group articles that share the same *central topic or narrative*. 
Avoid mixing unrelated domains (e.g., AI art vs hardware manufacturing vs regulation). 
Distinct clusters should differ clearly in subject matter, not just phrasing.

You can:
1. MOVE misplaced articles to better-fitting clusters (max {settings_obj.REFINEMENT_MAX_MOVES} moves)
2. MERGE clusters with overlapping or identical topics
3. SPLIT clusters that mix unrelated topics
4. CREATE new clusters for orphaned topics (max {settings_obj.REFINEMENT_MAX_NEW_CLUSTERS} new clusters)

CONSTRAINTS:
- Minimum cluster size: {settings_obj.REFINEMENT_MIN_CLUSTER_SIZE} articles
- Aim for {settings_obj.TARGET_CLUSTERS_MIN}-{settings_obj.TARGET_CLUSTERS_MAX + settings_obj.REFINEMENT_MAX_NEW_CLUSTERS} clusters total
- Avoid excessive merging that hides distinct subtopics
- Return ONLY valid JSON matching the schema below — no explanations, no commentary outside JSON

LABELS:
- Each label must capture the main theme of that cluster (≤ {settings_obj.LABEL_MAX_WORDS} words, Title Case)
- Be clear, descriptive, and human-readable (no jargon or placeholders)
- Use whatever terms best describe the content naturally
- Examples of good labels: "AI Policy & Regulation", "Hardware & Infrastructure", "Creative AI & Culture"
- Examples of bad labels: "Cluster 1", "Mixed Topics", "AI News Stuff"

OUTPUT FORMAT (JSON only):
{{
  "analysis": {{
    "problems_found": "Brief description of issues (e.g., mixed domains, duplicate topics)",
    "fixes_applied": "Summary of actions (split/merge/move)",
    "confidence": 4
  }},
  "refinements": [
    {{
      "action": "move",
      "article_index": 5,
      "from_cluster": 0,
      "to_cluster": 1,
      "reason": "Article about EU AI Act regulation fits policy cluster, not launches"
    }},
    {{
      "action": "merge",
      "clusters": [2, 3],
      "into_cluster": 2,
      "reason": "Both clusters cover NVIDIA and TSMC hardware announcements"
    }},
    {{
      "action": "split",
      "cluster": 0,
      "article_indices_for_new_cluster": [8, 12, 15],
      "reason": "Academic research papers distinct from commercial product launches"
    }}
  ],
  "final_labels": [
    {{"cluster_id": 0, "label": "AI Policy & Regulation"}},
    {{"cluster_id": 1, "label": "Hardware & Infrastructure"}},
    {{"cluster_id": 2, "label": "Creative AI & Culture"}}
  ]
}}

Reason fields must reference specific topics or keywords (e.g., "Both mention NVIDIA Blackwell hardware," not "similar theme").
If any field is uncertain, leave it empty rather than writing prose.
If clusters are already well-formed, return empty refinements list and just provide final_labels.
"""
    
    # Call API
    response = client.chat.completions.create(
        model=settings_obj.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert clustering system that outputs structured JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings_obj.REFINEMENT_TEMPERATURE,
        max_tokens=1000,
        response_format={"type": "json_object"}
    )
    
    # Parse response
    result_text = response.choices[0].message.content
    result = json.loads(result_text)
    
    return result


def _apply_refinements_clustering(
    labels: List[int],
    refinements: List[Dict[str, Any]],
    settings_obj
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Apply LLM-suggested clustering refinements (move/merge/split).
    Returns (updated_labels, stats).
    """
    labels = labels.copy()  # Don't modify original
    stats = {
        "moves": 0,
        "merges": 0,
        "splits": 0,
        "new_clusters": 0
    }
    
    max_cluster_id = max(labels) if labels else 0
    
    for refinement in refinements[:settings_obj.REFINEMENT_MAX_MOVES * 2]:  # Safety limit
        action = refinement.get("action", "")
        
        if action == "move":
            # Move single article
            if stats["moves"] >= settings_obj.REFINEMENT_MAX_MOVES:
                continue
            
            idx = refinement.get("article_index")
            from_cluster = refinement.get("from_cluster")
            to_cluster = refinement.get("to_cluster")
            
            # Validate
            if idx is None or not (0 <= idx < len(labels)):
                continue
            if labels[idx] != from_cluster:
                continue  # Article not in claimed source cluster
            if to_cluster < 0:
                continue
            
            labels[idx] = to_cluster
            stats["moves"] += 1
            print(f"[refiner] Move: article {idx} from cluster {from_cluster} → {to_cluster}")
        
        elif action == "merge":
            # Merge multiple clusters
            clusters_to_merge = refinement.get("clusters", [])
            into_cluster = refinement.get("into_cluster")
            
            if not clusters_to_merge or into_cluster is None:
                continue
            
            for i, label in enumerate(labels):
                if label in clusters_to_merge:
                    labels[i] = into_cluster
            
            stats["merges"] += 1
            print(f"[refiner] Merge: clusters {clusters_to_merge} → {into_cluster}")
        
        elif action == "split":
            # Split cluster: move specified articles to new cluster
            if stats["new_clusters"] >= settings_obj.REFINEMENT_MAX_NEW_CLUSTERS:
                continue
            
            source_cluster = refinement.get("cluster")
            article_indices = refinement.get("article_indices_for_new_cluster", [])
            
            if source_cluster is None or not article_indices:
                continue
            
            # Create new cluster
            max_cluster_id += 1
            for idx in article_indices:
                if 0 <= idx < len(labels) and labels[idx] == source_cluster:
                    labels[idx] = max_cluster_id
            
            stats["splits"] += 1
            stats["new_clusters"] += 1
            print(f"[refiner] Split: created cluster {max_cluster_id} from cluster {source_cluster} with {len(article_indices)} articles")
    
    return labels, stats


def _validate_and_fix_clusters(
    labels: List[int],
    embeddings: np.ndarray,
    settings_obj
) -> List[int]:
    """
    Validate refined clusters and fix issues (merge tiny clusters).
    """
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    # Find tiny clusters
    tiny_clusters = [cid for cid, size in cluster_sizes.items() 
                     if size < settings_obj.REFINEMENT_MIN_CLUSTER_SIZE]
    
    if not tiny_clusters:
        # Renumber clusters sequentially by size
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        old_to_new = {old: new for new, (old, _) in enumerate(sorted_clusters)}
        return [old_to_new[l] for l in labels]
    
    print(f"[refiner] Fixing {len(tiny_clusters)} tiny clusters")
    
    # Merge tiny clusters into nearest by centroid similarity
    non_tiny = [cid for cid in cluster_sizes.keys() if cid not in tiny_clusters]
    
    # Compute centroids for non-tiny clusters
    centroids = {}
    for cid in non_tiny:
        cluster_indices = [i for i, l in enumerate(labels) if l == cid]
        if cluster_indices:
            centroids[cid] = np.mean(embeddings[cluster_indices], axis=0)
    
    # Merge each tiny cluster
    for tiny_cid in tiny_clusters:
        tiny_indices = [i for i, l in enumerate(labels) if l == tiny_cid]
        if not tiny_indices or not centroids:
            continue
        
        # Find nearest cluster
        tiny_centroid = np.mean(embeddings[tiny_indices], axis=0)
        best_cid = None
        best_sim = -1
        
        for cid, centroid in centroids.items():
            sim = cosine_similarity(tiny_centroid.reshape(1, -1), centroid.reshape(1, -1))[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        
        # Merge
        if best_cid is not None:
            for i in tiny_indices:
                labels[i] = best_cid
    
    # Renumber clusters sequentially by size
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_clusters)}
    
    return [old_to_new[l] for l in labels]


# ===== LangGraph Workflow Nodes =====

def filter_articles_node(state: RefinementState) -> RefinementState:
    """
    LangGraph Node 1: Filter articles to top N most important.
    """
    items = state["items"]
    embeddings = state["embeddings"]
    initial_labels = state["initial_labels"]
    settings_obj = state["settings_obj"]
    
    print(f"[refiner] Node 1: Filtering {len(items)} → {settings_obj.REFINEMENT_MAX_FINAL_ARTICLES} articles...")
    
    # Skip filtering if already at or below target
    if len(items) <= settings_obj.REFINEMENT_MAX_FINAL_ARTICLES:
        print(f"[refiner] Already at target, skipping filtering")
        state["filtered_items"] = items
        state["filtered_embeddings"] = embeddings
        state["filtered_labels"] = initial_labels
        state["filter_stats"] = {"skipped": True}
        return state
    
    try:
        time.sleep(2)  # Rate limit protection
        
        filter_result = _call_llm_for_filtering(items, settings_obj)
        keep_indices = filter_result.get("keep_indices", [])
        filter_reasoning = filter_result.get("reasoning", "")
        
        # Validate indices
        keep_indices = [idx for idx in keep_indices if 0 <= idx < len(items)]
        
        if keep_indices:
            filtered_items = [items[i] for i in keep_indices]
            filtered_embeddings = embeddings[keep_indices]
            filtered_labels = [initial_labels[i] for i in keep_indices]
            
            state["filtered_items"] = filtered_items
            state["filtered_embeddings"] = filtered_embeddings
            state["filtered_labels"] = filtered_labels
            state["filter_stats"] = {
                "articles_before": len(items),
                "articles_after": len(filtered_items),
                "removed": len(items) - len(filtered_items),
                "reasoning": filter_reasoning[:200]
            }
            
            print(f"[refiner] ✓ Filtered: {len(items)} → {len(filtered_items)} articles")
            print(f"[refiner] Reason: {filter_reasoning}")
        else:
            # Fallback: keep all
            state["filtered_items"] = items
            state["filtered_embeddings"] = embeddings
            state["filtered_labels"] = initial_labels
            state["filter_stats"] = {"failed": True, "kept_all": True}
            
    except Exception as e:
        print(f"[refiner] Filtering failed: {e}, keeping all articles")
        state["filtered_items"] = items
        state["filtered_embeddings"] = embeddings
        state["filtered_labels"] = initial_labels
        state["filter_stats"] = {"failed": True, "error": str(e)}
        state["failed"] = True
    
    return state


def cluster_refinement_node(state: RefinementState) -> RefinementState:
    """
    LangGraph Node 2: Refine clusters and generate labels.
    """
    filtered_items = state["filtered_items"]
    filtered_embeddings = state["filtered_embeddings"]
    filtered_labels = state["filtered_labels"]
    settings_obj = state["settings_obj"]
    
    print(f"[refiner] Node 2: Refining {len(filtered_items)} articles in {len(set(filtered_labels))} clusters...")
    
    try:
        time.sleep(2)  # Rate limit protection
        
        # Build cluster summaries
        cluster_summaries = _build_cluster_summaries(filtered_items, filtered_embeddings, filtered_labels)
        
        # Call LLM for clustering
        result = _call_llm_for_clustering(cluster_summaries, settings_obj)
        
        refinements = result.get("refinements", [])
        final_labels_list = result.get("final_labels", [])
        analysis = result.get("analysis", {})
        
        # Handle analysis
        if isinstance(analysis, dict):
            analysis_text = f"{analysis.get('problems_found', '')} | {analysis.get('fixes_applied', '')}"
            confidence = analysis.get('confidence', 0)
            print(f"[refiner] Analysis: {analysis_text}")
            print(f"[refiner] Confidence: {confidence}/5")
        else:
            analysis_text = str(analysis)
            confidence = 0
        
        print(f"[refiner] Refinements: {len(refinements)}")
        
        # Apply clustering refinements
        refined_labels, apply_stats = _apply_refinements_clustering(filtered_labels, refinements, settings_obj)
        
        # Validate and fix
        refined_labels = _validate_and_fix_clusters(refined_labels, filtered_embeddings, settings_obj)
        
        # Build label map
        label_map = {}
        for label_item in final_labels_list:
            cluster_id = label_item.get("cluster_id")
            label = label_item.get("label", "")
            if cluster_id is not None:
                label_map[cluster_id] = label
        
        # Ensure all clusters have labels
        final_num_clusters = len(set(refined_labels))
        for cid in range(final_num_clusters):
            if cid not in label_map:
                label_map[cid] = f"Topic {cid + 1}"
        
        state["refined_labels"] = refined_labels
        state["label_map"] = label_map
        state["clustering_stats"] = {
            "moves": apply_stats["moves"],
            "merges": apply_stats["merges"],
            "splits": apply_stats["splits"],
            "new_clusters_created": apply_stats["new_clusters"],
            "clusters_after": final_num_clusters,
            "llm_analysis": analysis_text[:200],
            "confidence": confidence
        }
        
        print(f"[refiner] ✓ Final: {final_num_clusters} clusters with labels")
        
    except Exception as e:
        print(f"[refiner] Clustering failed: {e}")
        state["refined_labels"] = filtered_labels
        state["label_map"] = {}
        state["clustering_stats"] = {"failed": True, "error": str(e)}
        state["failed"] = True
        state["error"] = str(e)
    
    return state


def refine_clusters_with_llm(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    initial_labels: List[int],
    settings_obj
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int], Dict[int, str], Dict[str, Any]]:
    """
    Main entry point: Use LangGraph to orchestrate 2-step LLM refinement.
    
    Args:
        items: List of article dictionaries
        embeddings: Normalized embeddings matrix
        initial_labels: Initial cluster assignments from algorithmic clustering
        settings_obj: Settings object
    
    Returns:
        - filtered_items: Filtered list of important articles (max 20)
        - filtered_embeddings: Embeddings for filtered articles
        - refined_labels: Updated cluster assignments for filtered articles
        - label_map: {cluster_id: label}
        - refinement_stats: Metrics about changes made
    """
    print(f"[refiner] ===== LLM Cluster Refinement (LangGraph) =====")
    print(f"[refiner] Initial: {len(items)} articles in {len(set(initial_labels))} clusters")
    
    start_time = time.time()
    
    # Build LangGraph workflow
    workflow = StateGraph(RefinementState)
    
    # Add nodes
    workflow.add_node("filter", filter_articles_node)
    workflow.add_node("cluster", cluster_refinement_node)
    
    # Define edges: filter → cluster → END
    workflow.set_entry_point("filter")
    workflow.add_edge("filter", "cluster")
    workflow.add_edge("cluster", END)
    
    # Compile graph
    app = workflow.compile()
    
    # Initialize state
    initial_state: RefinementState = {
        "items": items,
        "embeddings": embeddings,
        "initial_labels": initial_labels,
        "settings_obj": settings_obj,
        "filtered_items": [],
        "filtered_embeddings": np.array([]),
        "filtered_labels": [],
        "filter_stats": {},
        "refined_labels": [],
        "label_map": {},
        "clustering_stats": {},
        "error": "",
        "failed": False
    }
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    # Extract results
    filtered_items = final_state["filtered_items"]
    filtered_embeddings = final_state["filtered_embeddings"]
    refined_labels = final_state["refined_labels"]
    label_map = final_state["label_map"]
    
    # Combine stats
    elapsed = time.time() - start_time
    stats = {
        "enabled": True,
        "failed": final_state.get("failed", False),
        "filtering": final_state.get("filter_stats", {}),
        "clustering": final_state.get("clustering_stats", {}),
        "articles_before": len(items),
        "articles_after": len(filtered_items),
        "time_sec": round(elapsed, 2),
        "workflow": "langgraph_2step"
    }
    
    if final_state.get("error"):
        stats["error"] = final_state["error"]
    
    print(f"[refiner] Final: {len(filtered_items)} articles in {len(set(refined_labels))} clusters")
    print(f"[refiner] LangGraph workflow complete in {elapsed:.1f}s")
    
    return filtered_items, filtered_embeddings, refined_labels, label_map, stats

