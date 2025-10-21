"""
Step 3: Deduplication & Clustering Agent

Cleans daily collected stories, removes exact and near-duplicates,
groups remaining stories into coherent clusters, and assigns
human-readable labels using hierarchical compression + LLM.
"""

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs, urlunparse

import numpy as np
import tldextract
from dateutil import parser as dateparser
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


def _load_and_validate_input(input_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate raw items from JSON file.
    Handle both flat array and {articles: [...]} wrapper formats.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both formats
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "articles" in data:
        items = data["articles"]
    else:
        items = []
    
    # Filter items with missing title or url, AND items without full text
    valid_items = []
    for item in items:
        if not item.get("title") or not item.get("url"):
            continue
        
        # ONLY process items that have full text (enriched)
        if not item.get("has_fulltext", False):
            continue
        
        # Map content → abstract if abstract is missing
        if not item.get("abstract") and item.get("content"):
            item["abstract"] = item["content"]
        
        # Use full_text for abstract if available (better for clustering)
        if item.get("full_text"):
            item["abstract"] = item["full_text"]
        
        # Extract source_domain if missing
        if not item.get("source_domain"):
            try:
                extracted = tldextract.extract(item["url"])
                item["source_domain"] = f"{extracted.domain}.{extracted.suffix}"
            except Exception:
                item["source_domain"] = "unknown"
        
        valid_items.append(item)
    
    return valid_items


def _normalize_title(title: str) -> str:
    """Normalize title: lowercase, strip, collapse whitespace, remove trailing punctuation."""
    # Lowercase and strip
    normalized = title.lower().strip()
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove trailing punctuation
    normalized = re.sub(r'[.,;:\-–—]+$', '', normalized)
    return normalized


def _canonicalize_url(url: str, settings) -> str:
    """
    Canonicalize URL:
    - Lowercase host if LOWERCASE_HOST
    - Remove tracking params
    - Drop fragments
    - Normalize www
    - Strip trailing slash
    """
    try:
        parsed = urlparse(url)
        
        # Lowercase host
        host = parsed.netloc
        if settings.LOWERCASE_HOST:
            host = host.lower()
        
        # Normalize www
        if settings.CANONICALIZE_WWW:
            if host.startswith("www."):
                host = host[4:]
        
        # Remove tracking params
        query_params = parse_qs(parsed.query)
        filtered_params = {
            k: v for k, v in query_params.items()
            if k not in settings.STRIP_QUERY_PARAMS
        }
        new_query = "&".join(f"{k}={v[0]}" for k, v in filtered_params.items())
        
        # Remove fragment
        fragment = ""
        
        # Strip trailing slash from path
        path = parsed.path
        if settings.STRIP_TRAILING_SLASH and path.endswith("/") and len(path) > 1:
            path = path[:-1]
        
        # Reconstruct URL
        canonical = urlunparse((
            parsed.scheme,
            host,
            path,
            parsed.params,
            new_query,
            fragment
        ))
        
        return canonical
    except Exception:
        return url


def _normalize_items(items: List[Dict[str, Any]], settings) -> List[Dict[str, Any]]:
    """Add normalization fields to items."""
    for item in items:
        item["title_norm"] = _normalize_title(item["title"])
        item["canonical_url"] = _canonicalize_url(item["url"], settings)
    return items


def _exact_dedup(items: List[Dict[str, Any]], settings) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove exact duplicates based on (title_norm, canonical_url).
    Representative selection:
    1. Higher SOURCE_PRIORITY
    2. Longer abstract
    3. Earlier published_at
    4. Lexicographically smallest canonical_url
    """
    original_count = len(items)
    
    # Group by key
    groups = defaultdict(list)
    for item in items:
        key = (item["title_norm"], item["canonical_url"])
        groups[key].append(item)
    
    # Select representatives
    deduped = []
    for key, duplicates in groups.items():
        if len(duplicates) == 1:
            deduped.append(duplicates[0])
        else:
            # Sort by priority
            def sort_key(item):
                source_domain = item.get("source_domain", "")
                priority = settings.SOURCE_PRIORITY.get(source_domain, 0)
                abstract_len = len(item.get("abstract", ""))
                
                # Parse published_at
                try:
                    pub_dt = dateparser.parse(item.get("published_at", ""))
                    pub_timestamp = pub_dt.timestamp() if pub_dt else 0
                except Exception:
                    pub_timestamp = 0
                
                canonical_url = item.get("canonical_url", "")
                
                # Return tuple for sorting (descending priority, descending abstract_len, ascending timestamp, ascending url)
                return (-priority, -abstract_len, pub_timestamp, canonical_url)
            
            duplicates.sort(key=sort_key)
            deduped.append(duplicates[0])
    
    return deduped, original_count


def _build_embed_text(item: Dict[str, Any], settings) -> str:
    """
    Build consistent embed text: title + first N chars of abstract.
    Normalizes RSS length differences and improves cluster separability.
    """
    title = item.get("title", "")
    abstract = item.get("abstract", "") or item.get("summary", "")
    
    if abstract:
        # Truncate to first N chars
        abstract_snippet = abstract[:settings.EMBED_TEXT_MAX_ABSTRACT_CHARS]
        return f"{title} — {abstract_snippet}"
    return title


def _generate_embeddings(items: List[Dict[str, Any]], settings) -> np.ndarray:
    """
    Generate L2-normalized embeddings using sentence-transformers.
    Uses hybrid text: title + first N chars of abstract for better clustering.
    """
    print(f"[dedup_cluster] Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    
    # Build standardized embed texts
    texts = []
    for item in items:
        text = _build_embed_text(item, settings)
        texts.append(text)
    
    print(f"[dedup_cluster] Generating embeddings for {len(texts)} items")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # L2-normalize for better cosine similarity behavior
    embeddings = normalize(embeddings, norm='l2')
    
    return embeddings


def _near_dedup(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    settings
) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    """
    Near-duplicate pruning using cosine similarity.
    Greedy suppression: keep newest, suppress similar items >= threshold.
    """
    pre_filter_count = len(items)
    
    # Sort by published_at descending (newest first)
    indexed_items = list(enumerate(items))
    
    def get_timestamp(idx_item):
        idx, item = idx_item
        try:
            pub_dt = dateparser.parse(item.get("published_at", ""))
            return pub_dt.timestamp() if pub_dt else 0
        except Exception:
            return 0
    
    indexed_items.sort(key=get_timestamp, reverse=True)
    
    # Greedy suppression
    kept_indices = []
    suppressed = set()
    
    for idx, item in indexed_items:
        if idx in suppressed:
            continue
        
        kept_indices.append(idx)
        
        # Suppress similar items
        for other_idx, other_item in indexed_items:
            if other_idx == idx or other_idx in suppressed:
                continue
            
            similarity = cosine_similarity(
                embeddings[idx].reshape(1, -1),
                embeddings[other_idx].reshape(1, -1)
            )[0, 0]
            
            if similarity >= settings.NEAR_DUP_THRESHOLD:
                suppressed.add(other_idx)
    
    # Filter items and embeddings
    kept_indices.sort()  # Restore original order
    filtered_items = [items[idx] for idx in kept_indices]
    filtered_embeddings = embeddings[kept_indices]
    
    return filtered_items, filtered_embeddings, pre_filter_count


def _cluster_dynamic(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    settings
) -> Tuple[List[int], int, Dict[str, Any]]:
    """
    Dynamic K clustering using agglomerative auto-tuning.
    Discovers optimal K in target range [TARGET_CLUSTERS_MIN, TARGET_CLUSTERS_MAX].
    Returns (cluster_labels, num_clusters, clustering_stats).
    """
    n_items = len(items)
    
    # Edge case: too few items
    if n_items < settings.MIN_ITEMS_FOR_CLUSTERING:
        print(f"[dedup_cluster] Only {n_items} items, using single cluster")
        stats = {
            "strategy": "single_cluster",
            "N_items": n_items,
            "K": 1
        }
        return [0] * n_items, 1, stats
    
    # Try agglomerative auto-tuning
    try:
        return _agglomerative_auto(items, embeddings, settings)
    except Exception as e:
        print(f"[dedup_cluster] Agglomerative failed: {e}, falling back to KMeans scan")
        return _kmeans_scan_fallback(items, embeddings, settings)


def _agglomerative_auto(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    settings
) -> Tuple[List[int], int, Dict[str, Any]]:
    """
    Agglomerative clustering with auto-tuning to hit target K range.
    """
    n_items = len(items)
    cutoff = settings.AGGLO_INITIAL_CUTOFF
    
    print(f"[dedup_cluster] Starting agglomerative auto-tuning (initial cutoff={cutoff:.2f})")
    
    # A1 & A2: Auto-tune loop
    best_labels = None
    best_k = 0
    
    for iteration in range(settings.AGGLO_MAX_ITERS):
        # Cluster with current cutoff
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cutoff,
            metric='cosine',
            linkage='average'
        )
        labels = clusterer.fit_predict(embeddings)
        k = len(set(labels))
        
        print(f"[dedup_cluster] Iter {iteration+1}: cutoff={cutoff:.3f} → K={k}")
        
        best_labels = labels
        best_k = k
        
        # Check if in target range
        if settings.TARGET_CLUSTERS_MIN <= k <= settings.TARGET_CLUSTERS_MAX:
            print(f"[dedup_cluster] K={k} in target range, stopping tuning")
            break
        
        # Adjust cutoff
        if k < settings.TARGET_CLUSTERS_MIN:
            # Too few clusters, decrease cutoff (more splits)
            cutoff -= 0.03
            cutoff = max(cutoff, settings.AGGLO_CUTOFF_MIN)
        elif k > settings.TARGET_CLUSTERS_MAX:
            # Too many clusters, increase cutoff (more merges)
            cutoff += 0.03
            cutoff = min(cutoff, settings.AGGLO_CUTOFF_MAX)
    
    # A3: Post-processing
    labels_list = best_labels.tolist()
    labels_list, final_k = _postprocess_clusters(items, embeddings, labels_list, settings)
    
    # Calculate silhouette score
    if final_k > 1:
        try:
            silhouette = silhouette_score(embeddings, labels_list, metric='cosine')
        except:
            silhouette = 0.0
    else:
        silhouette = 0.0
    
    # Calculate cluster sizes
    cluster_sizes = defaultdict(int)
    for label in labels_list:
        cluster_sizes[label] += 1
    
    sizes = list(cluster_sizes.values())
    largest_size = max(sizes) if sizes else 0
    smallest_size = min(sizes) if sizes else 0
    
    stats = {
        "strategy": "agglomerative_auto",
        "N_items": n_items,
        "initial_cutoff": settings.AGGLO_INITIAL_CUTOFF,
        "final_cutoff": round(cutoff, 3),
        "tuning_iterations": iteration + 1,
        "K": final_k,
        "largest_cluster_size": largest_size,
        "largest_cluster_frac": round(largest_size / n_items, 3) if n_items > 0 else 0,
        "smallest_cluster_size": smallest_size,
        "silhouette_cosine": round(silhouette, 3)
    }
    
    print(f"[dedup_cluster] Final: K={final_k}, silhouette={silhouette:.3f}")
    
    return labels_list, final_k, stats


def _postprocess_clusters(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    labels: List[int],
    settings
) -> Tuple[List[int], int]:
    """
    Post-processing: merge tiny clusters, split oversized clusters.
    """
    n_items = len(items)
    
    # Step 1: Merge tiny clusters
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    tiny_clusters = [cid for cid, size in cluster_sizes.items() 
                     if size < settings.MIN_FINAL_CLUSTER_SIZE]
    
    if tiny_clusters:
        print(f"[dedup_cluster] Merging {len(tiny_clusters)} tiny clusters")
        
        # Compute centroids for non-tiny clusters
        non_tiny = [cid for cid in cluster_sizes.keys() if cid not in tiny_clusters]
        centroids = {}
        for cid in non_tiny:
            cluster_indices = [i for i, l in enumerate(labels) if l == cid]
            if cluster_indices:
                centroids[cid] = np.mean(embeddings[cluster_indices], axis=0)
        
        # Merge each tiny cluster into nearest
        for tiny_cid in tiny_clusters:
            tiny_indices = [i for i, l in enumerate(labels) if l == tiny_cid]
            if not tiny_indices or not centroids:
                continue
            
            # Find nearest cluster by centroid cosine
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
    
    # Step 2: Check for oversized clusters
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    max_size = max(cluster_sizes.values()) if cluster_sizes else 0
    if max_size > settings.MAX_LARGEST_CLUSTER_FRAC * n_items and max_size > settings.MIN_FINAL_CLUSTER_SIZE * 2:
        print(f"[dedup_cluster] Largest cluster ({max_size} items) > {settings.MAX_LARGEST_CLUSTER_FRAC*100:.0f}%, attempting split")
        
        # Find the largest cluster
        largest_cid = max(cluster_sizes.items(), key=lambda x: x[1])[0]
        large_indices = [i for i, l in enumerate(labels) if l == largest_cid]
        large_embeddings = embeddings[large_indices]
        
        # Try 2-way split
        if len(large_indices) >= settings.MIN_FINAL_CLUSTER_SIZE * 2:
            try:
                sub_clusterer = AgglomerativeClustering(
                    n_clusters=2,
                    metric='cosine',
                    linkage='average'
                )
                sub_labels = sub_clusterer.fit_predict(large_embeddings)
                
                # Check if both children are large enough
                sub_sizes = [sum(1 for l in sub_labels if l == 0), sum(1 for l in sub_labels if l == 1)]
                if all(s >= settings.MIN_FINAL_CLUSTER_SIZE for s in sub_sizes):
                    # Apply split: reassign half to a new cluster ID
                    new_cid = max(set(labels)) + 1
                    for i, sub_label in enumerate(sub_labels):
                        if sub_label == 1:
                            labels[large_indices[i]] = new_cid
                    print(f"[dedup_cluster] Split applied: {sub_sizes[0]} + {sub_sizes[1]} items")
            except Exception as e:
                print(f"[dedup_cluster] Split failed: {e}")
    
    # Finalize: renumber clusters by size (descending)
    cluster_sizes = defaultdict(int)
    for label in labels:
        cluster_sizes[label] += 1
    
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_clusters)}
    
    labels = [old_to_new[l] for l in labels]
    final_k = len(set(labels))
    
    return labels, final_k


def _kmeans_scan_fallback(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    settings
) -> Tuple[List[int], int, Dict[str, Any]]:
    """
    Fallback: KMeans scan to find best K by silhouette score.
    """
    n_items = len(items)
    k_min = max(2, settings.TARGET_CLUSTERS_MIN)
    k_max = min(n_items - 1, settings.TARGET_CLUSTERS_MAX)
    
    best_k = k_min
    best_labels = None
    best_score = -1
    
    print(f"[dedup_cluster] KMeans scan fallback: trying K ∈ [{k_min}, {k_max}]")
    
    for k in range(k_min, k_max + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=settings.KMEANS_RANDOM_STATE, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels, metric='cosine')
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except:
            continue
    
    if best_labels is None:
        # Ultimate fallback
        best_labels = [0] * n_items
        best_k = 1
    
    stats = {
        "strategy": "kmeans_scan_fallback",
        "N_items": n_items,
        "K": best_k,
        "silhouette_cosine": round(best_score, 3) if best_score > -1 else 0.0
    }
    
    return best_labels.tolist() if hasattr(best_labels, 'tolist') else best_labels, best_k, stats


def _trim_abstract(abstract: str, settings) -> str:
    """
    Trim abstract to MAX_ABSTRACT_SENTENCES and MAX_ABSTRACT_CHARS.
    Remove outlet names, bylines, embedded URLs.
    """
    if not abstract:
        return ""
    
    # Remove embedded URLs
    text = re.sub(r'https?://\S+', '', abstract)
    
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Take first MAX_ABSTRACT_SENTENCES
    trimmed_sentences = sentences[:settings.MAX_ABSTRACT_SENTENCES]
    trimmed = ". ".join(trimmed_sentences)
    
    # Trim to MAX_ABSTRACT_CHARS
    if len(trimmed) > settings.MAX_ABSTRACT_CHARS:
        trimmed = trimmed[:settings.MAX_ABSTRACT_CHARS].rsplit(' ', 1)[0]
    
    return trimmed


def _extract_keyphrases(abstracts: List[str], settings) -> List[str]:
    """
    Simple keyphrase extraction: find multi-word noun phrases and frequent terms.
    This is a simplified heuristic version.
    """
    # Combine all abstracts
    combined = " ".join(abstracts)
    
    # Extract potential keyphrases (2-3 word sequences)
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', combined)
    
    # Count frequencies
    freq = defaultdict(int)
    for phrase in words:
        freq[phrase] += 1
    
    # Sort by frequency and take top N
    sorted_phrases = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keyphrases = [phrase for phrase, count in sorted_phrases[:settings.MAX_LABELING_KEYPHRASES]]
    
    return keyphrases


def _prepare_label_inputs(
    items: List[Dict[str, Any]],
    embeddings: np.ndarray,
    cluster_labels: List[int],
    settings
) -> List[Dict[str, Any]]:
    """
    Prepare compressed slate for each cluster for labeling.
    Returns list of {cluster_id, keyphrases, representatives}.
    """
    n_clusters = max(cluster_labels) + 1
    slates = []
    
    for cluster_id in range(n_clusters):
        # Get items in this cluster
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_items = [items[i] for i in cluster_indices]
        cluster_embeddings = embeddings[cluster_indices]
        
        if len(cluster_items) == 0:
            continue
        
        # Trim abstracts
        trimmed_abstracts = []
        for item in cluster_items:
            abstract = item.get("abstract", "")
            trimmed = _trim_abstract(abstract, settings)
            trimmed_abstracts.append(trimmed)
        
        # Generate embeddings for trimmed abstracts
        if len(trimmed_abstracts) > 0:
            model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            trimmed_embeddings = model.encode(trimmed_abstracts, show_progress_bar=False, convert_to_numpy=True)
        else:
            trimmed_embeddings = cluster_embeddings
        
        # Intra-cluster dedup (0.90 threshold)
        kept_indices = []
        suppressed = set()
        
        for i in range(len(trimmed_abstracts)):
            if i in suppressed:
                continue
            kept_indices.append(i)
            
            for j in range(i + 1, len(trimmed_abstracts)):
                if j in suppressed:
                    continue
                
                similarity = cosine_similarity(
                    trimmed_embeddings[i].reshape(1, -1),
                    trimmed_embeddings[j].reshape(1, -1)
                )[0, 0]
                
                if similarity >= settings.INTRA_CLUSTER_DUP_THRESHOLD:
                    suppressed.add(j)
        
        # Ensure source diversity
        kept_cluster_items = [cluster_items[i] for i in kept_indices]
        sources = set(item.get("source_domain", "") for item in kept_cluster_items)
        
        if len(sources) < settings.LABELING_MIN_SOURCE_DIVERSITY and len(sources) < len(cluster_items):
            # Add items from different sources
            existing_sources = set(item.get("source_domain", "") for item in kept_cluster_items)
            for i, item in enumerate(cluster_items):
                if i not in kept_indices:
                    if item.get("source_domain", "") not in existing_sources:
                        kept_indices.append(i)
                        existing_sources.add(item.get("source_domain", ""))
                        if len(existing_sources) >= settings.LABELING_MIN_SOURCE_DIVERSITY:
                            break
        
        # Select top-K by centroid similarity
        if len(kept_indices) > settings.TOP_K_FOR_LABELING:
            kept_embeddings = cluster_embeddings[kept_indices]
            centroid = np.mean(kept_embeddings, axis=0)
            
            similarities = cosine_similarity(kept_embeddings, centroid.reshape(1, -1)).flatten()
            top_k_indices = np.argsort(similarities)[::-1][:settings.TOP_K_FOR_LABELING]
            kept_indices = [kept_indices[i] for i in top_k_indices]
        
        # Extract keyphrases
        all_abstracts = [item.get("abstract", "") for item in cluster_items]
        keyphrases = _extract_keyphrases(all_abstracts, settings)
        
        # Build representatives
        representatives = []
        for idx in kept_indices:
            item = cluster_items[idx]
            abstract_trimmed = trimmed_abstracts[idx]
            
            # Micro summary (first sentence, capped at word limit)
            if settings.USE_LLM_FOR_MICRO_SUMMARIES:
                # Not implemented for now
                micro_summary = abstract_trimmed
            else:
                # First sentence, capped at word limit
                first_sentence = abstract_trimmed.split('.')[0] if abstract_trimmed else ""
                words = first_sentence.split()
                if len(words) > settings.MICRO_SUMMARY_WORD_LIMIT:
                    words = words[:settings.MICRO_SUMMARY_WORD_LIMIT]
                micro_summary = " ".join(words)
            
            representatives.append({
                "title": item.get("title", ""),
                "abstract_trimmed": abstract_trimmed,
                "micro_summary": micro_summary
            })
        
        slates.append({
            "cluster_id": cluster_id,
            "keyphrases": keyphrases,
            "representatives": representatives
        })
    
    return slates


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_openai_for_labels(slates: List[Dict[str, Any]], settings) -> Dict[str, Any]:
    """Call OpenAI API to generate cluster labels."""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Build prompt
    prompt = """You are a news clustering system. Given clusters of AI-related news stories, generate concise, distinct labels for each cluster.

Requirements:
- Each label must be ≤ 4 words
- Use Title Case
- Labels must be distinct from each other
- Prefer these terms when relevant: Policy, Regulation, Models, Launches, Research, Hardware, Security, Funding, Ecosystem
- Avoid company/publisher names unless essential

Output format (JSON only):
{
  "labels": [
    {"cluster_id": 0, "label": "AI Policy & Regulation"},
    {"cluster_id": 1, "label": "Major Model Launches"}
  ]
}

Clusters to label:
"""
    
    for slate in slates:
        prompt += f"\n\nCluster {slate['cluster_id']}:\n"
        prompt += f"Keyphrases: {', '.join(slate['keyphrases'][:5])}\n"
        prompt += "Sample stories:\n"
        for i, rep in enumerate(slate['representatives'][:3], 1):
            prompt += f"{i}. {rep['title']}\n"
            if rep['micro_summary']:
                prompt += f"   {rep['micro_summary']}\n"
    
    # Call API
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        response_format={"type": "json_object"}
    )
    
    # Parse response
    result_text = response.choices[0].message.content
    result = json.loads(result_text)
    
    return result


def _heuristic_labeler(slates: List[Dict[str, Any]], settings) -> Dict[str, Any]:
    """
    Fallback heuristic labeler using keyword mapping.
    """
    labels = []
    
    keyword_map = {
        "policy|regulation|compliance|law|govern": "AI Policy & Regulation",
        "launch|release|model|gpt|llm|parameter": "Major Model Launches",
        "research|paper|benchmark|sota|diffusion|transformer": "Research & Tech Advances",
        "hardware|gpu|chip|nvidia|tpu|computing": "AI Hardware & Infrastructure",
        "security|safety|abuse|guardrail|risk": "AI Safety & Security",
        "funding|investment|vc|acquisition|raise": "Funding & Investment"
    }
    
    for slate in slates:
        cluster_id = slate["cluster_id"]
        
        # Combine keyphrases and titles
        text = " ".join(slate["keyphrases"]).lower()
        text += " " + " ".join(rep["title"].lower() for rep in slate["representatives"])
        
        # Find best match
        best_label = "AI News & Updates"
        for pattern, label in keyword_map.items():
            if re.search(pattern, text):
                best_label = label
                break
        
        labels.append({"cluster_id": cluster_id, "label": best_label})
    
    # Ensure distinctness
    seen_labels = set()
    for item in labels:
        original_label = item["label"]
        counter = 2
        while item["label"] in seen_labels:
            item["label"] = f"{original_label} ({counter})"
            counter += 1
        seen_labels.add(item["label"])
    
    return {"labels": labels}


def _generate_labels_with_llm(slates: List[Dict[str, Any]], settings) -> Dict[str, Any]:
    """
    Generate labels using LLM with retry and fallback.
    """
    try:
        result = _call_openai_for_labels(slates, settings)
        
        # Validate result
        if "labels" not in result:
            raise ValueError("Missing 'labels' in response")
        
        # Ensure coverage and distinctness
        cluster_ids = set(slate["cluster_id"] for slate in slates)
        result_ids = set(item["cluster_id"] for item in result["labels"])
        
        if cluster_ids != result_ids:
            raise ValueError("Cluster IDs mismatch")
        
        # Check distinctness
        label_texts = [item["label"] for item in result["labels"]]
        if len(label_texts) != len(set(label_texts)):
            # Try to fix duplicates
            seen = set()
            for item in result["labels"]:
                original = item["label"]
                counter = 2
                while item["label"] in seen:
                    item["label"] = f"{original} ({counter})"
                    counter += 1
                seen.add(item["label"])
        
        return result
        
    except Exception as e:
        print(f"[dedup_cluster] LLM labeling failed: {e}, using heuristic fallback")
        return _heuristic_labeler(slates, settings)


def _assemble_clusters_output(
    items: List[Dict[str, Any]],
    cluster_labels: List[int],
    label_map: Dict[int, str],
    date_iso: str,
    settings,
    original_count: int = 0,
    after_exact_dedup: int = 0
) -> Dict[str, Any]:
    """
    Assemble final clusters.json output.
    """
    n_clusters = max(cluster_labels) + 1
    total_items = len(items)
    clusters = []
    
    for cluster_id in range(n_clusters):
        # Get items in cluster
        cluster_items = [
            items[i] for i, label in enumerate(cluster_labels)
            if label == cluster_id
        ]
        
        # Sort by published_at desc, then source_priority desc
        def sort_key(item):
            try:
                pub_dt = dateparser.parse(item.get("published_at", ""))
                pub_timestamp = -(pub_dt.timestamp() if pub_dt else 0)
            except Exception:
                pub_timestamp = 0
            
            source_domain = item.get("source_domain", "")
            priority = -settings.SOURCE_PRIORITY.get(source_domain, 0)
            
            return (pub_timestamp, priority)
        
        cluster_items.sort(key=sort_key)
        
        # Clean items (remove normalization fields and embeddings)
        clean_items = []
        for item in cluster_items:
            clean_item = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "canonical_url": item.get("canonical_url", ""),
                "source": item.get("source", ""),
                "source_domain": item.get("source_domain", ""),
                "published_at": item.get("published_at", ""),
                "summary": item.get("summary", ""),
                "full_text": item.get("full_text", ""),
                "language": item.get("language", "en")
            }
            clean_items.append(clean_item)
        
        clusters.append({
            "cluster_id": cluster_id,
            "label": label_map.get(cluster_id, f"Cluster {cluster_id}"),
            "count": len(clean_items),
            "items": clean_items
        })
    
    output = {
        "date": date_iso,
        "count": total_items,
        "original_count": original_count,
        "after_exact_dedup": after_exact_dedup,
        "duplicates_removed": original_count - total_items if original_count > 0 else 0,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "near_dup_threshold": settings.NEAR_DUP_THRESHOLD,
        "num_clusters": n_clusters,
        "clusters": clusters
    }
    
    return output


def _write_metrics(metrics_data: Dict[str, Any], output_dir: str) -> None:
    """Write metrics to _logs/step3_metrics.json"""
    log_dir = os.path.join(output_dir, settings.LOG_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)
    
    metrics_path = os.path.join(log_dir, "step3_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    print(f"[dedup_cluster] Metrics written to {metrics_path}")


def dedup_and_cluster() -> None:
    """
    Main entry point for Step 3: Deduplication & Clustering.
    
    Automatically finds today's raw_items.json and processes it.
    All paths and dates are determined from settings.
    All outputs are written to JSON files (clusters.json, metrics, etc).
    """
    # Get today's date and paths from settings
    date_iso = settings.TODAY
    output_dir_for_day = settings.OUTPUT_TODAY_DIR
    input_raw_items_path = os.path.join(output_dir_for_day, "raw_items.json")
    
    print(f"[dedup_cluster] ===== STEP 3: Deduplication & Clustering =====")
    print(f"[dedup_cluster] Date: {date_iso}")
    print(f"[dedup_cluster] Input: {input_raw_items_path}")
    
    # Check if input file exists
    if not os.path.exists(input_raw_items_path):
        print(f"[dedup_cluster] ERROR: {input_raw_items_path} not found")
        return
    
    start_time = time.time()
    timings = {}
    
    # 3.0 Load & validate input
    t0 = time.time()
    items = _load_and_validate_input(input_raw_items_path)
    input_items_count = len(items)
    print(f"[dedup_cluster] Loaded {input_items_count} valid items")
    
    if input_items_count == 0:
        # Empty input, write empty clusters and return
        empty_output = {
            "date": date_iso,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "near_dup_threshold": settings.NEAR_DUP_THRESHOLD,
            "num_clusters": 0,
            "clusters": []
        }
        
        clusters_path = os.path.join(output_dir_for_day, "clusters.json")
        with open(clusters_path, "w", encoding="utf-8") as f:
            json.dump(empty_output, f, indent=2, ensure_ascii=False)
        
        print(f"[dedup_cluster] No items to cluster")
        return
    
    # 3.1 Normalize
    items = _normalize_items(items, settings)
    
    # 3.2 Exact dedup
    t1 = time.time()
    items, original_count = _exact_dedup(items, settings)
    after_exact_dedup = len(items)
    timings["exact_dedup"] = time.time() - t1
    print(f"[dedup_cluster] After exact dedup: {after_exact_dedup} items")
    
    # Optional: write deduped_items.json with metadata
    deduped_path = os.path.join(output_dir_for_day, "deduped_items.json")
    deduped_output = {
        "count": len(items),
        "original_count": input_items_count,
        "duplicates_removed": input_items_count - len(items),
        "deduped_at": datetime.now().isoformat(),
        "articles": items
    }
    with open(deduped_path, "w", encoding="utf-8") as f:
        json.dump(deduped_output, f, indent=2, ensure_ascii=False)
    
    # 3.3 Embeddings
    t2 = time.time()
    embeddings = _generate_embeddings(items, settings)
    timings["embedding"] = time.time() - t2
    
    # 3.4 Near-duplicate pruning
    t3 = time.time()
    items, embeddings, pre_near_count = _near_dedup(items, embeddings, settings)
    after_near_dedup = len(items)
    timings["near_dedup"] = time.time() - t3
    print(f"[dedup_cluster] After near-dup: {after_near_dedup} items")
    
    # 3.5 Dynamic Clustering
    t4 = time.time()
    cluster_labels, num_clusters, clustering_stats = _cluster_dynamic(items, embeddings, settings)
    timings["clustering"] = time.time() - t4
    
    avg_items_per_cluster = after_near_dedup / num_clusters if num_clusters > 0 else 0
    print(f"[dedup_cluster] Clusters: {num_clusters}, Avg items/cluster: {avg_items_per_cluster:.2f}")
    
    # 3.6 LLM Cluster Refinement & Labeling
    refinement_stats = {}
    
    if settings.ENABLE_LLM_CLUSTER_REFINEMENT and settings.OPENAI_API_KEY:
        # Use LLM to refine clusters, filter articles, AND generate labels in one call
        from agents.cluster_refiner import refine_clusters_with_llm
        
        t5 = time.time()
        items, embeddings, cluster_labels, label_map, refinement_stats = refine_clusters_with_llm(
            items, embeddings, cluster_labels, settings
        )
        timings["refinement_and_labeling"] = time.time() - t5
        
        # Update counts after refinement and filtering
        after_near_dedup = len(items)  # Update with filtered count
        num_clusters = len(set(cluster_labels))
        avg_items_per_cluster = after_near_dedup / num_clusters if num_clusters > 0 else 0
        
        print(f"[dedup_cluster] After LLM filtering: {after_near_dedup} articles in {num_clusters} clusters")
        
    else:
        # Original flow: just label the algorithmic clusters
        t5 = time.time()
        slates = _prepare_label_inputs(items, embeddings, cluster_labels, settings)
        timings["compression"] = time.time() - t5
        
        # Write label inputs for traceability
        tmp_dir = os.path.join(output_dir_for_day, settings.TMP_SUBDIR)
        os.makedirs(tmp_dir, exist_ok=True)
        label_inputs_path = os.path.join(tmp_dir, "label_inputs.json")
        with open(label_inputs_path, "w", encoding="utf-8") as f:
            json.dump(slates, f, indent=2, ensure_ascii=False)
        
        t6 = time.time()
        label_result = _generate_labels_with_llm(slates, settings)
        timings["labeling"] = time.time() - t6
        
        # Build label map
        label_map = {item["cluster_id"]: item["label"] for item in label_result["labels"]}
        
        refinement_stats = {"enabled": False}
    
    # 3.7 Assemble clusters.json
    clusters_output = _assemble_clusters_output(
        items, cluster_labels, label_map, date_iso, settings,
        original_count=input_items_count,
        after_exact_dedup=after_exact_dedup
    )
    
    # Write clusters.json
    clusters_path = os.path.join(output_dir_for_day, "clusters.json")
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(clusters_output, f, indent=2, ensure_ascii=False)
    
    print(f"[dedup_cluster] Clusters written to {clusters_path}")
    
    # 3.8 Metrics
    total_time = time.time() - start_time
    
    # Calculate average keyphrases (only available in non-refinement mode)
    avg_keyphrases = 0
    if not settings.ENABLE_LLM_CLUSTER_REFINEMENT:
        avg_keyphrases = sum(len(s["keyphrases"]) for s in slates) / len(slates) if slates else 0
    
    # Build cluster size summary
    cluster_sizes = defaultdict(int)
    for label in cluster_labels:
        cluster_sizes[label] += 1
    
    clusters_summary = [
        {"cluster_id": cid, "size": size}
        for cid, size in sorted(cluster_sizes.items())
    ]
    
    metrics = {
        "date": date_iso,
        "input_items": input_items_count,
        "after_exact_dedup": after_exact_dedup,
        "after_near_dedup": after_near_dedup,
        "clusters_count": num_clusters,
        "avg_items_per_cluster": round(avg_items_per_cluster, 2),
        "clustering": clustering_stats,
        "clusters_summary": clusters_summary,
        "refinement": refinement_stats,
        "labeling": {
            "mode": "B" if not settings.ENABLE_LLM_CLUSTER_REFINEMENT else "LLM-Refined",
            "top_k_for_labeling": settings.TOP_K_FOR_LABELING,
            "max_abstract_chars": settings.MAX_ABSTRACT_CHARS,
            "keyphrases_per_cluster_avg": round(avg_keyphrases, 2),
            "llm_provider": "OpenAI",
            "model": settings.OPENAI_MODEL,
            "temperature": settings.TEMPERATURE,
            "duplicates_fixed_post_label": 0
        },
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "total_time_sec": round(total_time, 2)
    }
    
    _write_metrics(metrics, output_dir_for_day)
    
    print(f"[dedup_cluster] ===== STEP 3 COMPLETE ({total_time:.1f}s) =====")

