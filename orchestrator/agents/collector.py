import json
import os
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from config import settings


def _read_allowlist(feeds_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(feeds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    defaults = data.get("defaults", {})
    feeds = data.get("feeds", [])
    return defaults, feeds


def _http_get(url: str, timeout_s: int, max_retries: int, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout_s, headers=headers)
            if 200 <= resp.status_code < 300:
                return resp
            # Retry on transient server errors
            if resp.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            # Do not retry on 4xx
            return None
        except requests.RequestException:
            time.sleep(backoff)
            backoff *= 2
    return None


def _parse_feed(content: bytes) -> feedparser.FeedParserDict:
    return feedparser.parse(content)


def _clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Prefer article tag text if present
    article = soup.find("article")
    text = article.get_text(separator="\n", strip=True) if article else soup.get_text(separator="\n", strip=True)
    # Collapse excessive newlines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _hash_url(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _parse_published(entry: Dict[str, Any]) -> datetime:
    # Try common fields in order
    for key in ("published", "updated", "created"):
        val = entry.get(key)
        if val:
            try:
                dt = dateparser.parse(str(val))
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    # Fallback to now
    return datetime.now(timezone.utc)

def _is_recent_article(published_dt: datetime, cutoff_days: int = 3) -> bool:
    """Check if article is recent (within last N days)"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
    return published_dt >= cutoff


def _fetch_article_body(url: str) -> Optional[str]:
    headers = {"User-Agent": settings.USER_AGENT}
    resp = _http_get(url, timeout_s=settings.REQUEST_TIMEOUT, max_retries=settings.MAX_RETRIES, headers=headers)
    if not resp:
        return None
    return _clean_html_to_text(resp.text)


def _ensure_dirs():
    os.makedirs(settings.OUTPUT_TODAY_DIR, exist_ok=True)
    os.makedirs(settings.CACHE_TODAY_DIR, exist_ok=True)


def _load_existing_items() -> List[Dict[str, Any]]:
    """Load existing items from file if it exists"""
    out_path = os.path.join(settings.OUTPUT_TODAY_DIR, "raw_items.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both old format (list) and new format (object with articles)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "articles" in data:
                    return data["articles"]
        except Exception:
            pass
    return []

def _save_items(items: List[Dict[str, Any]]) -> None:
    """Save items to JSON file with count metadata"""
    out_path = os.path.join(settings.OUTPUT_TODAY_DIR, "raw_items.json")
    
    # Create output with metadata
    output_data = {
        "count": len(items),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "articles": items
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

def collect_all() -> str:
    _ensure_dirs()
    start_time = time.time()
    max_total_time = 300  # 5 minutes max

    print(f"[collector] Starting collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[collector] Target: collect recent articles (last 3 days)")
    print(f"[collector] Max time limit: {max_total_time/60:.1f} minutes")

    defaults, feeds = _read_allowlist(settings.FEEDS_FILE)
    ua = defaults.get("user_agent", settings.USER_AGENT)
    headers = {"User-Agent": ua}

    # Load existing items (for incremental saving)
    items = _load_existing_items()
    print(f"[collector] Loaded {len(items)} existing items")

    total_feeds = len(feeds)
    processed_feeds = 0
    recent_articles_collected = 0

    for feed_idx, feed in enumerate(feeds, 1):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_total_time:
            print(f"[collector] Time limit reached ({elapsed:.1f}s), stopping")
            break

        feed_url = feed.get("url")
        if not feed_url:
            continue

        source_label = feed.get("label") or feed_url
        topics = feed.get("topics", [])
        
        print(f"[collector] [{feed_idx}/{total_feeds}] Processing: {source_label}")
        
        resp = _http_get(feed_url, timeout_s=settings.REQUEST_TIMEOUT, max_retries=settings.MAX_RETRIES, headers=headers)
        if not resp:
            print(f"[collector] Failed to fetch feed: {source_label}")
            processed_feeds += 1
            continue

        parsed = _parse_feed(resp.content)
        entries = parsed.entries if hasattr(parsed, "entries") else []

        # Filter for recent articles only
        recent_entries = []
        for entry in entries:
            try:
                published_dt = _parse_published(entry)
                if _is_recent_article(published_dt, cutoff_days=3):
                    recent_entries.append(entry)
            except Exception:
                continue
        
        if not recent_entries:
            print(f"[collector] No recent articles from {source_label}")
            processed_feeds += 1
            continue

        # Process recent articles
        feed_items_added = 0
        for entry in recent_entries:
            try:
                url = entry.get("link") or entry.get("id") or ""
                title = (entry.get("title") or "").strip()
                if not url or not title:
                    continue

                published_dt = _parse_published(entry)
                url_hash = _hash_url(url)
                
                # Check if already collected
                if any(item.get("item_id") == url_hash for item in items):
                    continue
                
                # Use feed summary for now (much faster)
                summary = entry.get("summary", "") or entry.get("description", "")
                if summary:
                    full_text = _clean_html_to_text(summary)
                else:
                    full_text = _fetch_article_body(url)
                    if not full_text:
                        continue

                # Cache body by date/hash
                cache_path = os.path.join(settings.CACHE_TODAY_DIR, f"{url_hash}.json")
                with open(cache_path, "w", encoding="utf-8") as cf:
                    json.dump({
                        "url": url,
                        "title": title,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "content": full_text,
                    }, cf, ensure_ascii=False)

                item = {
                    "item_id": url_hash,
                    "source": source_label,
                    "url": url,
                    "title": title,
                    "published_at": published_dt.isoformat(),
                    "topics": topics,
                    "content": full_text,
                    "cached_path": cache_path.replace("\\", "/"),
                }
                items.append(item)
                feed_items_added += 1
                recent_articles_collected += 1
                
            except Exception as e:
                continue

        # Save incrementally after each feed
        if feed_items_added > 0:
            _save_items(items)
            print(f"[collector] Collected {feed_items_added} articles from {source_label}")
        else:
            print(f"[collector] No new articles from {source_label}")
        
        processed_feeds += 1

    # Final summary
    elapsed = time.time() - start_time
    non_empty = [it for it in items if it.get("title") and it.get("url") and it.get("content")]
    
    print(f"[collector] ===== COLLECTION COMPLETE =====")
    print(f"[collector] Time taken: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"[collector] Feeds processed: {processed_feeds}/{total_feeds}")
    print(f"[collector] Total articles collected: {len(non_empty)}")
    
    if len(non_empty) < settings.MIN_ITEMS:
        print(f"[collector] Warning: only {len(non_empty)} items (< {settings.MIN_ITEMS})")
    else:
        print(f"[collector] Success: {len(non_empty)} items collected")

    out_path = os.path.join(settings.OUTPUT_TODAY_DIR, "raw_items.json")
    return out_path


