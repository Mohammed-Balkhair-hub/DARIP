"""
Step 3.5: Full-Text Enrichment Agent

LangGraph workflow that enriches clusters.json with full article text.

Workflow:
N0: LoadInput → N1: PickNext → N2: RobotsCheck → N3: FetchHTML → 
N4: VariantDiscovery → N5: ExtractReadable → N6: QualityGate → 
N7: AdvanceCursor → (loop to N1) → N8: WriteOutput → N9: Done
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from config import settings
from extractors import RobotsChecker, HTMLFetcher, TextExtractor

logger = logging.getLogger(__name__)


class EnricherState(TypedDict):
    """LangGraph state for full-text enrichment."""
    run_date: str
    clusters: Dict[str, Any]  # Loaded clusters.json
    worklist: List[Dict[str, Any]]  # Flattened items with coordinates
    cursor: Dict[str, int]  # {next: int, total: int}
    stats: Dict[str, Any]  # Stats and failure tracking
    config: Dict[str, Any]  # Enricher config
    
    # Shared resources (not serialized to JSON)
    robots_checker: Optional[RobotsChecker]
    html_fetcher: Optional[HTMLFetcher]
    text_extractor: Optional[TextExtractor]
    
    # Current item being processed
    current_item: Optional[Dict[str, Any]]
    current_cluster_id: Optional[int]
    current_item_index: Optional[int]
    
    # Fetch results for current item
    html_bytes: Optional[bytes]
    status_code: Optional[int]
    final_url: Optional[str]
    paywall_detected: bool


def node_0_load_input(state: EnricherState) -> EnricherState:
    """
    N0: LoadInput
    
    Read raw_items.json, build worklist, initialize state.
    """
    logger.info("[enricher N0] Loading raw_items.json")
    
    run_date = state['run_date']
    raw_items_path = os.path.join(settings.OUTPUT_DIR, run_date, 'raw_items.json')
    
    # Read raw_items.json
    with open(raw_items_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Build worklist from articles array
    worklist = []
    articles = raw_data.get('articles', [])
    
    for item_index, item in enumerate(articles):
        # Extract domain from URL
        from urllib.parse import urlparse
        import tldextract
        
        url = item.get('url', '')
        try:
            extracted = tldextract.extract(url)
            source_domain = f"{extracted.domain}.{extracted.suffix}"
        except:
            source_domain = 'unknown'
        
        worklist.append({
            'item_index': item_index,
            'url': url,
            'title': item.get('title', ''),
            'source_domain': source_domain,
            'language': 'en',  # Assume English for now
            'abstract': item.get('content', ''),  # Use content field as abstract
            'has_fulltext': item.get('has_fulltext', False),
            'item_id': item.get('item_id', '')
        })
    
    # Initialize stats
    stats = {
        'attempted': 0,
        'succeeded': 0,
        'failed': 0,
        'skipped': 0,
        'failed_by_reason': {}
    }
    
    # Initialize shared extractors
    robots_checker = RobotsChecker(
        user_agent=state['config']['user_agent'],
        timeout=10
    )
    
    html_fetcher = HTMLFetcher(
        user_agent=state['config']['user_agent'],
        timeout_sec=state['config']['timeout_sec'],
        max_retries=state['config']['max_retries'],
        per_domain_qps=state['config']['per_domain_qps'],
        per_domain_concurrency=state['config']['per_domain_concurrency'],
        follow_redirects=state['config']['follow_redirects'],
        respect_paywalls=state['config']['respect_paywalls']
    )
    
    text_extractor = TextExtractor(
        extractor_sequence=state['config']['extractor_sequence'],
        min_accept_words=state['config']['min_accept_words'],
        max_chars=state['config']['max_full_text_chars'],
        allow_amp_variants=state['config']['allow_amp_print_variants']
    )
    
    logger.info(f"[enricher N0] Built worklist with {len(worklist)} items from raw_items.json")
    
    state['clusters'] = raw_data  # Store raw_data structure
    state['worklist'] = worklist
    state['cursor'] = {'next': 0, 'total': len(worklist)}
    state['stats'] = stats
    state['robots_checker'] = robots_checker
    state['html_fetcher'] = html_fetcher
    state['text_extractor'] = text_extractor
    
    return state


def node_1_pick_next(state: EnricherState) -> EnricherState:
    """
    N1: PickNext
    
    Pick next item from worklist, or signal completion.
    Skip items already enriched if not FORCE mode.
    """
    cursor = state['cursor']
    
    # Check if done
    if cursor['next'] >= cursor['total']:
        logger.info(f"[enricher N1] All items processed ({cursor['total']} total)")
        state['current_item'] = None
        return state
    
    # Get next item
    current = state['worklist'][cursor['next']]
    
    # Check if already enriched and not FORCE mode
    if current['has_fulltext'] and not state['config']['force']:
        logger.debug(f"[enricher N1] Skipping already enriched: {current['url']}")
        state['stats']['skipped'] += 1
        state['cursor']['next'] += 1
        state['current_item'] = None  # Signal to loop back
        return state
    
    # Process this item
    logger.info(
        f"[enricher N1] Processing [{cursor['next']+1}/{cursor['total']}]: {current['url']}"
    )
    
    state['current_item'] = current
    state['current_cluster_id'] = None  # Not used for raw_items
    state['current_item_index'] = current['item_index']
    state['stats']['attempted'] += 1
    
    return state


def node_2_robots_check(state: EnricherState) -> EnricherState:
    """
    N2: RobotsCheck
    
    Check if URL is allowed by robots.txt.
    """
    current = state['current_item']
    url = current['url']
    
    robots_checker = state['robots_checker']
    allowed = robots_checker.is_allowed(url)
    
    if not allowed:
        logger.info(f"[enricher N2] Robots.txt disallows: {url}")
        _record_failure(state, 'robots_disallow')
        state['current_item'] = None  # Signal to skip to N7
    
    return state


def node_3_fetch_html(state: EnricherState) -> EnricherState:
    """
    N3: FetchHTML
    
    Fetch HTML content with rate limiting and retries.
    """
    current = state['current_item']
    url = current['url']
    
    html_fetcher = state['html_fetcher']
    
    # Run async fetch
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, use run_until_complete
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status_code, final_url, html_bytes, paywall_detected = loop.run_until_complete(
            html_fetcher.fetch(url)
        )
    else:
        # Event loop is running, this shouldn't happen in normal flow
        # Fall back to creating async task
        logger.warning("[enricher N3] Event loop already running, using workaround")
        loop = asyncio.new_event_loop()
        status_code, final_url, html_bytes, paywall_detected = loop.run_until_complete(
            html_fetcher.fetch(url)
        )
    
    state['status_code'] = status_code
    state['final_url'] = final_url
    state['html_bytes'] = html_bytes
    state['paywall_detected'] = paywall_detected
    
    # Check for failures
    if paywall_detected:
        logger.info(f"[enricher N3] Paywall detected: {url}")
        _record_failure(state, 'paywall')
        state['current_item'] = None
        return state
    
    if status_code >= 400 and status_code < 500:
        logger.warning(f"[enricher N3] HTTP {status_code} for {url}")
        _record_failure(state, 'http_4xx')
        state['current_item'] = None
        return state
    
    if status_code >= 500:
        logger.warning(f"[enricher N3] HTTP {status_code} for {url}")
        _record_failure(state, 'http_5xx')
        state['current_item'] = None
        return state
    
    if status_code == 0 or status_code == 408:
        logger.warning(f"[enricher N3] Timeout/connection error for {url}")
        _record_failure(state, 'timeout')
        state['current_item'] = None
        return state
    
    if len(html_bytes) == 0:
        logger.warning(f"[enricher N3] Empty response for {url}")
        _record_failure(state, 'empty_response')
        state['current_item'] = None
        return state
    
    logger.debug(f"[enricher N3] Fetched {len(html_bytes)} bytes from {url}")
    
    return state


def node_4_variant_discovery(state: EnricherState) -> EnricherState:
    """
    N4: VariantDiscovery
    
    Check for AMP/print variants (optional optimization).
    For now, we skip this and use the original HTML.
    """
    # Optional: Could implement variant refetching here
    # For simplicity, we proceed with the HTML we have
    logger.debug("[enricher N4] Variant discovery (skipped)")
    return state


def node_5_extract_readable(state: EnricherState) -> EnricherState:
    """
    N5: ExtractReadable
    
    Extract clean text using extractor sequence.
    """
    html_bytes = state['html_bytes']
    url = state['current_item']['url']
    
    text_extractor = state['text_extractor']
    result = text_extractor.extract(html_bytes, url)
    
    if result is None:
        logger.warning(f"[enricher N5] Extraction failed for {url}")
        _record_failure(state, 'empty_extract')
        state['current_item'] = None
        return state
    
    # Store extraction result in current_item for next node
    state['current_item']['extraction_result'] = result
    
    logger.info(
        f"[enricher N5] Extracted {result['word_count']} words "
        f"using {result['extraction_method']} from {url}"
    )
    
    return state


def node_6_quality_gate(state: EnricherState) -> EnricherState:
    """
    N6: QualityGate
    
    Validate language and content quality.
    """
    current = state['current_item']
    result = current['extraction_result']
    
    # Check language allowlist
    lang_allowlist = state['config']['lang_allowlist']
    detected_lang = result['language_detected']
    
    if lang_allowlist and detected_lang not in lang_allowlist:
        logger.info(
            f"[enricher N6] Language mismatch: {detected_lang} not in {lang_allowlist} "
            f"for {current['url']}"
        )
        _record_failure(state, 'lang_mismatch')
        state['current_item'] = None
        return state
    
    # Check content quality: absolute minimum OR 2x abstract length
    abstract = current.get('abstract', '')
    full_text = result['clean_text']
    min_absolute_chars = state['config']['min_full_text_chars']
    
    # Accept if EITHER condition is met:
    # 1. Has at least min_full_text_chars (configurable, default 1000)
    # 2. OR is at least 2x the abstract length (handles short abstracts)
    if len(full_text) < min_absolute_chars and len(full_text) < len(abstract) * 2:
        logger.warning(
            f"[enricher N6] Full text too short ({len(full_text)} chars, "
            f"min {min_absolute_chars} OR 2x abstract {len(abstract)} chars) for {current['url']}"
        )
        _record_failure(state, 'too_short')
        state['current_item'] = None
        return state
    
    # Success! Mark for writing
    current['full_text'] = full_text
    current['has_fulltext'] = True
    current['extraction_method'] = result['extraction_method']
    current['full_text_words'] = result['word_count']
    
    state['stats']['succeeded'] += 1
    
    logger.info(f"[enricher N6] Quality gate passed for {current['url']}")
    
    return state


def node_7_advance_cursor(state: EnricherState) -> EnricherState:
    """
    N7: AdvanceCursor
    
    Write item fields to in-memory raw_items JSON and advance cursor.
    """
    current = state['current_item']
    
    if current is not None:
        # Write fields to raw_items JSON (articles array)
        item_index = current['item_index']
        item = state['clusters']['articles'][item_index]
        
        # Update item with new fields
        item['has_fulltext'] = current.get('has_fulltext', False)
        item['full_text'] = current.get('full_text', '')
        
        # Optional metadata
        if 'extraction_method' in current:
            item['extraction_method'] = current['extraction_method']
        if 'full_text_words' in current:
            item['full_text_words'] = current['full_text_words']
        if 'failure_reason' in current:
            item['failure_reason'] = current['failure_reason']
    
    # Advance cursor
    state['cursor']['next'] += 1
    
    # Clear current item for next iteration
    state['current_item'] = None
    state['html_bytes'] = None
    
    return state


def node_8_write_output(state: EnricherState) -> EnricherState:
    """
    N8: WriteOutput
    
    Atomically write enriched raw_items.json.
    """
    logger.info("[enricher N8] Writing enriched raw_items.json")
    
    # Compute final stats
    stats = state['stats']
    stats['failed'] = stats['attempted'] - stats['succeeded']
    
    # Add enrichment_stats to raw_items JSON
    state['clusters']['enrichment_stats'] = {
        'attempted': stats['attempted'],
        'succeeded': stats['succeeded'],
        'failed': stats['failed'],
        'skipped': stats['skipped'],
        'failed_by_reason': stats['failed_by_reason'],
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Write to temp file, then atomic rename
    run_date = state['run_date']
    output_dir = os.path.join(settings.OUTPUT_DIR, run_date)
    tmp_path = os.path.join(output_dir, 'raw_items.tmp.json')
    final_path = os.path.join(output_dir, 'raw_items.json')
    
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(state['clusters'], f, indent=2, ensure_ascii=False)
    
    # Atomic rename
    os.replace(tmp_path, final_path)
    
    logger.info(f"[enricher N8] Wrote enriched raw_items to {final_path}")
    logger.info(
        f"[enricher N8] Stats: {stats['succeeded']}/{stats['attempted']} succeeded, "
        f"{stats['failed']} failed, {stats['skipped']} skipped"
    )
    
    return state


def node_9_done(state: EnricherState) -> EnricherState:
    """
    N9: Done
    
    Final node, return summary.
    """
    logger.info("[enricher N9] Enrichment complete")
    return state


def _record_failure(state: EnricherState, reason: str) -> None:
    """
    Record a failure in stats and current item.
    
    Args:
        state: Current state
        reason: Failure reason code
    """
    state['stats']['failed_by_reason'][reason] = \
        state['stats']['failed_by_reason'].get(reason, 0) + 1
    
    if state['current_item']:
        state['current_item']['failure_reason'] = reason


def route_from_pick_next(state: EnricherState) -> str:
    """Route from N1: PickNext."""
    if state['current_item'] is None:
        # Either done or skipped
        if state['cursor']['next'] >= state['cursor']['total']:
            return 'write_output'
        else:
            return 'pick_next'  # Loop for skipped items
    else:
        return 'robots_check'


def route_from_robots_check(state: EnricherState) -> str:
    """Route from N2: RobotsCheck."""
    if state['current_item'] is None:
        return 'advance_cursor'  # Failed robots check
    else:
        return 'fetch_html'


def route_from_fetch_html(state: EnricherState) -> str:
    """Route from N3: FetchHTML."""
    if state['current_item'] is None:
        return 'advance_cursor'  # Failed fetch
    else:
        return 'variant_discovery'


def route_from_extract_readable(state: EnricherState) -> str:
    """Route from N5: ExtractReadable."""
    if state['current_item'] is None:
        return 'advance_cursor'  # Failed extraction
    else:
        return 'quality_gate'


def route_from_quality_gate(state: EnricherState) -> str:
    """Route from N6: QualityGate."""
    # Always advance cursor (success or failure was already recorded)
    return 'advance_cursor'


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(EnricherState)
    
    # Add nodes
    workflow.add_node("load_input", node_0_load_input)
    workflow.add_node("pick_next", node_1_pick_next)
    workflow.add_node("robots_check", node_2_robots_check)
    workflow.add_node("fetch_html", node_3_fetch_html)
    workflow.add_node("variant_discovery", node_4_variant_discovery)
    workflow.add_node("extract_readable", node_5_extract_readable)
    workflow.add_node("quality_gate", node_6_quality_gate)
    workflow.add_node("advance_cursor", node_7_advance_cursor)
    workflow.add_node("write_output", node_8_write_output)
    workflow.add_node("done", node_9_done)
    
    # Set entry point
    workflow.set_entry_point("load_input")
    
    # Add edges
    workflow.add_edge("load_input", "pick_next")
    workflow.add_conditional_edges("pick_next", route_from_pick_next)
    workflow.add_conditional_edges("robots_check", route_from_robots_check)
    workflow.add_conditional_edges("fetch_html", route_from_fetch_html)
    workflow.add_edge("variant_discovery", "extract_readable")
    workflow.add_conditional_edges("extract_readable", route_from_extract_readable)
    workflow.add_conditional_edges("quality_gate", route_from_quality_gate)
    workflow.add_edge("advance_cursor", "pick_next")  # Loop back
    workflow.add_edge("write_output", "done")
    workflow.add_edge("done", END)
    
    return workflow.compile()


def enrich_fulltext(date: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Main entry point for full-text enrichment.
    
    Args:
        date: Date string (YYYY-MM-DD), defaults to today
        force: If True, re-fetch already enriched items
        
    Returns:
        Dict with enrichment stats
    """
    if date is None:
        date = settings.TODAY
    
    logger.info(f"[enricher] ===== STEP 3.5: Full-Text Enrichment =====")
    logger.info(f"[enricher] Date: {date}")
    logger.info(f"[enricher] Force mode: {force}")
    
    start_time = time.time()
    
    # Build config from settings
    config = {
        'user_agent': settings.ENRICHER_USER_AGENT,
        'timeout_sec': settings.ENRICHER_HTTP_TIMEOUT_SEC,
        'max_retries': settings.ENRICHER_HTTP_MAX_RETRIES,
        'per_domain_qps': settings.ENRICHER_PER_DOMAIN_QPS,
        'per_domain_concurrency': settings.ENRICHER_PER_DOMAIN_CONCURRENCY,
        'follow_redirects': settings.ENRICHER_FOLLOW_REDIRECTS,
        'respect_paywalls': settings.ENRICHER_RESPECT_PAYWALLS,
        'extractor_sequence': settings.ENRICHER_EXTRACTOR_SEQUENCE,
        'min_accept_words': settings.ENRICHER_MIN_ACCEPT_WORDS,
        'max_full_text_chars': settings.ENRICHER_MAX_FULL_TEXT_CHARS,
        'min_full_text_chars': settings.ENRICHER_MIN_FULL_TEXT_CHARS,
        'allow_amp_print_variants': settings.ENRICHER_ALLOW_AMP_PRINT_VARIANTS,
        'lang_allowlist': settings.ENRICHER_LANG_ALLOWLIST,
        'force': force or settings.ENRICHER_FORCE
    }
    
    # Initialize state
    initial_state = EnricherState(
        run_date=date,
        clusters={},
        worklist=[],
        cursor={'next': 0, 'total': 0},
        stats={},
        config=config,
        robots_checker=None,
        html_fetcher=None,
        text_extractor=None,
        current_item=None,
        current_cluster_id=None,
        current_item_index=None,
        html_bytes=None,
        status_code=None,
        final_url=None,
        paywall_detected=False
    )
    
    # Build and run workflow
    app = build_workflow()
    
    # Set higher recursion limit for large worklists
    # Each item requires ~7 node traversals, so for 500 items we need ~3500 limit
    # Setting to 5000 to handle up to ~700 items safely
    config = {"recursion_limit": 5000}
    
    final_state = app.invoke(initial_state, config=config)
    
    elapsed = time.time() - start_time
    
    logger.info(f"[enricher] ===== ENRICHMENT COMPLETE ({elapsed:.1f}s) =====")
    
    return {
        'date': date,
        'stats': final_state['stats'],
        'elapsed_sec': round(elapsed, 2)
    }


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    result = enrich_fulltext()
    print(json.dumps(result, indent=2))

