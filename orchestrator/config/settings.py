import os
from datetime import datetime

# Feed and data paths
FEEDS_FILE = os.getenv('FEEDS_FILE', '/data/feeds/allowlist_feeds.json')
CACHE_DIR = os.getenv('CACHE_DIR', '/data/cache')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/data/outputs')

# Collection settings
MIN_ITEMS = int(os.getenv('MIN_ITEMS', '10'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '15'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))

# User agent from feed defaults or custom
USER_AGENT = os.getenv('USER_AGENT', 'Mozilla/5.0 (compatible; DaripBot/1.0; +https://example.com/bot)')

# TTS settings
USE_FAKE_TTS = os.getenv('USE_FAKE_TTS', '1') == '1'
VOX_A = os.getenv('VOX_A', 'en-GB-Neural2-D')
VOX_B = os.getenv('VOX_B', 'en-GB-Neural2-B')

# Current date for output directory
TODAY = datetime.now().strftime('%Y-%m-%d')
OUTPUT_TODAY_DIR = os.path.join(OUTPUT_DIR, TODAY)
CACHE_TODAY_DIR = os.path.join(CACHE_DIR, TODAY)

# Ensure directories exist
os.makedirs(OUTPUT_TODAY_DIR, exist_ok=True)
os.makedirs(CACHE_TODAY_DIR, exist_ok=True)


# OpenAI settings (used by script writer for headline extraction, sequencing, and dialogue generation)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '128'))  # Default for headliner; nodes override as needed

# Paths
LOG_SUBDIR = "_logs"
TMP_SUBDIR = "_tmp"

# Ensure log and tmp directories exist
LOG_TODAY_DIR = os.path.join(OUTPUT_TODAY_DIR, LOG_SUBDIR)
TMP_TODAY_DIR = os.path.join(OUTPUT_TODAY_DIR, TMP_SUBDIR)
os.makedirs(LOG_TODAY_DIR, exist_ok=True)
os.makedirs(TMP_TODAY_DIR, exist_ok=True)

# ===== STEP 4: Podcast Script Generation =====
PODCAST_SCRIPT_SUBDIR = "podcast_script"

# Script Generation Settings (Node 3: DuoScript)
SCRIPT_SPEAKER_NAMES = os.getenv('SCRIPT_SPEAKER_NAMES', 'Adam,Sara').split(',')
SCRIPT_TARGET_TOTAL_SECS = int(os.getenv('SCRIPT_TARGET_TOTAL_SECS', '480'))  # 8 minutes default
SCRIPT_MAX_LINES_PER_ITEM = int(os.getenv('SCRIPT_MAX_LINES_PER_ITEM', '8'))
SCRIPT_MAX_WORDS_PER_LINE = int(os.getenv('SCRIPT_MAX_WORDS_PER_LINE', '100'))
SCRIPT_DEFAULT_PAUSE_MS = int(os.getenv('SCRIPT_DEFAULT_PAUSE_MS', '120'))
SCRIPT_SEGMENT_PREFIX = os.getenv('SCRIPT_SEGMENT_PREFIX', 'SEG_')
SCRIPT_ALLOW_CONSECUTIVE_SPEAKER = os.getenv('SCRIPT_ALLOW_CONSECUTIVE_SPEAKER', '1') == '1'
SCRIPT_DIGITS_TO_SPEECH = os.getenv('SCRIPT_DIGITS_TO_SPEECH', '1') == '1'
SCRIPT_SPELL_ACRONYMS = os.getenv('SCRIPT_SPELL_ACRONYMS', '1') == '1'
SCRIPT_WPM_ESTIMATE = int(os.getenv('SCRIPT_WPM_ESTIMATE', '150'))  # Words per minute for duration estimation

# Polish Settings (Node 4: Naturalizer)
POLISH_PACE = os.getenv('POLISH_PACE', 'medium')  # brisk, medium, relaxed
POLISH_TONE = os.getenv('POLISH_TONE', 'warm, clear, no slang')
POLISH_ENABLE_MICRO_TRANSITIONS = os.getenv('POLISH_ENABLE_MICRO_TRANSITIONS', '1') == '1'
POLISH_ENABLE_SSML = os.getenv('POLISH_ENABLE_SSML', '0') == '1'
POLISH_LOCK_SEGMENT_BOUNDARIES = os.getenv('POLISH_LOCK_SEGMENT_BOUNDARIES', '1') == '1'
POLISH_ALLOW_MERGE_LINES = os.getenv('POLISH_ALLOW_MERGE_LINES', '1') == '1'
POLISH_ALLOW_SPLIT_LINES = os.getenv('POLISH_ALLOW_SPLIT_LINES', '1') == '1'
POLISH_ENFORCE_BALANCE = os.getenv('POLISH_ENFORCE_BALANCE', '1') == '1'
POLISH_BALANCE_TOLERANCE_PCT = float(os.getenv('POLISH_BALANCE_TOLERANCE_PCT', '10'))  # ±10%

# ===== STEP 3.5: Full-Text Enrichment =====

# Concurrency & Rate Limiting
ENRICHER_GLOBAL_CONCURRENCY = int(os.getenv('ENRICHER_GLOBAL_CONCURRENCY', '4'))
ENRICHER_PER_DOMAIN_CONCURRENCY = int(os.getenv('ENRICHER_PER_DOMAIN_CONCURRENCY', '1'))
ENRICHER_PER_DOMAIN_QPS = float(os.getenv('ENRICHER_PER_DOMAIN_QPS', '0.5'))

# HTTP Settings
ENRICHER_HTTP_TIMEOUT_SEC = int(os.getenv('ENRICHER_HTTP_TIMEOUT_SEC', '20'))
ENRICHER_HTTP_MAX_RETRIES = int(os.getenv('ENRICHER_HTTP_MAX_RETRIES', '3'))
ENRICHER_USER_AGENT = os.getenv('ENRICHER_USER_AGENT', 'Mozilla/5.0 (compatible; DaripBot/1.0; +https://example.com/bot)')
ENRICHER_FOLLOW_REDIRECTS = os.getenv('ENRICHER_FOLLOW_REDIRECTS', '1') == '1'
ENRICHER_PREFER_IPV4 = os.getenv('ENRICHER_PREFER_IPV4', '1') == '1'

# Extraction Settings
ENRICHER_EXTRACTOR_SEQUENCE = os.getenv('ENRICHER_EXTRACTOR_SEQUENCE', 'trafilatura,readability,newspaper3k,goose3').split(',')
ENRICHER_MIN_ACCEPT_WORDS = int(os.getenv('ENRICHER_MIN_ACCEPT_WORDS', '200'))
ENRICHER_ALLOW_AMP_PRINT_VARIANTS = os.getenv('ENRICHER_ALLOW_AMP_PRINT_VARIANTS', '1') == '1'
ENRICHER_MAX_FULL_TEXT_CHARS = int(os.getenv('ENRICHER_MAX_FULL_TEXT_CHARS', '50000'))

# Quality Gate Settings
ENRICHER_MIN_FULL_TEXT_CHARS = int(os.getenv('ENRICHER_MIN_FULL_TEXT_CHARS', '1000'))  # Absolute minimum for quality gate

# Politeness & Safety
ENRICHER_RESPECT_PAYWALLS = os.getenv('ENRICHER_RESPECT_PAYWALLS', '1') == '1'
ENRICHER_LANG_ALLOWLIST = os.getenv('ENRICHER_LANG_ALLOWLIST', 'en').split(',')

# Behavior
ENRICHER_FORCE = os.getenv('ENRICHER_FORCE', '0') == '1'  # Re-fetch already enriched items

# ===== STEP 3: RAG Query Retrieval =====

# Embedding Model
RAG_EMBEDDING_MODEL = os.getenv('RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# Chunking Settings
RAG_CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '400'))  # tokens (~1600 chars)
RAG_CHUNK_OVERLAP = float(os.getenv('RAG_CHUNK_OVERLAP', '0.15'))  # 15%

# Retrieval Settings
RAG_K_ANN = int(os.getenv('RAG_K_ANN', '300'))  # Initial ANN candidates per query
RAG_K_BM25 = int(os.getenv('RAG_K_BM25', '200'))  # Initial BM25 candidates per query
RAG_K_MERGE = int(os.getenv('RAG_K_MERGE', '500'))  # Merged pool before rerank

# Scoring Weights
RAG_ALPHA_LEXICAL = float(os.getenv('RAG_ALPHA_LEXICAL', '0.25'))  # Weight for BM25 in hybrid

# Diversity & Final Selection
RAG_MMR_LAMBDA = float(os.getenv('RAG_MMR_LAMBDA', '0.30'))  # Diversity vs relevance tradeoff
RAG_FINAL_LIMIT = int(os.getenv('RAG_FINAL_LIMIT', '30'))  # Final article limit
RAG_PER_QUERY_CAP = int(os.getenv('RAG_PER_QUERY_CAP', '8'))  # Max articles per query (primary query only)
# NOTE: For maximum recall, set RAG_PER_QUERY_CAP=999 to effectively disable cap. MMR will still ensure diversity.

# Queries Configuration File
RAG_QUERIES_FILE = os.getenv('RAG_QUERIES_FILE', os.path.join(os.path.dirname(__file__), 'queries.json'))
