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

# ===== STEP 3: Deduplication & Clustering =====

# Embeddings & near-dup
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
NEAR_DUP_THRESHOLD = float(os.getenv('NEAR_DUP_THRESHOLD', '0.75'))
EMBED_TEXT_MAX_ABSTRACT_CHARS = int(os.getenv('EMBED_TEXT_MAX_ABSTRACT_CHARS', '160'))

# Dynamic Clustering (always uses agglomerative_auto)
TARGET_CLUSTERS_MIN = int(os.getenv('TARGET_CLUSTERS_MIN', '3'))
TARGET_CLUSTERS_MAX = int(os.getenv('TARGET_CLUSTERS_MAX', '5'))

# Agglomerative clustering tuning
AGGLO_INITIAL_CUTOFF = float(os.getenv('AGGLO_INITIAL_CUTOFF', '0.40'))
AGGLO_CUTOFF_MIN = float(os.getenv('AGGLO_CUTOFF_MIN', '0.30'))
AGGLO_CUTOFF_MAX = float(os.getenv('AGGLO_CUTOFF_MAX', '0.55'))
AGGLO_MAX_ITERS = int(os.getenv('AGGLO_MAX_ITERS', '6'))

# Post-processing controls
MIN_FINAL_CLUSTER_SIZE = int(os.getenv('MIN_FINAL_CLUSTER_SIZE', '2'))
MAX_LARGEST_CLUSTER_FRAC = float(os.getenv('MAX_LARGEST_CLUSTER_FRAC', '0.40'))

# Fallback settings
KMEANS_RANDOM_STATE = int(os.getenv('KMEANS_RANDOM_STATE', '42'))
MIN_ITEMS_FOR_CLUSTERING = int(os.getenv('MIN_ITEMS_FOR_CLUSTERING', '6'))

# Legacy (deprecated)
NUM_CLUSTERS = int(os.getenv('NUM_CLUSTERS', '4'))  # Unused with dynamic clustering

# URL normalization
STRIP_QUERY_PARAMS = [
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid"
]
CANONICALIZE_WWW = os.getenv('CANONICALIZE_WWW', '1') == '1'
LOWERCASE_HOST = os.getenv('LOWERCASE_HOST', '1') == '1'
STRIP_TRAILING_SLASH = os.getenv('STRIP_TRAILING_SLASH', '1') == '1'

# LLM Cluster Refinement (Default mode - recommended)
ENABLE_LLM_CLUSTER_REFINEMENT = os.getenv('ENABLE_LLM_CLUSTER_REFINEMENT', '1') == '1'
REFINEMENT_MAX_FINAL_ARTICLES = int(os.getenv('REFINEMENT_MAX_FINAL_ARTICLES', '20'))
REFINEMENT_MAX_MOVES = int(os.getenv('REFINEMENT_MAX_MOVES', '15'))
REFINEMENT_MAX_NEW_CLUSTERS = int(os.getenv('REFINEMENT_MAX_NEW_CLUSTERS', '2'))
REFINEMENT_MIN_CLUSTER_SIZE = int(os.getenv('REFINEMENT_MIN_CLUSTER_SIZE', '2'))
REFINEMENT_TEMPERATURE = float(os.getenv('REFINEMENT_TEMPERATURE', '0.2'))
LABEL_MAX_WORDS = int(os.getenv('LABEL_MAX_WORDS', '4'))  # Used in both modes

# Legacy Labeling (Mode B - used only if ENABLE_LLM_CLUSTER_REFINEMENT = False)
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
MAX_ABSTRACT_CHARS = int(os.getenv('MAX_ABSTRACT_CHARS', '300'))
MAX_ABSTRACT_SENTENCES = int(os.getenv('MAX_ABSTRACT_SENTENCES', '2'))
INTRA_CLUSTER_DUP_THRESHOLD = float(os.getenv('INTRA_CLUSTER_DUP_THRESHOLD', '0.85'))
TOP_K_FOR_LABELING = int(os.getenv('TOP_K_FOR_LABELING', '6'))
MAX_LABELING_KEYPHRASES = int(os.getenv('MAX_LABELING_KEYPHRASES', '10'))
MICRO_SUMMARY_WORD_LIMIT = int(os.getenv('MICRO_SUMMARY_WORD_LIMIT', '25'))
USE_LLM_FOR_MICRO_SUMMARIES = os.getenv('USE_LLM_FOR_MICRO_SUMMARIES', '0') == '1'
LABELING_MAX_INPUT_TOKENS = int(os.getenv('LABELING_MAX_INPUT_TOKENS', '5000'))
LABELING_MIN_SOURCE_DIVERSITY = int(os.getenv('LABELING_MIN_SOURCE_DIVERSITY', '2'))

# Source preference (optional)
SOURCE_PRIORITY = {
    "arstechnica.com": 3,
    "theverge.com": 3,
    "techcrunch.com": 2
}

# OpenAI settings for labeling
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '128'))

# Paths
LOG_SUBDIR = "_logs"
TMP_SUBDIR = "_tmp"

# Ensure log and tmp directories exist
LOG_TODAY_DIR = os.path.join(OUTPUT_TODAY_DIR, LOG_SUBDIR)
TMP_TODAY_DIR = os.path.join(OUTPUT_TODAY_DIR, TMP_SUBDIR)
os.makedirs(LOG_TODAY_DIR, exist_ok=True)
os.makedirs(TMP_TODAY_DIR, exist_ok=True)
