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
