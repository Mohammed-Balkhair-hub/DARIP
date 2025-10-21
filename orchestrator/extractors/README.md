# Full-Text Enrichment Extractors

Utilities for fetching and extracting full article text from URLs. Used by the Full-Text Enricher agent (Step 2).

---

## LangGraph Node Flow

The Full-Text Enricher agent uses these extractors in a **10-node workflow**:

```
START: raw_items.json (468 articles)
  ↓
N0: LoadInput → Initialize extractors, build worklist
  ↓
N1: PickNext → Get next article, skip if enriched
  ↓
N2: RobotsCheck [RobotsChecker] → Check robots.txt
  ↓ (if allowed)
N3: FetchHTML [HTMLFetcher] → Fetch with rate limiting
  ↓ (if success, no paywall)  
N4: VariantDiscovery → (currently skipped)
  ↓
N5: ExtractReadable [TextExtractor] → Try 4 libraries
  ↓ (if extracted ≥200 words)
N6: QualityGate → Validate language & length
  ↓ (success or failure)
N7: AdvanceCursor → Write to memory, next article
  ↓
  └──> LOOP to N1
  
(when all done)
  ↓
N8: WriteOutput → Atomic write raw_items.json
  ↓
N9: Done → Return stats
  ↓
END: enriched raw_items.json (~40-65% success)
```

### Routing Logic

**Success:** N1→N2→N3→N4→N5→N6→N7→N1 (loop)  
**Failures:** N2/N3/N5/N6 → N7 → N1 (loop)  
**Complete:** N1 → N8 → N9 → END

---

## Components

### 1. RobotsChecker (`robots_checker.py`)

**Used in:** Node N2  
**Purpose:** Check if robots.txt allows crawling  
**Behavior:** Caches robots.txt per domain, fails open (allows) if unreachable

```python
from extractors import RobotsChecker

checker = RobotsChecker(user_agent="DaripBot/1.0")
allowed = checker.is_allowed("https://example.com/article")
```

**Result:**
- ✅ Allowed → Continue to N3 (fetch)
- ❌ Disallowed → Skip to N7, mark `robots_disallow`

---

### 2. HTMLFetcher (`html_fetcher.py`)

**Used in:** Node N3  
**Purpose:** Fetch HTML with politeness controls  
**Features:**
- Per-domain QPS: 0.5 req/sec (2s interval)
- Per-domain concurrency: max 1
- Retries: 3× with exponential backoff on 429, 5xx
- Paywall detection

```python
from extractors import HTMLFetcher
import asyncio

fetcher = HTMLFetcher(user_agent="DaripBot/1.0", per_domain_qps=0.5)
status, final_url, html_bytes, paywall = await fetcher.fetch(url)
```

**Results:**
- ✅ Status 200, no paywall → Continue to N5 (extract)
- ❌ Paywall → Skip to N7, mark `paywall`
- ❌ 4xx → Skip to N7, mark `http_4xx`
- ❌ 5xx → Skip to N7, mark `http_5xx`
- ❌ Timeout → Skip to N7, mark `timeout`

---

### 3. TextExtractor (`text_extractor.py`)

**Used in:** Node N5  
**Purpose:** Extract clean text using 4-library fallback chain  
**Sequence:**
1. **trafilatura** (primary, fastest, most accurate)
2. **readability** (Mozilla port)
3. **newspaper3k** (news-optimized)
4. **goose3** (alternative)

Accepts first result with ≥200 words.

```python
from extractors import TextExtractor

extractor = TextExtractor(
    extractor_sequence=['trafilatura', 'readability', 'newspaper3k', 'goose3'],
    min_accept_words=200
)

result = extractor.extract(html_bytes, url)
# {'clean_text': '...', 'word_count': 450, 'extraction_method': 'trafilatura'}
```

**Results:**
- ✅ Any library gets ≥200 words → Continue to N6 (quality gate)
- ❌ All fail → Skip to N7, mark `empty_extract`

---

## Quality Gate (N6)

After extraction, articles must pass:

**Length check (EITHER):**
- full_text ≥ 1,000 characters (absolute minimum)
- OR full_text ≥ 2× abstract length

**Language check:**
- Detected language in allowlist (default: `['en']`)

**Pass:** Mark `has_fulltext=true`, write full_text  
**Fail:** Mark `has_fulltext=false`, record failure reason

---

## Configuration

All settings in `config/settings.py`:

```python
# Rate Limiting
ENRICHER_PER_DOMAIN_QPS = 0.5          # 2 seconds per request
ENRICHER_PER_DOMAIN_CONCURRENCY = 1    # Max 1 concurrent/domain

# HTTP
ENRICHER_HTTP_TIMEOUT_SEC = 20         # Request timeout
ENRICHER_HTTP_MAX_RETRIES = 3          # Max retries on 429, 5xx

# Extraction
ENRICHER_EXTRACTOR_SEQUENCE = ['trafilatura', 'readability', 'newspaper3k', 'goose3']
ENRICHER_MIN_ACCEPT_WORDS = 200        # Min words to accept
ENRICHER_MIN_FULL_TEXT_CHARS = 1000    # Quality gate threshold

# Safety
ENRICHER_RESPECT_PAYWALLS = True       # Detect and skip paywalls
ENRICHER_LANG_ALLOWLIST = ['en']       # Only English articles
```

---

## Common Failures

| Failure | Cause | % of Total |
|---------|-------|------------|
| `paywall` | NYTimes, WSJ, Bloomberg, etc. | ~25-35% |
| `timeout` | Slow servers, poor network | ~5-10% |
| `empty_extract` | JS-heavy sites, PDFs | ~5-10% |
| `robots_disallow` | Blocked by robots.txt | ~2-5% |
| `http_4xx` | 404, 403 errors | ~3-8% |
| `http_5xx` | Server errors | ~2-5% |
| `lang_mismatch` | Non-English content | ~1-3% |
| `too_short` | Partial extraction | ~2-5% |

**Success rate:** ~40-65% (200-300 out of 468 articles)

---

## Performance

**Processing time:** ~15-20 minutes for 468 articles  
**Bottleneck:** Per-domain rate limiting (intentional for politeness)  
**Speed per article:** ~2-6 seconds (dominated by network + 2s QPS delay)

**To speed up (less polite):**
```bash
export ENRICHER_PER_DOMAIN_QPS=2.0  # Faster: ~4-6 minutes
```

⚠️ Higher QPS may trigger blocks from some sites.
