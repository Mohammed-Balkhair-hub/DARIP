# Podcast Script Writer Agent

Multi-node LangGraph workflow for generating TTS-ready podcast scripts from news articles.

## Overview

The Script Writer transforms curated news articles into a polished, two-host dialogue script ready for text-to-speech conversion. It uses a four-stage pipeline with OpenAI LLM integration at each step.

## Architecture

### Workflow Pipeline

```
queried_news.json
    ↓
[Node 1: Headliner]     → headliners.json
    ↓
[Node 2: Sequencer]     → sequenced.json
    ↓
[Node 3: DuoScript]     → script_lines.jsonl
    ↓
[Node 4: Naturalizer]   → script_lines_polished.jsonl
```

## Nodes

### Node 1: Headliner (`headliner.py`)

**Purpose:** Extract verifiable facts from article full text into bullet-point headlines.

**Input:** `queried_news.json` (RAG-selected articles with full text)

**Output:** `headliners.json`
```json
{
  "generated_at": "2025-10-22T...",
  "total_processed": 30,
  "total_skipped": 0,
  "items": [
    {
      "item_id": "abc123...",
      "title": "Original article title",
      "source": "Source name",
      "url": "https://...",
      "headlines": "• Fact one\n• Fact two\n• Fact three"
    }
  ]
}
```

**Process:**
- Reads each article's `full_text`
- Calls OpenAI to extract key facts (dates, numbers, concrete nouns)
- Quotes limited to 25 consecutive words
- Skips articles with empty full text
- One API call per article

### Node 2: Sequencer (`sequencer.py`)

**Purpose:** Reorder articles for optimal podcast flow and topical coherence.

**Input:** `headliners.json`

**Output:** `sequenced.json` (same structure, reordered items)

**Process:**
- Sends all articles with headlines to OpenAI
- LLM groups related topics (research, hardware, policy, etc.)
- Returns optimal order as index array
- Maps indices back to item_ids
- Preserves all items, just changes order
- One API call total

**Sequencing Strategy:**
- Start with high-impact flagship story
- Group related topics together
- Smooth transitions between groups
- End with forward-looking piece
- Alternate dense/light content for pacing

### Node 3: DuoScript (`duo_script.py`)

**Purpose:** Generate TTS-ready line-by-line dialogue between two hosts.

**Input:** `sequenced.json`

**Output:** `script_lines.jsonl` (one JSON object per line)

**Format:**
```json
{"line_id":1,"segment_id":"SEG_autonomous-driving","speaker":"Adam","voice":null,"item_id":"abc123","text":"NVIDIA is unlocking level four autonomous driving with six AI breakthroughs.","refs":[0],"pause_ms_after":180,"secs_estimate":4.2}
```

**Process:**
- Splits headlines into array of bullets per item
- Calls OpenAI to generate 6-8 dialogue lines per item
- Validates each line (word count, refs, speaker)
- Assigns global line IDs and segment IDs
- Estimates spoken duration
- One API call per article

**Coverage Pattern (per item):**
1. **Hook:** Big picture / what happened
2. **Scope:** Who/where/when context
3. **Key Facts:** Numbers, dates, concrete claims
4. **Impact:** Why it matters
5. **Button:** Short close or segue

**TTS Transformations:**
- Numbers: "768 GB" → "seven hundred sixty-eight gigabytes"
- Acronyms: "GDDR7" → "G-D-D-R-seven"
- Times: "07:00 BST" → "zero seven hundred B-S-T"

**Speaker Alternation:**
- Allows 1-3 consecutive lines per speaker
- Alternates at least once per item
- Rotates starting speaker per item

### Node 4: Naturalizer (`naturalizer.py`)

**Purpose:** Polish dialogue for natural conversational flow while preserving facts.

**Input:** `script_lines.jsonl`

**Output:** `script_lines_polished.jsonl` (enhanced version)

**Process:**
- Groups lines by segment
- Polishes each segment for naturalness
- Validates fact preservation (refs must be maintained)
- Checks speaker balance across episode
- Optionally adds SSML tags
- One API call per segment
- **NEW:** Generates intro segment (SEG_0) with energetic host introduction
- **NEW:** Generates outro segment (SEG_END) with daily podcast engagement
- **NEW:** Normalizes DARIP text variations to lowercase "darip" for TTS
- **NEW:** Sorts and reassigns sequential line_ids (1, 2, 3...)

**Enhancements:**
- More natural phrasing and rhythm
- Conversational openers ("Quick update...", "Meanwhile...")
- Varied line lengths (8-16 words)
- Smooth micro-transitions between topics
- Pronunciation notes for tricky terms
- Balanced speaker distribution (±10%)
- **NEW:** Energetic intro with host introductions and topic preview
- **NEW:** Professional outro encouraging daily engagement
- **NEW:** TTS-optimized text (DARIP → darip for proper pronunciation)

**Allowed Operations:**
- Rephrase for naturalness
- Merge adjacent lines (same speaker)
- Split long lines
- Adjust pause timing
- Add micro-transitions

**Disallowed:**
- Adding new facts not in refs
- Moving lines across segments
- Removing segments
- Changing fact content

## Configuration

All settings are in `orchestrator/config/settings.py` and configurable via environment variables.

### Script Generation Settings (Node 3)

```python
SCRIPT_SPEAKER_NAMES = "Adam,Sara"           # Comma-separated host names
SCRIPT_TARGET_TOTAL_SECS = 480               # Target duration (8 minutes)
SCRIPT_MAX_LINES_PER_ITEM = 8                # Lines per article
SCRIPT_MAX_WORDS_PER_LINE = 18               # Word limit per line
SCRIPT_DEFAULT_PAUSE_MS = 120                # Default pause duration
SCRIPT_SEGMENT_PREFIX = "SEG_"               # Segment ID prefix
SCRIPT_ALLOW_CONSECUTIVE_SPEAKER = 1         # Allow same speaker multiple lines
SCRIPT_DIGITS_TO_SPEECH = 1                  # Convert numbers to words
SCRIPT_SPELL_ACRONYMS = 1                    # Spell acronyms with hyphens
SCRIPT_WPM_ESTIMATE = 150                    # Words per minute for duration
```

### Polish Settings (Node 4)

```python
POLISH_PACE = "medium"                       # brisk, medium, relaxed
POLISH_TONE = "warm, clear, no slang"        # Dialogue tone
POLISH_ENABLE_MICRO_TRANSITIONS = 1          # Add "Meanwhile..." etc.
POLISH_ENABLE_SSML = 0                       # Generate SSML tags
POLISH_LOCK_SEGMENT_BOUNDARIES = 1           # Don't move lines across segments
POLISH_ALLOW_MERGE_LINES = 1                 # Merge adjacent same-speaker lines
POLISH_ALLOW_SPLIT_LINES = 1                 # Split long lines
POLISH_ENFORCE_BALANCE = 1                   # Balance speaker distribution
POLISH_BALANCE_TOLERANCE_PCT = 10            # ±10% tolerance
```

## Usage

### Run Full Pipeline

```bash
# From run_daily.py
python run_daily.py
```

The script writer runs as Step 4 after RAG retrieval.

### Run Script Writer Only

```python
from agents.script_writer import script_writer
result = script_writer()
```

### Override Settings

```bash
# Via environment variables
export SCRIPT_TARGET_TOTAL_SECS=600  # 10 minutes
export SCRIPT_MAX_WORDS_PER_LINE=20
export POLISH_PACE=brisk
export POLISH_ENABLE_SSML=1

python run_daily.py
```

## Output Files

All outputs are saved to `/data/outputs/{date}/podcast_script/`:

1. **`headliners.json`** - Condensed headlines with bullet points
2. **`sequenced.json`** - Articles reordered for flow
3. **`script_lines.jsonl`** - Raw TTS-ready dialogue (JSONL format)
4. **`script_lines_polished.jsonl`** - Polished natural dialogue (JSONL format)

## JSONL Schema

Each line in the JSONL files follows this schema:

```json
{
  "line_id": 1,
  "segment_id": "SEG_autonomous-driving",
  "speaker": "Adam",
  "voice": null,
  "item_id": "426062dfa98321a28816598d898f9fac",
  "text": "NVIDIA is unlocking level four autonomous driving with six breakthroughs.",
  "refs": [0, 1],
  "pause_ms_after": 180,
  "secs_estimate": 4.2,
  "ssml": "<speak>...</speak>",
  "notes": {"pronunciation": [{"term": "NVIDIA", "hint": "en-VID-ee-ah"}]}
}
```

**Field Descriptions:**
- `line_id`: Global unique line number (1, 2, 3...)
- `segment_id`: Stable ID for this article's segment
- `speaker`: Host name ("Adam" or "Sara")
- `voice`: TTS voice ID (future use)
- `item_id`: Original article ID (null for intro/outro)
- `text`: TTS-ready sentence (≤18 words)
- `refs`: Array of indices into item's headline bullets (0-based)
- `pause_ms_after`: Milliseconds to pause after this line
- `secs_estimate`: Estimated spoken duration
- `ssml`: (Optional) SSML version of text
- `notes`: (Optional) Pronunciation hints

## Error Handling

### Validation Checks

**DuoScript (Node 3):**
- Required fields present
- Speaker in allowed list
- Text ≤ max_words_per_line
- Refs valid (indices into headlines array)
- At least one ref per content line

**Naturalizer (Node 4):**
- All original refs preserved
- No new facts added
- Word count limits maintained
- Speaker balance within tolerance

### Fallback Strategy

If a node fails:
1. Retry with exponential backoff (3 attempts)
2. Log error and continue with fallback
3. Use previous node's output unchanged
4. Pipeline continues to completion

### Common Issues

**Missing items in sequencer output:**
- Increase `max_tokens` in sequencer
- LLM response may be truncated

**Speaker imbalance:**
- Adjust `POLISH_BALANCE_TOLERANCE_PCT`
- Check `POLISH_ENFORCE_BALANCE` setting

**Duration mismatch:**
- Tune `SCRIPT_WPM_ESTIMATE` (affects duration calculation)
- Adjust `SCRIPT_MAX_LINES_PER_ITEM` to add/remove content
- Modify `SCRIPT_TARGET_TOTAL_SECS` target

## Performance

**API Calls:**
- Headliner: N calls (one per article)
- Sequencer: 1 call (all articles at once)
- DuoScript: N calls (one per article)
- Naturalizer: M calls (one per segment, M ≈ N)

**Total:** Approximately 3N + 1 OpenAI API calls for N articles

**Typical Run (30 articles):**
- API calls: ~91 calls
- Duration: ~2-3 minutes (with rate limits)
- Output: ~200-250 dialogue lines
- Cost: ~$0.05-0.10 (with gpt-4o-mini)

## Development

### Adding a New Node

1. Create `script_writer_agent/my_node.py`
2. Implement `run_my_node(state: Dict) -> Dict`
3. Import in `agents/script_writer.py`
4. Add to LangGraph workflow
5. Update state TypedDict
6. Add settings to `config/settings.py`

### Testing Individual Nodes

```python
from script_writer_agent.headliner import run_headliner

state = {
    "items": [...],
    "output_dir": "/tmp/test"
}
result = run_headliner(state)
```

## Dependencies

- `openai` - OpenAI API client
- `langgraph` - Workflow orchestration
- `tenacity` - Retry logic with exponential backoff

## Recent Updates

### ✅ Intro/Outro Generation (COMPLETED)
- **Intro Segment (SEG_0):** Energetic host introduction with topic preview
- **Outro Segment (SEG_END):** Professional closing encouraging daily engagement
- **Daily Podcast Focus:** Outro emphasizes DARIP as a daily podcast

### ✅ TTS Text Normalization (COMPLETED)
- **DARIP Normalization:** Converts all variations (DARIP, Darip, etc.) to lowercase "darip"
- **TTS Optimization:** Prevents TTS from spelling out "D-A-R-I-P" letter by letter
- **Natural Pronunciation:** Ensures "darip" is pronounced as a word

### ✅ Script Ordering & Skip Logic (COMPLETED)
- **Sequential Line IDs:** All lines get proper sequential IDs (1, 2, 3...)
- **Skip Logic:** Nodes can skip execution if output files already exist
- **Efficient Re-runs:** Allows partial pipeline execution for debugging

## Future Enhancements

- Voice ID mapping per speaker
- Custom pronunciation dictionaries
- Music/sfx cue insertion
- Multi-language support
- Speaker emotion/emphasis hints
- Advanced SSML features

