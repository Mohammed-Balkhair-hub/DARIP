"""
DuoScript Node - Generate TTS-ready line-by-line dialogue script

Converts sequenced headlines into structured JSONL dialogue with:
- Two-host alternation (Adam & Sara)
- Fact-only content with traceable refs
- TTS-friendly text (numbers spoken, acronyms spelled)
- Flexible speaker run-length
- Pause timing for natural rhythm
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


def _estimate_duration(text: str) -> float:
    """
    Estimate spoken duration in seconds based on word count.
    
    Args:
        text: The text to estimate
        
    Returns:
        Estimated duration in seconds
    """
    word_count = len(text.split())
    return (word_count / settings.SCRIPT_WPM_ESTIMATE) * 60


def _generate_segment_id(item: Dict[str, Any]) -> str:
    """
    Generate a stable segment ID from item title or ID.
    
    Args:
        item: Article item dictionary
        
    Returns:
        Segment ID string (e.g., "SEG_level-4-autonomous")
    """
    title = item.get("title", "")
    # Convert to kebab-case, max 40 chars
    segment = re.sub(r'[^a-z0-9]+', '-', title.lower())[:40].strip('-')
    return f"{settings.SCRIPT_SEGMENT_PREFIX}{segment}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_openai_for_script(
    item: Dict[str, Any],
    headlines_array: List[str],
    constraints: Dict[str, Any],
    item_idx: int
) -> str:
    """
    Call OpenAI to generate script lines for one item.
    
    Args:
        item: Article item dictionary
        headlines_array: List of headline bullet points
        constraints: Script generation constraints
        item_idx: Index of this item in sequence
        
    Returns:
        Raw JSONL text from LLM
        
    Raises:
        Exception: If OpenAI API call fails after retries
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    system_prompt = (
        "You are a senior broadcast writer generating a two-host, line-by-line script from sequenced news items. "
        "Output JSONL, one JSON object per line. Each line must be ≤18 words and contain exactly one idea. "
        "Facts must come strictly from the provided headlines. "
        "Convert digits to spoken numbers (e.g., '768 GB' → 'seven hundred sixty-eight gigabytes'). "
        "Spell acronyms with hyphens (e.g., 'GDDR7' → 'G-D-D-R-seven', 'RT' → 'R-T'). "
        "Allow consecutive lines by the same speaker within an item for natural rhythm. "
        "Attach refs as 0-based indices into the item's bullet list. "
        "Include pause_ms_after (ms) for pacing. "
        "Do not add external facts. Do not include markdown or code fences. Produce only JSON lines."
    )
    
    # Build the item context
    item_context = {
        "item_id": item["item_id"],
        "title": item["title"],
        "source": item["source"],
        "url": item["url"],
        "headlines": headlines_array
    }
    
    user_prompt = (
        f"Generate {constraints['max_lines_per_item']} script lines for this news item.\n\n"
        f"**Speakers:** {', '.join(constraints['speaker_names'])}\n"
        f"**Max words per line:** {constraints['max_words_per_line']}\n"
        f"**Default pause:** {constraints['default_pause_ms']}ms\n"
        f"**Style:** Neutral, concise, friendly, no hype\n"
        f"**Allow consecutive speaker lines:** {constraints['allow_consecutive_speaker_lines']}\n\n"
        f"**Item {item_idx + 1}:**\n"
        f"{json.dumps(item_context, indent=2)}\n\n"
        f"**Coverage pattern:**\n"
        f"Line 1 (HOOK): Big picture or 'what happened'\n"
        f"Line 2 (SCOPE): Constrain context (who/where/when)\n"
        f"Lines 3-4 (KEY FACTS): Numbers, dates, concrete claims (map to refs)\n"
        f"Line 5 (IMPACT): Why it matters to devs/enterprises/users\n"
        f"Line 6+ (BUTTON): Short close or segue\n\n"
        f"**Output format (JSONL, one object per line):**\n"
        f'{{"line_id":1,"segment_id":"SEG_...","speaker":"Adam","voice":null,"item_id":"...","text":"...","refs":[0],"pause_ms_after":180,"secs_estimate":4.2}}\n\n'
        f"Start with {'Adam' if item_idx % 2 == 0 else 'Sara'}. Alternate speakers at least once within the item."
    )
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Factual, structured
            max_tokens=4096
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[duo_script] OpenAI API error for item {item_idx}: {e}")
        raise


def _validate_line(
    line: Dict[str, Any],
    item: Dict[str, Any],
    headlines_array: List[str],
    constraints: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Validate a single script line.
    
    Args:
        line: Script line dictionary
        item: Original article item
        headlines_array: List of headline bullets
        constraints: Script constraints
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    required_fields = ["line_id", "segment_id", "speaker", "item_id", "text", "refs", "pause_ms_after"]
    for field in required_fields:
        if field not in line:
            return False, f"Missing required field: {field}"
    
    # Check speaker valid
    if line["speaker"] not in constraints["speaker_names"]:
        return False, f"Invalid speaker: {line['speaker']}"
    
    # Check word count
    word_count = len(line["text"].split())
    if word_count > constraints["max_words_per_line"]:
        return False, f"Text too long: {word_count} words > {constraints['max_words_per_line']}"
    
    # Check refs valid
    if line["item_id"] and line["item_id"] != "null":
        if not line["refs"] or len(line["refs"]) == 0:
            return False, "Content line must have at least one ref"
        for ref_idx in line["refs"]:
            if not isinstance(ref_idx, int) or ref_idx < 0 or ref_idx >= len(headlines_array):
                return False, f"Invalid ref index: {ref_idx} (headlines has {len(headlines_array)} items)"
    
    return True, ""


def _parse_jsonl_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSONL response from LLM.
    
    Args:
        response_text: Raw JSONL text
        
    Returns:
        List of parsed line dictionaries
    """
    lines = []
    
    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("jsonl") or response_text.startswith("json"):
            response_text = response_text.split('\n', 1)[1]
        response_text = response_text.strip()
    
    # Parse each line
    for line_text in response_text.split('\n'):
        line_text = line_text.strip()
        if not line_text:
            continue
        try:
            line_obj = json.loads(line_text)
            lines.append(line_obj)
        except json.JSONDecodeError as e:
            print(f"[duo_script] Failed to parse line: {line_text[:100]}")
            print(f"[duo_script] Error: {e}")
            continue
    
    return lines


def _generate_item_lines(
    item: Dict[str, Any],
    constraints: Dict[str, Any],
    item_idx: int,
    global_line_id: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate script lines for a single item.
    
    Args:
        item: Article item dictionary
        constraints: Script generation constraints
        item_idx: Index of this item in sequence
        global_line_id: Starting line ID
        
    Returns:
        Tuple of (lines_list, next_global_line_id)
    """
    print(f"[duo_script] Generating lines for item {item_idx + 1}: {item['title'][:60]}...")
    
    # Split headlines into array
    headlines_text = item.get("headlines", "")
    headlines_array = [h.strip() for h in headlines_text.split('\n') if h.strip() and h.strip().startswith('•')]
    headlines_array = [h[1:].strip() for h in headlines_array]  # Remove bullet
    
    if not headlines_array:
        print(f"[duo_script] Warning: No headlines for item {item_idx + 1}, skipping")
        return [], global_line_id
    
    # Generate segment ID
    segment_id = _generate_segment_id(item)
    
    try:
        # Call OpenAI
        response_text = _call_openai_for_script(item, headlines_array, constraints, item_idx)
        
        # Parse JSONL
        lines = _parse_jsonl_response(response_text)
        
        if not lines:
            print(f"[duo_script] Warning: No lines parsed from LLM response for item {item_idx + 1}")
            return [], global_line_id
        
        # Validate and fix each line
        validated_lines = []
        for line in lines:
            # Assign global line_id
            line["line_id"] = global_line_id
            global_line_id += 1
            
            # Ensure segment_id is set
            if "segment_id" not in line or not line["segment_id"]:
                line["segment_id"] = segment_id
            
            # Ensure secs_estimate is set
            if "secs_estimate" not in line:
                line["secs_estimate"] = _estimate_duration(line.get("text", ""))
            
            # Validate
            is_valid, error_msg = _validate_line(line, item, headlines_array, constraints)
            if not is_valid:
                print(f"[duo_script] Line {line['line_id']} validation failed: {error_msg}")
                continue
            
            validated_lines.append(line)
        
        print(f"[duo_script] Generated {len(validated_lines)} valid lines for item {item_idx + 1}")
        return validated_lines, global_line_id
    
    except Exception as e:
        print(f"[duo_script] Error generating lines for item {item_idx + 1}: {e}")
        return [], global_line_id


def run_duo_script(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function: Generate TTS-ready script lines from sequenced items.
    
    This node:
    1. Reads sequenced.json from state
    2. For each item, generates 6-8 dialogue lines
    3. Validates all lines (word count, refs, speakers)
    4. Saves to script_lines.jsonl
    5. Returns updated state with duo_script_file path
    
    Args:
        state: Dictionary containing:
            - sequenced_file: Path to sequenced.json
            - output_dir: Path to output directory
            
    Returns:
        Updated state dictionary with duo_script_file added
    """
    print("[duo_script] ===== STARTING DUO SCRIPT NODE =====")
    
    sequenced_file = state.get("sequenced_file", "")
    output_dir = state.get("output_dir", "")
    
    if not sequenced_file or not os.path.exists(sequenced_file):
        print(f"[duo_script] ERROR: Sequenced file not found: {sequenced_file}")
        state["duo_script_file"] = ""
        return state
    
    # Read sequenced items
    with open(sequenced_file, "r", encoding="utf-8") as f:
        sequenced_data = json.load(f)
    
    items = sequenced_data.get("items", [])
    print(f"[duo_script] Loaded {len(items)} sequenced items")
    
    # Build constraints from settings
    constraints = {
        "speaker_names": settings.SCRIPT_SPEAKER_NAMES,
        "target_total_secs": settings.SCRIPT_TARGET_TOTAL_SECS,
        "max_lines_per_item": settings.SCRIPT_MAX_LINES_PER_ITEM,
        "max_words_per_line": settings.SCRIPT_MAX_WORDS_PER_LINE,
        "allow_consecutive_speaker_lines": settings.SCRIPT_ALLOW_CONSECUTIVE_SPEAKER,
        "digits_to_speech": settings.SCRIPT_DIGITS_TO_SPEECH,
        "spell_acronyms": settings.SCRIPT_SPELL_ACRONYMS,
        "default_pause_ms": settings.SCRIPT_DEFAULT_PAUSE_MS,
        "segment_prefix": settings.SCRIPT_SEGMENT_PREFIX
    }
    
    # Generate lines for all items
    all_lines = []
    global_line_id = 1
    
    for item_idx, item in enumerate(items):
        lines, global_line_id = _generate_item_lines(item, constraints, item_idx, global_line_id)
        all_lines.extend(lines)
    
    if not all_lines:
        print("[duo_script] ERROR: No lines generated!")
        state["duo_script_file"] = ""
        return state
    
    # Calculate total estimated duration
    total_secs = sum(line.get("secs_estimate", 0) + line.get("pause_ms_after", 0)/1000 for line in all_lines)
    
    print(f"[duo_script] Generated {len(all_lines)} total lines")
    print(f"[duo_script] Estimated duration: {total_secs:.1f}s ({total_secs/60:.1f} min)")
    print(f"[duo_script] Target: {constraints['target_total_secs']}s ({constraints['target_total_secs']/60:.1f} min)")
    
    # Save as JSONL
    output_file = os.path.join(output_dir, "script_lines.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    print(f"[duo_script] Saved {len(all_lines)} lines to: {output_file}")
    print("[duo_script] ===== DUO SCRIPT NODE COMPLETE =====")
    
    # Update state
    state["duo_script_file"] = output_file
    
    return state

