"""
Naturalizer Node - Polish dialogue for natural flow while preserving facts

Transforms rigid factual lines into smooth, natural dialogue:
- Maintains fact-consistency via refs
- Adds conversational flow and micro-transitions
- Balances speaker distribution
- Optional SSML generation for TTS
- Respects segment boundaries
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_openai_for_polish(
    lines_jsonl: str,
    constraints: Dict[str, Any]
) -> str:
    """
    Call OpenAI to polish a segment's lines.
    
    Args:
        lines_jsonl: Raw JSONL text from Node 3
        constraints: Polish constraints
        
    Returns:
        Polished JSONL text
        
    Raises:
        Exception: If OpenAI API call fails after retries
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    system_prompt = (
        "You are a dialogue naturalizer for a two-host tech podcast. "
        "You receive a JSONL script where each line is a factual, TTS-friendly sentence with references. "
        "Your task is to polish the wording, rhythm, and flow into natural conversation without changing facts. "
        "Preserve refs as provided. You may merge adjacent lines by the same speaker or split a long line, "
        "but do not move lines across segments. Keep or add brief transitions, adjust pause durations, "
        "and ensure consistent spoken forms for numbers, times, and acronyms. "
        "Output only JSONL, one line per object."
    )
    
    user_prompt = (
        f"**Policy:**\n"
        f"- Tone: {constraints['tone']}\n"
        f"- Pace: {constraints['pace']}\n"
        f"- Max words per line: {constraints['max_words_per_line']}\n"
        f"- Micro-transitions: {constraints['micro_transitions']}\n"
        f"- Lock segment boundaries: {constraints['lock_segment_boundaries']}\n"
        f"- Allow merge adjacent lines: {constraints['allow_merge_adjacent_lines']}\n"
        f"- Allow split long lines: {constraints['allow_split_long_line']}\n\n"
        f"**Style notes:**\n"
        f"- Avoid hype, no slang\n"
        f"- UK English preferred\n"
        f"- Vary line lengths (8-16 words ideal)\n"
        f"- Add conversational openers sparingly: 'Quick update...', 'Meanwhile...'\n"
        f"- Keep numbers and acronyms in spoken form\n\n"
        f"**Script to polish (JSONL):**\n"
        f"{lines_jsonl}\n\n"
        f"**Output:** Polished JSONL with same schema. Preserve refs. Add 'notes' field if needed for pronunciation."
    )
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,  # More creative for naturalness
            max_tokens=16384
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[naturalizer] OpenAI API error: {e}")
        raise


def _parse_jsonl_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse JSONL response from LLM."""
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
            print(f"[naturalizer] Failed to parse line: {line_text[:100]}")
            continue
    
    return lines


def _validate_polish(
    original_lines: List[Dict[str, Any]],
    polished_lines: List[Dict[str, Any]],
    constraints: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Validate polished lines against originals.
    
    Args:
        original_lines: Lines from Node 3
        polished_lines: Polished lines
        constraints: Polish constraints
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Collect all original refs
    original_refs = set()
    for line in original_lines:
        if line.get("refs"):
            original_refs.update(line["refs"])
    
    # Collect all polished refs
    polished_refs = set()
    for line in polished_lines:
        if line.get("refs"):
            polished_refs.update(line["refs"])
    
    # Check refs preserved (polished can be superset due to merges)
    if not polished_refs.issuperset(original_refs):
        missing_refs = original_refs - polished_refs
        return False, f"Missing refs in polished version: {missing_refs}"
    
    # Check word counts
    for line in polished_lines:
        word_count = len(line.get("text", "").split())
        if word_count > constraints["max_words_per_line"]:
            return False, f"Line {line.get('line_id')} too long: {word_count} words"
    
    return True, ""


def _check_speaker_balance(lines: List[Dict[str, Any]], constraints: Dict[str, Any]) -> bool:
    """
    Check if speaker distribution is balanced.
    
    Args:
        lines: All script lines
        constraints: Polish constraints
        
    Returns:
        True if balanced within tolerance
    """
    speaker_counts = defaultdict(int)
    for line in lines:
        speaker_counts[line.get("speaker")] += 1
    
    if len(speaker_counts) < 2:
        return True  # Only one speaker, can't balance
    
    total_lines = sum(speaker_counts.values())
    percentages = {speaker: (count / total_lines) * 100 for speaker, count in speaker_counts.items()}
    
    # Check if any speaker is outside tolerance
    expected_pct = 100 / len(speaker_counts)
    tolerance = constraints.get("balance_tolerance_pct", 10)
    
    for speaker, pct in percentages.items():
        if abs(pct - expected_pct) > tolerance:
            print(f"[naturalizer] Speaker balance warning: {speaker} has {pct:.1f}% (expected ~{expected_pct:.1f}%)")
            return False
    
    return True


def _add_ssml(line: Dict[str, Any]) -> str:
    """
    Generate SSML for a line.
    
    Args:
        line: Script line dictionary
        
    Returns:
        SSML string
    """
    text = line.get("text", "")
    pause_ms = line.get("pause_ms_after", 0)
    
    # Simple SSML wrapping
    ssml = f"<speak>{text}"
    if pause_ms > 0:
        ssml += f"<break time='{pause_ms}ms'/>"
    ssml += "</speak>"
    
    return ssml


def _polish_segment_lines(
    segment_lines: List[Dict[str, Any]],
    constraints: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Polish lines for a single segment.
    
    Args:
        segment_lines: Lines from one segment
        constraints: Polish constraints
        
    Returns:
        Polished lines for the segment
    """
    if not segment_lines:
        return []
    
    segment_id = segment_lines[0].get("segment_id", "unknown")
    print(f"[naturalizer] Polishing segment {segment_id} ({len(segment_lines)} lines)...")
    
    # Convert lines to JSONL string
    lines_jsonl = "\n".join(json.dumps(line, ensure_ascii=False) for line in segment_lines)
    
    try:
        # Call OpenAI for polish
        polished_text = _call_openai_for_polish(lines_jsonl, constraints)
        
        # Parse response
        polished_lines = _parse_jsonl_response(polished_text)
        
        if not polished_lines:
            print(f"[naturalizer] Warning: No polished lines returned for {segment_id}, using original")
            return segment_lines
        
        # Validate
        is_valid, error_msg = _validate_polish(segment_lines, polished_lines, constraints)
        if not is_valid:
            print(f"[naturalizer] Validation failed for {segment_id}: {error_msg}")
            print(f"[naturalizer] Using original lines as fallback")
            return segment_lines
        
        # Add SSML if enabled
        if constraints.get("ssml", False):
            for line in polished_lines:
                if "ssml" not in line:
                    line["ssml"] = _add_ssml(line)
        
        print(f"[naturalizer] Polished {len(polished_lines)} lines for {segment_id}")
        return polished_lines
    
    except Exception as e:
        print(f"[naturalizer] Error polishing {segment_id}: {e}")
        print(f"[naturalizer] Using original lines as fallback")
        return segment_lines


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _generate_intro_segment(items: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate an energetic introduction segment (SEG_0).
    
    Args:
        items: List of news items to preview
        constraints: Script constraints
        
    Returns:
        List of intro line dictionaries
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Extract titles for preview
    titles = [item.get("title", "")[:80] for item in items[:5]]  # First 5 items
    topics_preview = "\n".join(f"- {title}" for title in titles if title)
    
    system_prompt = (
        "You are a podcast script writer creating an energetic introduction for a tech news podcast called DARIP. "
        "Generate 4-6 dialogue lines between hosts Adam and Sara. "
        "The intro should: (1) Have the hosts introduce themselves warmly, "
        "(2) Introduce the podcast name 'DARIP' with energy and enthusiasm, "
        "(3) Give a quick, exciting preview of today's topics. "
        "Keep it conversational, upbeat, and engaging. "
        "Output JSONL format with the same schema as the main script."
    )
    
    user_prompt = (
        f"**Hosts:** Adam and Sara\n"
        f"**Podcast name:** DARIP\n"
        f"**Today's topics preview:**\n{topics_preview}\n\n"
        f"**Requirements:**\n"
        f"- 4-6 lines total\n"
        f"- Alternate between Adam and Sara\n"
        f"- Max {constraints['max_words_per_line']} words per line\n"
        f"- Energetic, welcoming tone\n"
        f"- Mention 2-3 key topics from the list\n\n"
        f"**Output format (JSONL):**\n"
        f'{{"line_id":1,"segment_id":"SEG_0","speaker":"Adam","voice":null,"item_id":"intro","text":"...","refs":[],"pause_ms_after":180,"secs_estimate":3.5}}\n\n'
        f"Start with Adam welcoming listeners."
    )
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,  # More creative for intro
            max_tokens=2048
        )
        
        response_text = response.choices[0].message.content.strip()
        intro_lines = _parse_jsonl_response(response_text)
        
        # Ensure all lines have SEG_0 and item_id="intro"
        for line in intro_lines:
            line["segment_id"] = "SEG_0"
            line["item_id"] = "intro"
            if "refs" not in line:
                line["refs"] = []
            if "secs_estimate" not in line:
                line["secs_estimate"] = len(line.get("text", "").split()) / settings.SCRIPT_WPM_ESTIMATE * 60
        
        print(f"[naturalizer] Generated {len(intro_lines)} intro lines for SEG_0")
        return intro_lines
        
    except Exception as e:
        print(f"[naturalizer] Error generating intro: {e}")
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _generate_outro_segment(constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a closing outro segment (SEG_END).
    
    Args:
        constraints: Script constraints
        
    Returns:
        List of outro line dictionaries
    """
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    system_prompt = (
        "You are a podcast script writer creating a closing outro for a tech news podcast called DARIP. "
        "Generate 3-4 dialogue lines between hosts Adam and Sara. "
        "The outro should: (1) Thank listeners for tuning in, "
        "(2) Mention that DARIP is a DAILY podcast with new episodes every day, "
        "(3) Encourage them to follow for daily updates, "
        "(4) End with a memorable sign-off. "
        "Keep it warm, professional, and engaging. "
        "Output JSONL format with the same schema as the main script."
    )
    
    user_prompt = (
        f"**Hosts:** Adam and Sara\n"
        f"**Podcast name:** DARIP\n"
        f"**Important:** DARIP is a DAILY podcast that publishes new episodes every day on the website\n"
        f"**Requirements:**\n"
        f"- 3-4 lines total\n"
        f"- Alternate between Adam and Sara\n"
        f"- Max {constraints['max_words_per_line']} words per line\n"
        f"- Warm, professional closing tone\n"
        f"- Thank listeners and encourage daily engagement\n"
        f"- Mention that new episodes come out daily\n\n"
        f"**Output format (JSONL):**\n"
        f'{{"line_id":1,"segment_id":"SEG_END","speaker":"Adam","voice":null,"item_id":"outro","text":"...","refs":[],"pause_ms_after":180,"secs_estimate":3.5}}\n\n'
        f"Start with Adam thanking listeners."
    )
    
    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,  # More creative for outro
            max_tokens=2048
        )
        
        response_text = response.choices[0].message.content.strip()
        outro_lines = _parse_jsonl_response(response_text)
        
        # Ensure all lines have SEG_END and item_id="outro"
        for line in outro_lines:
            line["segment_id"] = "SEG_END"
            line["item_id"] = "outro"
            if "refs" not in line:
                line["refs"] = []
            if "secs_estimate" not in line:
                line["secs_estimate"] = len(line.get("text", "").split()) / settings.SCRIPT_WPM_ESTIMATE * 60
        
        print(f"[naturalizer] Generated {len(outro_lines)} outro lines for SEG_END")
        return outro_lines
        
    except Exception as e:
        print(f"[naturalizer] Error generating outro: {e}")
        return []


def _normalize_text_for_tts(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize text in script lines for TTS pronunciation.
    
    Converts all variations of DARIP (DARIP, Darip, DaRiP, etc.) to lowercase "darip"
    so TTS pronounces it as a word instead of spelling it out.
    
    Args:
        lines: List of script line dictionaries
        
    Returns:
        List of lines with normalized text
    """
    import re
    
    for line in lines:
        if "text" in line and line["text"]:
            # Replace all case variations of DARIP with lowercase "darip"
            # Use word boundary to avoid replacing partial matches
            line["text"] = re.sub(r'\bDARIP\b', 'darip', line["text"])
            line["text"] = re.sub(r'\bDarip\b', 'darip', line["text"])
            line["text"] = re.sub(r'\bDaRiP\b', 'darip', line["text"])  # Any other variations
    
    return lines


def run_naturalizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function: Polish script for natural dialogue flow.
    
    This node:
    1. Reads script_lines.jsonl from state
    2. Groups lines by segment
    3. Polishes each segment for naturalness
    4. Validates fact preservation and speaker balance
    5. Saves to script_lines_polished.jsonl
    6. Returns updated state with polished_script_file path
    
    Args:
        state: Dictionary containing:
            - duo_script_file: Path to script_lines.jsonl
            - output_dir: Path to output directory
            
    Returns:
        Updated state dictionary with polished_script_file added
    """
    print("[naturalizer] ===== STARTING NATURALIZER NODE =====")
    
    duo_script_file = state.get("duo_script_file", "")
    output_dir = state.get("output_dir", "")
    
    if not duo_script_file or not os.path.exists(duo_script_file):
        print(f"[naturalizer] ERROR: Script file not found: {duo_script_file}")
        state["polished_script_file"] = ""
        return state
    
    # Read script lines
    lines = []
    with open(duo_script_file, "r", encoding="utf-8") as f:
        for line_text in f:
            line_text = line_text.strip()
            if line_text:
                lines.append(json.loads(line_text))
    
    print(f"[naturalizer] Loaded {len(lines)} lines from script")
    
    # Build constraints from settings
    constraints = {
        "target_total_secs": settings.SCRIPT_TARGET_TOTAL_SECS,
        "pace": settings.POLISH_PACE,
        "tone": settings.POLISH_TONE,
        "micro_transitions": settings.POLISH_ENABLE_MICRO_TRANSITIONS,
        "ssml": settings.POLISH_ENABLE_SSML,
        "max_words_per_line": settings.SCRIPT_MAX_WORDS_PER_LINE,
        "lock_segment_boundaries": settings.POLISH_LOCK_SEGMENT_BOUNDARIES,
        "allow_merge_adjacent_lines": settings.POLISH_ALLOW_MERGE_LINES,
        "allow_split_long_line": settings.POLISH_ALLOW_SPLIT_LINES,
        "balance_tolerance_pct": settings.POLISH_BALANCE_TOLERANCE_PCT
    }
    
    # Group lines by segment
    segments = defaultdict(list)
    for line in lines:
        segment_id = line.get("segment_id", "unknown")
        segments[segment_id].append(line)
    
    print(f"[naturalizer] Processing {len(segments)} segments...")
    
    # Polish each segment
    all_polished_lines = []
    for segment_id in sorted(segments.keys()):
        segment_lines = segments[segment_id]
        polished_lines = _polish_segment_lines(segment_lines, constraints)
        all_polished_lines.extend(polished_lines)
    
    # Generate introduction segment (SEG_0)
    print("[naturalizer] Generating introduction segment (SEG_0)...")
    intro_lines = _generate_intro_segment(state.get("items", []), constraints)
    
    # Generate outro segment (SEG_END)
    print("[naturalizer] Generating outro segment (SEG_END)...")
    outro_lines = _generate_outro_segment(constraints)
    
    # Separate intro lines from news content lines
    if intro_lines:
        print(f"[naturalizer] Added {len(intro_lines)} intro lines to beginning")
        # Keep intro lines separate - they should stay at the beginning
        for line in intro_lines:
            line["line_id"] = 0  # Temporary ID for intro lines
    
    if outro_lines:
        print(f"[naturalizer] Added {len(outro_lines)} outro lines to end")
        # Keep outro lines separate - they should stay at the end
        for line in outro_lines:
            line["line_id"] = 999999  # High temporary ID for outro lines
    
    # Sort only the news content lines by line_id
    print(f"[naturalizer] Sorting {len(all_polished_lines)} news content lines by line_id...")
    all_polished_lines.sort(key=lambda x: x.get("line_id", 0))
    
    # Combine intro lines (at beginning) + sorted news lines + outro lines (at end)
    final_lines = []
    if intro_lines:
        final_lines.extend(intro_lines)
    final_lines.extend(all_polished_lines)
    if outro_lines:
        final_lines.extend(outro_lines)
    
    # Reassign sequential line_ids starting from 1
    for idx, line in enumerate(final_lines, start=1):
        line["line_id"] = idx
    
    print(f"[naturalizer] Reassigned line_ids from 1 to {len(final_lines)}")
    
    # Normalize text for TTS (convert DARIP variations to lowercase)
    print("[naturalizer] Normalizing text for TTS pronunciation...")
    final_lines = _normalize_text_for_tts(final_lines)
    print("[naturalizer] Converted DARIP variations to 'darip' for proper TTS pronunciation")
    
    # Update all_polished_lines to use the final combined list
    all_polished_lines = final_lines
    
    # Check speaker balance if enabled
    if constraints.get("enforce_speaker_balance", settings.POLISH_ENFORCE_BALANCE):
        is_balanced = _check_speaker_balance(all_polished_lines, constraints)
        if not is_balanced:
            print("[naturalizer] Speaker balance is outside tolerance (±10%)")
    
    # Calculate total duration
    total_secs = sum(
        line.get("secs_estimate", 0) + line.get("pause_ms_after", 0)/1000
        for line in all_polished_lines
    )
    
    print(f"[naturalizer] Polished {len(all_polished_lines)} total lines")
    print(f"[naturalizer] Estimated duration: {total_secs:.1f}s ({total_secs/60:.1f} min)")
    
    # Save as JSONL
    output_file = os.path.join(output_dir, "script_lines_polished.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for line in all_polished_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    print(f"[naturalizer] Saved polished script to: {output_file}")
    print("[naturalizer] ===== NATURALIZER NODE COMPLETE =====")
    
    # Update state
    state["polished_script_file"] = output_file
    
    return state

