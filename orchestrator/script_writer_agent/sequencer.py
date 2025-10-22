"""
Sequencer Node - Orders items for optimal coherence and flow

Reads the headliners output and uses OpenAI to intelligently reorder articles
so that related topics flow naturally (e.g., research news together, similar themes adjacent).
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_openai_for_sequencing(items: List[Dict[str, Any]], settings_obj: Any) -> List[str]:
    """
    Call OpenAI API to determine optimal ordering of articles.
    
    Args:
        items: List of article dictionaries with headlines
        settings_obj: Settings object containing OpenAI configuration
        
    Returns:
        List of item_ids in the optimal order
        
    Raises:
        Exception: If OpenAI API call fails after retries
    """
    client = OpenAI(api_key=settings_obj.OPENAI_API_KEY)
    
    # Build a concise representation for the LLM
    items_summary = []
    for idx, item in enumerate(items):
        items_summary.append({
            "index": idx,
            "item_id": item["item_id"],
            "title": item["title"],
            "source": item["source"],
            "headlines_preview": item["headlines"][:150] + "..."  # Just a preview for context
        })
    
    system_prompt = (
        "You are an expert podcast producer. Given a list of news articles with their headlines, "
        "you MUST reorder them to create the best narrative flow and coherence for a podcast. "
        "\n\nGuidelines:"
        "\n1. Group related topics together (e.g., all AI research papers, all hardware/GPU news, all cloud/infrastructure, all policy/regulation)"
        "\n2. Start with a high-impact flagship story to hook listeners"
        "\n3. Create smooth transitions between topic groups"
        "\n4. End with a forward-looking or thought-provoking piece"
        "\n5. Alternate between dense technical content and lighter news for pacing"
        "\n\nIMPORTANT: Actively reorder the items - do NOT return them in the original order (0,1,2,3...). "
        "Return ONLY a JSON array of index numbers in your chosen order. No explanations, no markdown."
    )
    
    user_prompt = (
        f"Here are {len(items)} articles. Reorder them for optimal podcast flow:\n\n"
        f"{json.dumps(items_summary, indent=2)}\n\n"
        f"Return a JSON array with the INDEX numbers (0-{len(items)-1}) in your chosen order.\n"
        f"Example format: [0, 5, 12, 3, 8, ...]"
    )
    
    try:
        response = client.chat.completions.create(
            model=settings_obj.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Some creativity but stay focused
            max_tokens=16384  # gpt-4o-mini maximum output limit
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Debug: Log the raw response
        print(f"[sequencer] LLM raw response (first 500 chars): {response_text[:500]}")
        
        # Parse the JSON array
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        ordered_ids_or_indices = json.loads(response_text)
        print(f"[sequencer] Parsed {len(ordered_ids_or_indices)} entries from LLM response")
        
        if not isinstance(ordered_ids_or_indices, list):
            raise ValueError("LLM did not return a list")
        
        # LLM might return either item_ids or index numbers - handle both
        ordered_ids = []
        for entry in ordered_ids_or_indices:
            if isinstance(entry, int) or (isinstance(entry, str) and entry.isdigit()):
                # LLM returned an index - map it to item_id
                idx = int(entry)
                if 0 <= idx < len(items):
                    ordered_ids.append(items[idx]["item_id"])
                else:
                    print(f"[sequencer] Warning: Invalid index {idx}, skipping")
            else:
                # LLM returned an item_id directly
                ordered_ids.append(str(entry))
        
        print(f"[sequencer] Mapped to {len(ordered_ids)} item IDs")
        
        # Validate we got all items
        if len(ordered_ids) != len(items):
            print(f"[sequencer] WARNING: LLM returned {len(ordered_ids)} IDs but we have {len(items)} items!")
            print(f"[sequencer] Missing {len(items) - len(ordered_ids)} items")
        
        return ordered_ids
    
    except Exception as e:
        print(f"[sequencer] OpenAI API error: {e}")
        raise


def run_sequencer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function: Reorder articles for optimal flow.
    
    This node:
    1. Reads the headliners.json file from state
    2. Calls OpenAI to determine optimal ordering
    3. Reorders the items based on LLM output
    4. Saves to sequenced.json
    5. Returns updated state with sequenced_file path
    
    Args:
        state: Dictionary containing:
            - headliners_file: Path to headliners.json
            - output_dir: Path to output directory
            
    Returns:
        Updated state dictionary with sequenced_file added
    """
    print("[sequencer] ===== STARTING SEQUENCER NODE =====")
    
    headliners_file = state.get("headliners_file", "")
    output_dir = state.get("output_dir", "")
    
    if not headliners_file or not os.path.exists(headliners_file):
        print(f"[sequencer] ERROR: Headliners file not found: {headliners_file}")
        state["sequenced_file"] = ""
        return state
    
    # Read headliners
    with open(headliners_file, "r", encoding="utf-8") as f:
        headliners_data = json.load(f)
    
    items = headliners_data.get("items", [])
    print(f"[sequencer] Loaded {len(items)} articles from headliners.json")
    
    if len(items) == 0:
        print("[sequencer] No items to sequence, skipping...")
        state["sequenced_file"] = headliners_file  # Pass through
        return state
    
    try:
        # Call OpenAI to get optimal ordering
        print("[sequencer] Calling OpenAI to determine optimal article order...")
        ordered_ids = _call_openai_for_sequencing(items, settings)
        
        # Create a mapping of item_id to item
        items_by_id = {item["item_id"]: item for item in items}
        
        # Reorder items based on LLM output
        sequenced_items = []
        used_ids = set()
        
        for item_id in ordered_ids:
            if item_id in items_by_id and item_id not in used_ids:
                sequenced_items.append(items_by_id[item_id])
                used_ids.add(item_id)
        
        # Add any items that weren't in the LLM output (fallback)
        for item in items:
            if item["item_id"] not in used_ids:
                sequenced_items.append(item)
                print(f"[sequencer] Warning: Item {item['item_id']} not in LLM output, appending to end")
        
        print(f"[sequencer] Reordered {len(sequenced_items)} articles")
        
        # Save sequenced results
        output_file = os.path.join(output_dir, "sequenced.json")
        
        output_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_file": headliners_file,
            "sequencing_method": "openai",
            "model": settings.OPENAI_MODEL,
            "total_items": len(sequenced_items),
            "items": sequenced_items
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[sequencer] Saved sequenced articles to: {output_file}")
        print("[sequencer] ===== SEQUENCER NODE COMPLETE =====")
        
        # Update state with sequenced file path
        state["sequenced_file"] = output_file
        
    except Exception as e:
        print(f"[sequencer] ERROR during sequencing: {e}")
        print("[sequencer] Falling back to original order...")
        state["sequenced_file"] = headliners_file  # Use original order as fallback
    
    return state

