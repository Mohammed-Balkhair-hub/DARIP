"""
Podcast Script Writer - LangGraph Orchestrator

This module orchestrates the multi-node workflow for generating podcast scripts.
Currently implements:
- Headliner: Condenses articles into crisp headings and key facts

Future nodes:
- Sequencer: Orders items for coherence and flow
- DuoScript: Writes dialogue between Adam and Sara
- Naturalizer: Smoothens phrasing and adds connective tissue
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from config import settings
from script_writer_agent.headliner import run_headliner
from script_writer_agent.sequencer import run_sequencer
from script_writer_agent.duo_script import run_duo_script
from script_writer_agent.naturalizer import run_naturalizer


class ScriptWriterState(TypedDict):
    """State for the podcast script writer workflow."""
    items: List[Dict[str, Any]]
    input_file: str
    output_dir: str
    headliners_file: str
    sequenced_file: str
    duo_script_file: str
    polished_script_file: str


def script_writer():
    """
    Main entry point for podcast script generation workflow.
    
    This function:
    1. Reads articles from queried_news.json
    2. Creates output directory for podcast scripts
    3. Builds and executes LangGraph workflow
    4. Returns the final state
    
    The workflow currently consists of:
    - Headliner node: Generates headline bullet points for each article
    
    Future nodes will be added to the graph as they are implemented.
    """
    print("[script_writer] ===== STARTING PODCAST SCRIPT GENERATION =====")
    
    # Set up paths
    input_file = os.path.join(settings.OUTPUT_TODAY_DIR, "queried_news.json")
    output_dir = os.path.join(settings.OUTPUT_TODAY_DIR, settings.PODCAST_SCRIPT_SUBDIR)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[script_writer] Input: {input_file}")
    print(f"[script_writer] Output: {output_dir}")
    
    # Read input articles
    if not os.path.exists(input_file):
        print(f"[script_writer] ERROR: Input file not found: {input_file}")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = data.get("items", [])
    print(f"[script_writer] Loaded {len(items)} articles from queried_news.json")
    
    # Initialize state
    initial_state: ScriptWriterState = {
        "items": items,
        "input_file": input_file,
        "output_dir": output_dir,
        "headliners_file": "",
        "sequenced_file": "",
        "duo_script_file": "",
        "polished_script_file": ""
    }
    
    # Build LangGraph workflow
    print("[script_writer] Building LangGraph workflow...")
    graph = StateGraph(ScriptWriterState)
    
    # Add nodes
    graph.add_node("headliner", run_headliner)
    graph.add_node("sequencer", run_sequencer)
    graph.add_node("duo_script", run_duo_script)
    graph.add_node("naturalizer", run_naturalizer)
    
    # Define workflow edges
    graph.set_entry_point("headliner")
    graph.add_edge("headliner", "sequencer")
    graph.add_edge("sequencer", "duo_script")
    graph.add_edge("duo_script", "naturalizer")
    graph.add_edge("naturalizer", END)
    
    # Compile the graph
    app = graph.compile()
    
    print("[script_writer] Executing workflow...")
    
    # Execute the workflow
    result = app.invoke(initial_state)
    
    print(f"[script_writer] Workflow complete!")
    print(f"[script_writer] Headliners saved to: {result.get('headliners_file', 'N/A')}")
    print(f"[script_writer] Sequenced saved to: {result.get('sequenced_file', 'N/A')}")
    print(f"[script_writer] Script lines saved to: {result.get('duo_script_file', 'N/A')}")
    print(f"[script_writer] Polished script saved to: {result.get('polished_script_file', 'N/A')}")
    print("[script_writer] ===== PODCAST SCRIPT GENERATION COMPLETE =====")
    
    return result

