"""
Podcast Script Writer Agent - Node Implementations

This package contains individual node implementations for the podcast script generation workflow.
Each node is a self-contained module that can be composed into a LangGraph workflow.

Nodes:
- headliner: Condenses articles into crisp headings and key facts
- sequencer: Orders items for coherence and flow (groups related topics)
- duo_script: Generates TTS-ready line-by-line dialogue between Adam and Sara
- naturalizer: Polishes phrasing for natural flow while preserving facts
"""

