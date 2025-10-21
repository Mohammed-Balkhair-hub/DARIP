"""
Extractors package for full-text enrichment.

Provides utilities for:
- Robots.txt checking (robots_checker)
- HTML fetching with rate limiting (html_fetcher)
- Text extraction from HTML (text_extractor)
"""

from .robots_checker import RobotsChecker
from .html_fetcher import HTMLFetcher
from .text_extractor import TextExtractor

__all__ = ['RobotsChecker', 'HTMLFetcher', 'TextExtractor']

