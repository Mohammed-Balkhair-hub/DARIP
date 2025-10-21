"""
Text extractor with multiple fallback libraries.

Supports:
- trafilatura (primary)
- readability-lxml
- newspaper3k
- goose3

Also handles:
- AMP/print variant discovery
- Language detection
- Quality validation
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Multi-library text extraction with fallback chain.
    """
    
    def __init__(
        self,
        extractor_sequence: List[str],
        min_accept_words: int = 200,
        max_chars: int = 50000,
        allow_amp_variants: bool = True
    ):
        """
        Initialize TextExtractor.
        
        Args:
            extractor_sequence: Ordered list of extractors to try
            min_accept_words: Minimum word count to accept extraction
            max_chars: Maximum characters to keep from extracted text
            allow_amp_variants: Whether to look for AMP/print variants
        """
        self.extractor_sequence = extractor_sequence
        self.min_accept_words = min_accept_words
        self.max_chars = max_chars
        self.allow_amp_variants = allow_amp_variants
    
    def extract(self, html_bytes: bytes, url: str) -> Optional[Dict]:
        """
        Extract text from HTML using fallback chain.
        
        Args:
            html_bytes: Raw HTML content
            url: Source URL (for context)
            
        Returns:
            Dict with {
                'clean_text': str,
                'word_count': int,
                'extraction_method': str,
                'language_detected': str
            } or None if extraction failed
        """
        html_text = html_bytes.decode('utf-8', errors='ignore')
        
        for extractor_name in self.extractor_sequence:
            try:
                result = self._try_extractor(extractor_name, html_text, url)
                
                if result and result['word_count'] >= self.min_accept_words:
                    logger.info(
                        f"[text_extractor] Extracted {result['word_count']} words "
                        f"using {extractor_name} from {url}"
                    )
                    return result
                    
            except Exception as e:
                logger.debug(f"[text_extractor] {extractor_name} failed for {url}: {e}")
                continue
        
        logger.warning(f"[text_extractor] All extractors failed for {url}")
        return None
    
    def _try_extractor(self, name: str, html: str, url: str) -> Optional[Dict]:
        """
        Try a specific extractor.
        
        Args:
            name: Extractor name
            html: HTML content as string
            url: Source URL
            
        Returns:
            Extraction result dict or None
        """
        if name == 'trafilatura':
            return self._extract_trafilatura(html, url)
        elif name == 'readability':
            return self._extract_readability(html, url)
        elif name == 'newspaper3k':
            return self._extract_newspaper(html, url)
        elif name == 'goose3':
            return self._extract_goose(html, url)
        else:
            logger.warning(f"[text_extractor] Unknown extractor: {name}")
            return None
    
    def _extract_trafilatura(self, html: str, url: str) -> Optional[Dict]:
        """Extract using trafilatura."""
        try:
            import trafilatura
            
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            
            if not text:
                return None
            
            # Trim to max chars
            text = text[:self.max_chars]
            word_count = len(text.split())
            
            # Detect language
            language = self._detect_language(text)
            
            return {
                'clean_text': text,
                'word_count': word_count,
                'extraction_method': 'trafilatura',
                'language_detected': language
            }
            
        except Exception as e:
            logger.debug(f"[text_extractor] trafilatura error: {e}")
            return None
    
    def _extract_readability(self, html: str, url: str) -> Optional[Dict]:
        """Extract using readability-lxml."""
        try:
            from readability import Document
            
            doc = Document(html)
            text = doc.summary()
            
            # Convert HTML to plain text
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            if not text:
                return None
            
            # Trim to max chars
            text = text[:self.max_chars]
            word_count = len(text.split())
            
            # Detect language
            language = self._detect_language(text)
            
            return {
                'clean_text': text,
                'word_count': word_count,
                'extraction_method': 'readability',
                'language_detected': language
            }
            
        except Exception as e:
            logger.debug(f"[text_extractor] readability error: {e}")
            return None
    
    def _extract_newspaper(self, html: str, url: str) -> Optional[Dict]:
        """Extract using newspaper3k."""
        try:
            from newspaper import Article
            
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            text = article.text
            
            if not text:
                return None
            
            # Trim to max chars
            text = text[:self.max_chars]
            word_count = len(text.split())
            
            # Detect language (newspaper provides this)
            language = article.meta_lang if hasattr(article, 'meta_lang') else self._detect_language(text)
            
            return {
                'clean_text': text,
                'word_count': word_count,
                'extraction_method': 'newspaper3k',
                'language_detected': language
            }
            
        except Exception as e:
            logger.debug(f"[text_extractor] newspaper3k error: {e}")
            return None
    
    def _extract_goose(self, html: str, url: str) -> Optional[Dict]:
        """Extract using goose3."""
        try:
            from goose3 import Goose
            
            g = Goose()
            article = g.extract(raw_html=html)
            
            text = article.cleaned_text
            
            if not text:
                return None
            
            # Trim to max chars
            text = text[:self.max_chars]
            word_count = len(text.split())
            
            # Detect language
            language = self._detect_language(text)
            
            return {
                'clean_text': text,
                'word_count': word_count,
                'extraction_method': 'goose3',
                'language_detected': language
            }
            
        except Exception as e:
            logger.debug(f"[text_extractor] goose3 error: {e}")
            return None
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., 'en')
        """
        try:
            from langdetect import detect
            
            # Use first 1000 chars for speed
            sample = text[:1000]
            lang = detect(sample)
            return lang
            
        except Exception as e:
            logger.debug(f"[text_extractor] Language detection failed: {e}")
            return 'unknown'
    
    def discover_variants(self, html_bytes: bytes, base_url: str) -> List[str]:
        """
        Discover AMP/print/mobile variants from HTML.
        
        Args:
            html_bytes: Raw HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of variant URLs found
        """
        if not self.allow_amp_variants:
            return []
        
        try:
            from urllib.parse import urljoin
            
            html_text = html_bytes.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_text, 'html.parser')
            
            variants = []
            
            # Look for AMP link
            amp_link = soup.find('link', rel='amphtml')
            if amp_link and amp_link.get('href'):
                amp_url = urljoin(base_url, amp_link['href'])
                variants.append(amp_url)
                logger.debug(f"[text_extractor] Found AMP variant: {amp_url}")
            
            # Look for print variant in meta or common patterns
            # Many sites have ?print=1 or /print/ variants
            if '/print' not in base_url and '?print' not in base_url:
                # Try common print URL patterns
                if base_url.endswith('/'):
                    print_url = base_url + 'print'
                else:
                    print_url = base_url + '/print'
                variants.append(print_url)
            
            return variants
            
        except Exception as e:
            logger.debug(f"[text_extractor] Variant discovery failed: {e}")
            return []

