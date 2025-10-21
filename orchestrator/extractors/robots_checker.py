"""
Robots.txt checker with caching.

Fetches and parses robots.txt for domains, caches decisions
to avoid repeated fetches during a single run.
"""

import logging
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests

logger = logging.getLogger(__name__)


class RobotsChecker:
    """
    Checks robots.txt compliance for URLs with per-domain caching.
    """
    
    def __init__(self, user_agent: str, timeout: int = 10):
        """
        Initialize RobotsChecker.
        
        Args:
            user_agent: User agent string to check permissions for
            timeout: Timeout in seconds for fetching robots.txt
        """
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache: Dict[str, RobotFileParser] = {}
        self.failed_domains: set = set()
    
    def is_allowed(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.
        
        Args:
            url: Full URL to check
            
        Returns:
            True if allowed (or if robots.txt fetch fails), False if disallowed
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Check if we've already failed to fetch robots.txt for this domain
            if domain in self.failed_domains:
                # Assume allowed if robots.txt is unreachable
                return True
            
            # Get or fetch robots.txt parser for this domain
            parser = self._get_parser(domain, parsed.scheme)
            
            if parser is None:
                # Failed to fetch, assume allowed
                return True
            
            # Check if URL is allowed
            allowed = parser.can_fetch(self.user_agent, url)
            
            if not allowed:
                logger.info(f"[robots_checker] Disallowed by robots.txt: {url}")
            
            return allowed
            
        except Exception as e:
            logger.warning(f"[robots_checker] Error checking robots.txt for {url}: {e}")
            # On error, assume allowed (fail open)
            return True
    
    def _get_parser(self, domain: str, scheme: str = 'https') -> Optional[RobotFileParser]:
        """
        Get cached RobotFileParser for domain, or fetch and cache it.
        
        Args:
            domain: Domain name (e.g., 'example.com')
            scheme: URL scheme ('http' or 'https')
            
        Returns:
            RobotFileParser instance or None if fetch failed
        """
        # Check cache
        if domain in self.cache:
            return self.cache[domain]
        
        # Fetch robots.txt
        robots_url = f"{scheme}://{domain}/robots.txt"
        
        try:
            logger.debug(f"[robots_checker] Fetching robots.txt from {robots_url}")
            
            parser = RobotFileParser()
            parser.set_url(robots_url)
            
            # Fetch with timeout
            response = requests.get(
                robots_url,
                timeout=self.timeout,
                headers={'User-Agent': self.user_agent},
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Parse the robots.txt content
                parser.parse(response.text.splitlines())
                self.cache[domain] = parser
                logger.debug(f"[robots_checker] Cached robots.txt for {domain}")
                return parser
            else:
                # No robots.txt or error, assume allowed
                logger.debug(f"[robots_checker] No robots.txt for {domain} (status {response.status_code})")
                self.failed_domains.add(domain)
                return None
                
        except Exception as e:
            logger.warning(f"[robots_checker] Failed to fetch robots.txt for {domain}: {e}")
            self.failed_domains.add(domain)
            return None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache size and failed domains count
        """
        return {
            'cached_domains': len(self.cache),
            'failed_domains': len(self.failed_domains)
        }

