"""
HTML fetcher with async support, rate limiting, and paywall detection.

Features:
- Async HTTP with aiohttp
- Per-domain rate limiting (QPS)
- Per-domain concurrency control
- Exponential backoff retries
- Paywall detection
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


class HTMLFetcher:
    """
    Async HTML fetcher with rate limiting and politeness controls.
    """
    
    def __init__(
        self,
        user_agent: str,
        timeout_sec: int = 20,
        max_retries: int = 3,
        per_domain_qps: float = 0.5,
        per_domain_concurrency: int = 1,
        follow_redirects: bool = True,
        respect_paywalls: bool = True
    ):
        """
        Initialize HTMLFetcher.
        
        Args:
            user_agent: User agent string
            timeout_sec: Timeout for HTTP requests
            max_retries: Max retries for failed requests
            per_domain_qps: Max queries per second per domain
            per_domain_concurrency: Max concurrent requests per domain
            follow_redirects: Whether to follow redirects
            respect_paywalls: Whether to detect and respect paywalls
        """
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.per_domain_qps = per_domain_qps
        self.follow_redirects = follow_redirects
        self.respect_paywalls = respect_paywalls
        
        # Per-domain tracking
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.domain_last_request: Dict[str, float] = {}
        self.per_domain_concurrency = per_domain_concurrency
    
    async def fetch(self, url: str) -> Tuple[int, str, bytes, bool]:
        """
        Fetch HTML from URL with rate limiting and retries.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (status_code, final_url, html_bytes, paywall_detected)
        """
        domain = urlparse(url).netloc
        
        # Get or create semaphore for this domain
        if domain not in self.domain_semaphores:
            self.domain_semaphores[domain] = asyncio.Semaphore(self.per_domain_concurrency)
        
        semaphore = self.domain_semaphores[domain]
        
        # Acquire semaphore and enforce QPS
        async with semaphore:
            await self._enforce_qps(domain)
            
            # Attempt fetch with retries
            return await self._fetch_with_retries(url)
    
    async def _enforce_qps(self, domain: str) -> None:
        """
        Enforce QPS limit for domain by sleeping if needed.
        
        Args:
            domain: Domain name
        """
        if domain in self.domain_last_request:
            elapsed = time.time() - self.domain_last_request[domain]
            min_interval = 1.0 / self.per_domain_qps
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"[html_fetcher] Sleeping {sleep_time:.2f}s for QPS limit on {domain}")
                await asyncio.sleep(sleep_time)
        
        self.domain_last_request[domain] = time.time()
    
    async def _fetch_with_retries(self, url: str) -> Tuple[int, str, bytes, bool]:
        """
        Fetch URL with exponential backoff retries.
        
        Args:
            url: URL to fetch
            
        Returns:
            Tuple of (status_code, final_url, html_bytes, paywall_detected)
        """
        retries = 0
        backoff = 1.0
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        while retries <= self.max_retries:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        url,
                        headers=headers,
                        allow_redirects=self.follow_redirects,
                        ssl=False  # More permissive for varied sites
                    ) as response:
                        status_code = response.status
                        final_url = str(response.url)
                        html_bytes = await response.read()
                        
                        # Check if we should retry
                        if status_code in {429, 500, 502, 503, 504} and retries < self.max_retries:
                            logger.warning(f"[html_fetcher] Retrying {url} (status {status_code}, attempt {retries + 1})")
                            retries += 1
                            await asyncio.sleep(backoff)
                            backoff *= 1.5
                            continue
                        
                        # Detect paywall
                        paywall_detected = False
                        if self.respect_paywalls and status_code == 200:
                            paywall_detected = self._detect_paywall(html_bytes, final_url)
                        
                        logger.debug(f"[html_fetcher] Fetched {url} -> {status_code} ({len(html_bytes)} bytes)")
                        
                        return status_code, final_url, html_bytes, paywall_detected
                        
            except asyncio.TimeoutError:
                logger.warning(f"[html_fetcher] Timeout fetching {url} (attempt {retries + 1})")
                retries += 1
                if retries <= self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 1.5
                else:
                    return 408, url, b'', False  # Request Timeout
                    
            except Exception as e:
                logger.error(f"[html_fetcher] Error fetching {url}: {e}")
                return 0, url, b'', False  # Connection error
        
        # Max retries exceeded
        return 503, url, b'', False  # Service Unavailable
    
    def _detect_paywall(self, html_bytes: bytes, url: str) -> bool:
        """
        Detect paywall patterns in HTML content.
        
        Args:
            html_bytes: Raw HTML bytes
            url: URL that was fetched
            
        Returns:
            True if paywall detected, False otherwise
        """
        try:
            html_text = html_bytes.decode('utf-8', errors='ignore').lower()
            
            # Common paywall indicators
            paywall_patterns = [
                'paywall',
                'subscriber-only',
                'subscription required',
                'subscribe to read',
                'register to continue',
                'become a member',
                'premium content',
                'meter-count',
                'regwall',
                'article-locked',
                'content-gate',
                'piano.io',  # Common paywall service
                'meter.md'   # Meter service
            ]
            
            for pattern in paywall_patterns:
                if pattern in html_text:
                    logger.info(f"[html_fetcher] Paywall detected in {url} (pattern: {pattern})")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"[html_fetcher] Error in paywall detection: {e}")
            return False

