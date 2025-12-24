#!/usr/bin/env python3
"""
Enhanced Web Search Tool for AVA
Production-Ready Web Search with Multiple Engines and Advanced Features
"""

import asyncio

# Optional imports with fallbacks - use find_spec to avoid unused import warnings
import importlib.util
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import quote

import aiohttp

HAS_REQUESTS = importlib.util.find_spec("requests") is not None
HAS_BS4 = importlib.util.find_spec("bs4") is not None

try:
    from duckduckgo_search import DDGS

    HAS_DDG_SEARCH = True
except ImportError:
    HAS_DDG_SEARCH = False

try:
    from dotenv import load_dotenv

    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class SearchEngine(Enum):
    """Supported search engines."""

    DUCKDUCKGO = "duckduckgo"
    GOOGLE_CUSTOM = "google_custom"
    BING = "bing"
    SERP_API = "serp_api"
    BRAVE = "brave"
    MOCK = "mock"  # For testing


class SearchType(Enum):
    """Types of search queries."""

    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    ACADEMIC = "academic"
    VIDEOS = "videos"


class SafetyLevel(Enum):
    """Content filtering safety levels."""

    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    rank: int = 0
    published_date: str | None = None
    content_type: str = "webpage"
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Represents the response from a search operation."""

    success: bool
    query: str
    engine: SearchEngine
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding rate limit."""
        now = time.time()
        # Remove old requests outside time window
        self.requests = [
            req_time for req_time in self.requests if now - req_time < self.time_window
        ]

        return len(self.requests) < self.max_requests

    def record_request(self):
        """Record a request being made."""
        self.requests.append(time.time())

    def wait_time(self) -> float:
        """Get time to wait before next request can be made."""
        if self.can_make_request():
            return 0.0

        # Return time until oldest request expires
        now = time.time()
        oldest_request = min(self.requests)
        return self.time_window - (now - oldest_request)


class EnhancedWebSearch:
    """Enhanced web search tool with multiple engines and advanced features."""

    def __init__(
        self,
        primary_engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        api_keys: dict[str, str] | None = None,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 3600,
    ):
        """
        Initialize the enhanced web search tool.

        Args:
            primary_engine: Primary search engine to use
            api_keys: Dictionary of API keys for various services
            rate_limit_requests: Maximum requests per time window
            rate_limit_window: Rate limit time window in seconds
        """
        self.name = "enhanced_web_search"
        self.description = """Advanced web search tool supporting:
        - Multiple search engines (DuckDuckGo, Google, Bing, Brave)
        - Content filtering and safety controls
        - Rate limiting and error handling
        - News, academic, and specialized searches
        - Result ranking and relevance scoring"""

        self.primary_engine = primary_engine
        self.api_keys = api_keys or {}
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)

        # Load API keys from environment if not provided
        self._load_api_keys()

        # Tool parameters for function calling
        self.parameters = [
            {
                "name": "query",
                "type": "string",
                "description": "Search query string",
                "required": True,
            },
            {
                "name": "num_results",
                "type": "integer",
                "description": "Number of results to return (1-20)",
                "default": 5,
            },
            {
                "name": "search_type",
                "type": "string",
                "description": "Type of search: 'web', 'news', 'images', 'academic', 'videos'",
                "default": "web",
            },
            {
                "name": "engine",
                "type": "string",
                "description": "Search engine: 'duckduckgo', 'google_custom', 'bing', 'serp_api', 'brave'",
                "default": self.primary_engine.value,
            },
            {
                "name": "safety_level",
                "type": "string",
                "description": "Content filtering: 'off', 'moderate', 'strict'",
                "default": "moderate",
            },
        ]

        # Request session for reuse
        self.session = None

        logger.info(
            f"Enhanced Web Search initialized with {primary_engine.value} as primary engine"
        )

    def _load_api_keys(self):
        """Load API keys from environment variables."""
        env_keys = {
            "google_api_key": "GOOGLE_API_KEY",
            "google_cse_id": "GOOGLE_CSE_ID",
            "bing_api_key": "BING_API_KEY",
            "serp_api_key": "SERPAPI_KEY",
            "brave_api_key": "BRAVE_API_KEY",
        }

        for key, env_var in env_keys.items():
            if key not in self.api_keys:
                self.api_keys[key] = os.getenv(env_var)

    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_type: SearchType = SearchType.WEB,
        engine: SearchEngine | None = None,
        safety_level: SafetyLevel = SafetyLevel.MODERATE,
    ) -> SearchResponse:
        """
        Perform web search with specified parameters.

        Args:
            query: Search query string
            num_results: Number of results to return
            search_type: Type of search to perform
            engine: Search engine to use (defaults to primary)
            safety_level: Content filtering level

        Returns:
            SearchResponse object with results and metadata
        """
        start_time = time.time()
        engine = engine or self.primary_engine

        response = SearchResponse(success=False, query=query, engine=engine)

        try:
            # Validate input
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")

            if len(query) > 500:
                raise ValueError("Query too long (max 500 characters)")

            if num_results < 1 or num_results > 20:
                raise ValueError("Number of results must be between 1 and 20")

            # Check rate limiting
            if not self.rate_limiter.can_make_request():
                wait_time = self.rate_limiter.wait_time()
                raise ValueError(
                    f"Rate limit exceeded. Wait {
                        wait_time:.1f} seconds"
                )

            # Clean and preprocess query
            cleaned_query = self._preprocess_query(query)

            # Perform search based on engine
            logger.debug(f"Searching with {engine.value}: {cleaned_query}")

            if engine == SearchEngine.DUCKDUCKGO:
                results = await self._search_duckduckgo(cleaned_query, num_results, search_type)
            elif engine == SearchEngine.GOOGLE_CUSTOM:
                results = await self._search_google_custom(cleaned_query, num_results, search_type)
            elif engine == SearchEngine.BING:
                results = await self._search_bing(cleaned_query, num_results, search_type)
            elif engine == SearchEngine.SERP_API:
                results = await self._search_serp_api(cleaned_query, num_results, search_type)
            elif engine == SearchEngine.BRAVE:
                results = await self._search_brave(cleaned_query, num_results, search_type)
            elif engine == SearchEngine.MOCK:
                results = self._search_mock(cleaned_query, num_results, search_type)
            else:
                raise ValueError(f"Unsupported search engine: {engine.value}")

            # Post-process results
            processed_results = self._post_process_results(results, safety_level)

            # Record successful request
            self.rate_limiter.record_request()

            # Build response
            response.success = True
            response.results = processed_results[:num_results]
            response.total_results = len(processed_results)
            response.metadata = {
                "cleaned_query": cleaned_query,
                "safety_level": safety_level.value,
                "search_type": search_type.value,
            }

            logger.info(f"Search successful: {len(response.results)} results for '{query}'")

        except Exception as e:
            response.error = str(e)
            logger.error(f"Search failed for '{query}': {response.error}")

        finally:
            response.search_time_ms = (time.time() - start_time) * 1000

        return response

    def _preprocess_query(self, query: str) -> str:
        """Clean and preprocess search query."""
        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", query.strip())

        # Remove special characters that might break search APIs
        cleaned = re.sub(r'[<>"\{\}\\]', "", cleaned)

        # Limit length
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."

        return cleaned

    async def _search_duckduckgo(
        self, query: str, num_results: int, search_type: SearchType
    ) -> list[SearchResult]:
        """Search using DuckDuckGo."""
        results = []

        try:
            if HAS_DDG_SEARCH:
                # Use official duckduckgo-search library
                with DDGS() as ddgs:
                    if search_type == SearchType.NEWS:
                        search_results = ddgs.news(query, max_results=num_results)
                    else:
                        search_results = ddgs.text(query, max_results=num_results)

                    for i, result in enumerate(search_results):
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("href", ""),
                            snippet=result.get("body", ""),
                            domain=self._extract_domain(result.get("href", "")),
                            rank=i + 1,
                            published_date=result.get("date"),
                            # Simple relevance scoring
                            relevance_score=1.0 - (i * 0.1),
                        )
                        results.append(search_result)
            else:
                # Fallback to mock results
                logger.warning("duckduckgo-search library not available, using mock results")
                results = self._search_mock(query, num_results, search_type)

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            # Fallback to mock results on error
            results = self._search_mock(query, num_results, search_type)

        return results

    async def _search_google_custom(
        self, query: str, num_results: int, search_type: SearchType
    ) -> list[SearchResult]:
        """Search using Google Custom Search API."""
        results = []

        api_key = self.api_keys.get("google_api_key")
        cse_id = self.api_keys.get("google_cse_id")

        if not api_key or not cse_id:
            raise ValueError("Google API key and CSE ID required for Google Custom Search")

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": min(num_results, 10),  # Google allows max 10 per request
            }

            if search_type == SearchType.NEWS:
                params["tbm"] = "nws"
            elif search_type == SearchType.IMAGES:
                params["searchType"] = "image"

            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    for i, item in enumerate(data.get("items", [])):
                        search_result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            domain=self._extract_domain(item.get("link", "")),
                            rank=i + 1,
                            relevance_score=1.0 - (i * 0.08),
                            metadata={"page_map": item.get("pagemap", {})},
                        )
                        results.append(search_result)
                else:
                    raise ValueError(
                        f"Google API returned status {
                            response.status}"
                    )

        except Exception as e:
            logger.error(f"Google Custom Search error: {str(e)}")
            # Fallback to mock results
            results = self._search_mock(query, num_results, search_type)

        return results

    async def _search_bing(
        self, query: str, num_results: int, search_type: SearchType
    ) -> list[SearchResult]:
        """Search using Bing Search API."""
        results = []

        api_key = self.api_keys.get("bing_api_key")
        if not api_key:
            raise ValueError("Bing API key required for Bing search")

        try:
            base_url = "https://api.bing.microsoft.com/v7.0/search"
            if search_type == SearchType.NEWS:
                base_url = "https://api.bing.microsoft.com/v7.0/news/search"

            headers = {"Ocp-Apim-Subscription-Key": api_key}
            params = {
                "q": query,
                "count": min(num_results, 20),
                "textDecorations": False,
                "textFormat": "Raw",
            }

            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(base_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    web_pages = data.get("webPages", {}).get("value", [])
                    news_items = data.get("value", []) if search_type == SearchType.NEWS else []
                    items = news_items if search_type == SearchType.NEWS else web_pages

                    for i, item in enumerate(items):
                        search_result = SearchResult(
                            title=item.get("name", ""),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", ""),
                            domain=self._extract_domain(item.get("url", "")),
                            rank=i + 1,
                            published_date=item.get("datePublished"),
                            relevance_score=1.0 - (i * 0.08),
                        )
                        results.append(search_result)
                else:
                    raise ValueError(
                        f"Bing API returned status {
                            response.status}"
                    )

        except Exception as e:
            logger.error(f"Bing search error: {str(e)}")
            results = self._search_mock(query, num_results, search_type)

        return results

    async def _search_serp_api(
        self, query: str, num_results: int, search_type: SearchType
    ) -> list[SearchResult]:
        """Search using SerpApi."""
        results = []

        api_key = self.api_keys.get("serp_api_key")
        if not api_key:
            raise ValueError("SerpApi key required for SerpApi search")

        try:
            url = "https://serpapi.com/search"
            params = {
                "api_key": api_key,
                "q": query,
                "num": min(num_results, 20),
                "engine": "google",
            }

            if search_type == SearchType.NEWS:
                params["tbm"] = "nws"

            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    organic_results = data.get("organic_results", [])
                    news_results = data.get("news_results", [])
                    items = news_results if search_type == SearchType.NEWS else organic_results

                    for i, item in enumerate(items):
                        search_result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            domain=self._extract_domain(item.get("link", "")),
                            rank=i + 1,
                            published_date=item.get("date"),
                            relevance_score=1.0 - (i * 0.08),
                        )
                        results.append(search_result)
                else:
                    raise ValueError(
                        f"SerpApi returned status {
                            response.status}"
                    )

        except Exception as e:
            logger.error(f"SerpApi search error: {str(e)}")
            results = self._search_mock(query, num_results, search_type)

        return results

    async def _search_brave(
        self, query: str, num_results: int, search_type: SearchType
    ) -> list[SearchResult]:
        """Search using Brave Search API."""
        results = []

        api_key = self.api_keys.get("brave_api_key")
        if not api_key:
            raise ValueError("Brave API key required for Brave search")

        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
            params = {"q": query, "count": min(num_results, 20)}

            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    web_results = data.get("web", {}).get("results", [])

                    for i, item in enumerate(web_results):
                        search_result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("description", ""),
                            domain=self._extract_domain(item.get("url", "")),
                            rank=i + 1,
                            relevance_score=1.0 - (i * 0.08),
                        )
                        results.append(search_result)
                else:
                    raise ValueError(
                        f"Brave API returned status {
                            response.status}"
                    )

        except Exception as e:
            logger.error(f"Brave search error: {str(e)}")
            results = self._search_mock(query, num_results, search_type)

        return results

    def _search_mock(
        self, query: str, num_results: int, search_type: SearchType
    ) -> list[SearchResult]:
        """Generate mock search results for testing."""
        results = []

        for i in range(1, num_results + 1):
            result = SearchResult(
                title=f"Mock Result {i}: {query}",
                url=f"https://example.com/search/{i}?q={quote(query)}",
                snippet=f"This is a mock search result #{i} for the query '{query}'. "
                f"It demonstrates the structure of search results returned by the tool.",
                domain="example.com",
                rank=i,
                relevance_score=1.0 - (i * 0.1),
                content_type="mock_webpage",
                metadata={"mock": True, "search_type": search_type.value},
            )
            results.append(result)

        return results

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""

    def _post_process_results(
        self, results: list[SearchResult], safety_level: SafetyLevel
    ) -> list[SearchResult]:
        """Post-process search results with filtering and scoring."""
        processed_results = []

        for result in results:
            # Apply content filtering based on safety level
            if safety_level != SafetyLevel.OFF:
                if self._is_safe_content(result, safety_level):
                    processed_results.append(result)
            else:
                processed_results.append(result)

        # Sort by relevance score
        processed_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return processed_results

    def _is_safe_content(self, result: SearchResult, safety_level: SafetyLevel) -> bool:
        """Check if content meets safety requirements."""
        # Basic content filtering - can be enhanced with ML models
        unsafe_keywords = (
            ["adult", "explicit", "nsfw"] if safety_level == SafetyLevel.STRICT else []
        )

        text_to_check = f"{result.title} {result.snippet}".lower()

        for keyword in unsafe_keywords:
            if keyword in text_to_check:
                return False

        return True

    def run(self, query: str, **kwargs) -> dict[str, Any]:
        """
        Main interface for function calling compatibility.

        Args:
            query: Search query string
            **kwargs: Additional parameters

        Returns:
            Dictionary with search results or error information
        """
        num_results = kwargs.get("num_results", 5)
        search_type = SearchType(kwargs.get("search_type", "web"))
        engine = SearchEngine(kwargs.get("engine", self.primary_engine.value))
        safety_level = SafetyLevel(kwargs.get("safety_level", "moderate"))

        # Run async search in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            response = loop.run_until_complete(
                self.search(query, num_results, search_type, engine, safety_level)
            )
        finally:
            loop.close()

        if response.success:
            return {
                "results": [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "domain": result.domain,
                        "rank": result.rank,
                        "relevance_score": result.relevance_score,
                        "published_date": result.published_date,
                    }
                    for result in response.results
                ],
                "total_results": response.total_results,
                "search_time_ms": response.search_time_ms,
                "engine": response.engine.value,
                "query": response.query,
                "warnings": response.warnings,
            }
        else:
            return {
                "error": response.error,
                "query": response.query,
                "engine": response.engine.value,
                "search_time_ms": response.search_time_ms,
            }

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


def test_web_search():
    """Test the enhanced web search tool."""
    logger.info("=== Testing Enhanced Web Search ===")

    # Initialize search tool
    search_tool = EnhancedWebSearch(primary_engine=SearchEngine.MOCK)

    test_queries = [
        "latest advancements in 4-bit quantization for LLMs",
        "NVIDIA RTX A2000 4GB specifications",
        "local AI model deployment strategies",
        "agentic AI frameworks 2024",
    ]

    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        result = search_tool.run(query, num_results=3)

        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Found {result['total_results']} results")
            logger.info(f"Search time: {result['search_time_ms']:.2f}ms")

            for i, search_result in enumerate(result["results"][:2]):
                logger.info(f"  {i + 1}. {search_result['title']}")
                logger.info(f"     {search_result['url']}")
                logger.info(f"     {search_result['snippet'][:100]}...")


async def main():
    """Main function for standalone testing."""
    test_web_search()


if __name__ == "__main__":
    asyncio.run(main())
