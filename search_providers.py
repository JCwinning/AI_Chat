"""
Search providers integration for AI Chat application.
Supports multiple search providers with Tavily AI as the primary implementation.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import requests


class SearchResult:
    """Represents a single search result"""
    def __init__(self, title: str, url: str, snippet: str, date: Optional[str] = None, source: Optional[str] = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.date = date
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "date": self.date,
            "source": self.source
        }


class SearchResponse:
    """Represents the complete search response"""
    def __init__(self, query: str, results: List[SearchResult], summary: Optional[str] = None):
        self.query = query
        self.results = results
        self.summary = summary
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary,
            "timestamp": self.timestamp
        }


class TavilySearchProvider:
    """Tavily AI search provider implementation"""

    def __init__(self, api_key: str, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://api.tavily.com"

    async def search(self, query: str, include_domains: Optional[List[str]] = None,
                   exclude_domains: Optional[List[str]] = None,
                   search_depth: str = "basic") -> SearchResponse:
        """
        Perform web search using Tavily API

        Args:
            query: Search query string
            include_domains: List of domains to limit search to
            exclude_domains: List of domains to exclude from search
            search_depth: "basic" or "advanced" search depth

        Returns:
            SearchResponse object with results
        """
        try:
            # Prepare search request payload
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_depth,
                "include_answer": True,
                "include_raw_content": False,
                "max_results": self.max_results,
                "include_images": False,
                "include_image_descriptions": False
            }

            # Add domain filters if provided
            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            # Make synchronous request (Tavily doesn't have async Python SDK yet)
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            for item in data.get("results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    date=item.get("published_date"),
                    source=self._extract_source_from_url(item.get("url", ""))
                )
                results.append(result)

            # Get answer summary if available
            summary = data.get("answer", "")

            return SearchResponse(
                query=query,
                results=results,
                summary=summary if summary else None
            )

        except requests.exceptions.RequestException as e:
            raise Exception(f"Search request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Search error: {str(e)}")

    def _extract_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            return domain.replace("www.", "")
        except:
            return "Unknown"


class SearchManager:
    """Manages search operations using Tavily provider"""

    def __init__(self, api_key: str, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results
        self.search_history = []  # Store recent searches for caching
        self.tavily_provider = TavilySearchProvider(api_key, max_results)

    async def perform_search(self, query: str, force_refresh: bool = False) -> SearchResponse:
        """
        Perform search with caching

        Args:
            query: Search query
            force_refresh: If True, ignore cache and perform fresh search

        Returns:
            SearchResponse object
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_result = self._get_cached_result(query)
            if cached_result:
                # Check if cache is still valid (less than 1 hour old)
                cached_time = datetime.fromisoformat(cached_result.timestamp)
                if datetime.now() - cached_time < timedelta(hours=1):
                    return cached_result

        # Perform new search
        result = await self.tavily_provider.search(query)

        # Cache the result
        self._cache_result(result)

        return result

    def _get_cached_result(self, query: str) -> Optional[SearchResponse]:
        """Get cached search result if available"""
        for cached in self.search_history:
            if cached.query.lower() == query.lower():
                return cached
        return None

    def _cache_result(self, result: SearchResponse):
        """Cache search result"""
        # Keep only last 20 searches in cache
        if len(self.search_history) >= 20:
            self.search_history.pop(0)

        self.search_history.append(result)

    def format_search_for_context(self, response: SearchResponse, max_chars: int = 2000) -> str:
        """
        Format search results for inclusion in LLM context

        Args:
            response: SearchResponse object
            max_chars: Maximum characters for the formatted context

        Returns:
            Formatted string with search results
        """
        formatted_parts = [f"## Web Search Results for: {response.query}\n"]

        if response.summary:
            formatted_parts.append(f"**Summary:** {response.summary}\n")

        formatted_parts.append("\n**Sources:**\n")

        for i, result in enumerate(response.results, 1):
            source_info = f"{i}. **{result.title}**\n"
            source_info += f"   URL: {result.url}\n"
            source_info += f"   Excerpt: {result.snippet[:200]}{'...' if len(result.snippet) > 200 else ''}\n"
            if result.date:
                source_info += f"   Date: {result.date}\n"
            source_info += f"   Source: {result.source}\n"
            formatted_parts.append(source_info)

        formatted_text = "\n".join(formatted_parts)

        # Truncate if too long
        if len(formatted_text) > max_chars:
            formatted_text = formatted_text[:max_chars] + "..."

        return formatted_text

    def is_search_query(self, message: str) -> bool:
        """
        DEPRECATED: This function is no longer needed.
        With the LLM-driven approach, the AI model decides whether to use search results.
        Always returns False to let the system proceed with search when enabled.
        """
        return False


# Global search manager instance
_search_manager: Optional[SearchManager] = None


def get_search_manager() -> SearchManager:
    """Get or create global search manager instance"""
    global _search_manager
    if _search_manager is None:
        # Load API key from environment
        api_key = os.getenv("tavily_api_key") or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise Exception("Tavily API key not found. Please set tavily_api_key in your environment.")

        _search_manager = SearchManager(api_key)
    return _search_manager