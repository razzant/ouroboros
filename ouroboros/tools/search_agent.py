"""
SearchAgent - Autonomous search agent using LLM with function calling.

Performs deep web searches via DuckDuckGo (HTML parsing, no API keys required),
reads pages with BeautifulSoup, and compiles synthesized answers with sources.
Uses OpenRouter credentials from environment.
"""

from __future__ import annotations

import os
import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

from ouroboros.tools.registry import ToolEntry

log = logging.getLogger(__name__)
load_dotenv()


class SearchAgent:
    """Internal agent that performs search and reads pages via DuckDuckGo."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_results: int = 10,
        request_delay: float = 1.0,
        max_iterations: int = 15,
        max_page_length: int = 8000,
        verbose: bool = False,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = model or os.getenv("OPENROUTER_MODEL", "qwen/qwen3-vl-235b-a22b-thinking")
        self.max_results = max_results
        self.request_delay = request_delay
        self.max_iterations = max_iterations
        self.max_page_length = max_page_length
        self.verbose = verbose

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.tools_schema = self._define_tools()

    def _define_tools(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web via DuckDuckGo (HTML). Returns list of results: title, url, snippet.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "Read the full text content of a webpage (URL).",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "description": "URL to read"}},
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_answer",
                    "description": "Finish the search and provide the final answer with sources.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Final synthesized answer"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of source URLs used",
                            },
                        },
                        "required": ["answer", "sources"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """Perform DuckDuckGo HTML search and return results."""
        time.sleep(self.request_delay)
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            log.warning("DuckDuckGo search failed: %s", e)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results: List[Dict[str, str]] = []
        for result in soup.select(".result")[: self.max_results]:
            title_elem = result.select_one(".result__a")
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            href = title_elem.get("href", "")
            # Extract real URL from uddg parameter
            if href.startswith("/l/?kh=-1&uddg="):
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                url_real = parsed.get("uddg", [""])[0]
            else:
                url_real = href

            snippet_elem = result.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            results.append({"title": title, "url": url_real, "snippet": snippet})
        return results

    def read_page(self, url: str) -> str:
        """Fetch and extract text from a webpage."""
        time.sleep(self.request_delay)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            return f"[Error loading page: {e}]"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > self.max_page_length:
            text = text[: self.max_page_length] + "\n...[text truncated]"
        return text

    # -------------------------------------------------------------------------
    # Main agent loop
    # -------------------------------------------------------------------------

    def run(self, query: str) -> Dict[str, Any]:
        """Run the agentic search loop for a given query."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": query},
        ]

        final_answer: Optional[str] = None
        sources: List[str] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            if self.verbose:
                log.info("SearchAgent iteration %d/%d", iterations, self.max_iterations)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                )
            except Exception as e:
                log.error("LLM call failed: %s", e)
                final_answer = f"LLM error: {e}"
                break

            msg = response.choices[0].message
            if not msg.tool_calls:
                # Model responded without tools — treat as answer (unexpected)
                final_answer = msg.content or "No answer produced."
                break

            # Add assistant message with tool calls
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

            # Execute each tool call
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result_str: str

                if name == "search_web":
                    results = self.search_web(args["query"])
                    result_str = json.dumps(results, ensure_ascii=False)
                elif name == "read_page":
                    result_str = self.read_page(args["url"])
                elif name == "finalize_answer":
                    final_answer = args["answer"]
                    sources = args.get("sources", [])
                    result_str = "Final answer recorded."
                    break  # Exit tool loop
                else:
                    result_str = f"Unknown tool: {name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

            if final_answer is not None:
                break

        if final_answer is None:
            final_answer = "Search did not produce an answer within iteration limit."
            sources = []

        return {"answer": final_answer, "sources": sources, "iterations": iterations}

    def _system_prompt(self) -> str:
        return (
            "You are an autonomous search agent. Your task: answer the user's query by searching the web and reading pages.\n"
            "Use the provided tools:\n"
            "- search_web(query): returns a list of results (title, url, snippet)\n"
            "- read_page(url): returns the full text of the page\n"
            "- finalize_answer(answer, sources): call when you have the answer\n\n"
            "Work iteratively. If snippets are insufficient, read the most relevant pages. "
            "Call finalize_answer with a concise, accurate answer and a list of source URLs. "
            "Do not exceed the maximum iterations."
        )


# -------------------------------------------------------------------------
# Tool registration
# -------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    """Return SearchAgent tool entry for the registry."""
    return [
        ToolEntry(
            name="search_agent",
            schema={
                "name": "search_agent",
                "description": "Autonomous web search using DuckDuckGo + LLM agent. Returns answer + sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Max agent iterations (default 15)",
                        },
                    },
                    "required": ["query"],
                },
            },
            handler=lambda ctx, **kwargs: SearchAgent(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-vl-235b-a22b-thinking"),
                verbose=kwargs.get("verbose", False),
            ).run(kwargs["query"]),
            is_code_tool=False,
            timeout_sec=180,
        )
    ]
