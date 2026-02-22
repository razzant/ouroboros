"""SearchAgent - Autonomous search agent using LLM with function calling.

No external search API required. Uses DuckDuckGo HTML parsing and BeautifulSoup.
"""

import os
import time
import json
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

from .registry import ToolEntry

load_dotenv()


class SearchAgent:
    """Autonomous agent that searches the web, reads pages, and compiles answers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 10,
        max_search_results: int = 5,
        max_page_length: int = 8000,
        request_delay: float = 1.0,
        verbose: bool = False
    ):
        # OpenAI-compatible client
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        )
        self.model = model or os.getenv("OPENROUTER_MODEL") or os.getenv("OUROBOROS_MODEL_LIGHT") or "gpt-3.5-turbo"
        self.max_iterations = max_iterations
        self.max_search_results = max_search_results
        self.max_page_length = max_page_length
        self.request_delay = request_delay
        self.verbose = verbose

        # System prompt
        self.system_prompt = self._build_system_prompt()

        # Tool schemas for function calling
        self.tools_schema = self._define_tools()

    def _build_system_prompt(self) -> str:
        return """Ты — автономный поисковый агент. Твоя задача — найти и предоставить точный ответ на запрос пользователя, используя доступные инструменты.

Инструменты:
1. search_web(query: str) — выполняет поиск в интернете. Возвращает список результатов с title, url, snippet.
2. read_page(url: str) — загружает и читает содержимое страницы.
3. finalize_answer(answer: str, sources: List[str]) — завершает поиск и возвращает финальный ответ со списком источников.

Алгоритм:
- Анализируй запрос. При необходимости делай несколько поисковых запросов.
- Изучай сниппеты. Если информации недостаточно — читай полные страницы.
- После сбора достаточных данных вызови finalize_answer.
- Не злоупотребляй количеством итераций. Вызывай finalize_answer, когда ответ готов.
"""

    def _define_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Выполняет поиск в интернете по запросу, возвращает список результатов с заголовками, ссылками и сниппетами",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Поисковый запрос"}
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "Загружает полное содержимое страницы по URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL страницы для загрузки"}
                        },
                        "required": ["url"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_answer",
                    "description": "Завершает поиск и возвращает финальный ответ с указанием источников",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Финальный ответ на запрос пользователя"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список URL использованных источников"
                            }
                        },
                        "required": ["answer", "sources"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

    # -------------------------------------------------------------------------
    # Search engine implementation (DuckDuckGo HTML parsing)
    # -------------------------------------------------------------------------

    def _duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        """Perform search using DuckDuckGo HTML interface (no API key needed)."""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            time.sleep(self.request_delay)
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] DuckDuckGo search error: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for result in soup.select(".result")[:self.max_search_results]:
            title_elem = result.select_one(".result__title .result__a")
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            href = title_elem.get("href", "")

            # Extract real URL from DuckDuckGo redirect
            if href.startswith("/l/?kh=-1&uddg="):
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                real_url = parsed.get("uddg", [""])[0]
            else:
                real_url = href

            snippet_elem = result.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            if real_url:
                results.append({
                    "title": title,
                    "url": real_url,
                    "snippet": snippet
                })

        return results

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """Public search method."""
        if self.verbose:
            print(f"[DEBUG] search_web: {query}")
        return self._duckduckgo_search(query)

    def read_page(self, url: str) -> str:
        """Download and extract text from a webpage."""
        if self.verbose:
            print(f"[DEBUG] read_page: {url}")

        time.sleep(self.request_delay)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            return f"[Ошибка загрузки страницы: {e}]"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > self.max_page_length:
            text = text[:self.max_page_length] + "\n...[обрезано]"
        return text

    # -------------------------------------------------------------------------
    # Main agent loop
    # -------------------------------------------------------------------------

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Run the agent loop to answer a user query."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        final_answer = None
        sources = []
        iteration = 0

        while iteration < self.max_iterations and final_answer is None:
            iteration += 1
            if self.verbose:
                print(f"[DEBUG] Iteration {iteration}")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    max_tokens=2000  # stay within free tier limits
                )
                msg = response.choices[0].message
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] API error: {e}")
                break

            if msg.tool_calls:
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if func_name == "search_web":
                        query = args.get("query", "")
                        results = self.search_web(query)
                        result_str = json.dumps(results, ensure_ascii=False)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str
                        })

                    elif func_name == "read_page":
                        url = args.get("url", "")
                        page_text = self.read_page(url)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": page_text
                        })

                    elif func_name == "finalize_answer":
                        final_answer = args.get("answer", "")
                        sources = args.get("sources", [])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Answer finalized"
                        })
                        break

                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Unknown tool: {func_name}"
                        })
            else:
                # Model responded without tools; finish with its content
                final_answer = msg.content or "No answer generated"
                sources = []
                break

        # Force completion if we reached max_iterations without finalize_answer
        if final_answer is None and iteration >= self.max_iterations:
            # Ask the model to produce an answer based on the conversation
            try:
                summary_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Синтезируй ответ на основе собранной информации. Верни JSON с ключами 'answer' (строка) и 'sources' (массив URL)."},
                        {"role": "user", "content": user_query}
                    ] + messages[-5:],  # include recent context
                    max_tokens=2000
                )
                summary_text = summary_resp.choices[0].message.content or "Превышено число итераций. Ответ не собран."
                final_answer = summary_text
                sources = []
            except Exception as e:
                final_answer = f"Превышено число итераций. Ответ не собран. Ошибка: {e}"
                sources = []

        return {
            "answer": final_answer,
            "sources": sources,
            "iterations": iteration
        }


# -------------------------------------------------------------------------
# Tool registration
# -------------------------------------------------------------------------

def search_agent_tool(query: str, max_iterations: int = 10) -> str:
    """Tool wrapper for SearchAgent: return JSON with answer, sources, iterations."""
    try:
        agent = SearchAgent(max_iterations=max_iterations, verbose=False)
        result = agent.process_query(query)
        return json.dumps({
            "answer": result["answer"],
            "sources": result["sources"],
            "iterations": result["iterations"]
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": str(e)
        }, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    """Return SearchAgent tool descriptor."""
    return [
        ToolEntry(
            name="search_agent",
            schema={
                "name": "search_agent",
                "description": "Поиск в интернете с глубоким анализом. Возвращает JSON с answer, sources, iterations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Поисковый запрос"}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            },
            handler=search_agent_tool
        )
    ]
