"""SearchAgent - Автономный поисковый агент на основе LLM с function calling.
Агент получает запрос, самостоятельно формирует поисковые запросы,
анализирует сниппеты, читает нужные страницы и возвращает ответ со ссылками.
Поиск через DuckDuckGo (не требует ключей), чтение страниц через requests+BeautifulSoup.
"""

import os
import time
import json
import requests
from typing import List, Dict, Any
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv

from .registry import ToolEntry

load_dotenv()


class SearchAgent:
    """Автономный поисковый агент."""

    def __init__(self, max_iterations: int = 15, verbose: bool = False):
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        self.model = os.getenv("OPENROUTER_MODEL", "StepFun/Step-3.5-Flash:free")
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.request_delay = 1.0
        self.max_search_results = 5
        self.max_page_length_chars = 8000
        self.tools = self._get_schema()

    def _get_schema(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Выполняет поиск в интернете, возвращает список результатов с заголовками, URL и сниппетами.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "Загружает страницу по URL и возвращает её текстовое содержание.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_answer",
                    "description": "Завершает поиск и возвращает финальный ответ с источниками.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "sources": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["answer", "sources"]
                    }
                }
            }
        ]

    def _duckduckgo_search(self, query: str):
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            time.sleep(self.request_delay)
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            if self.verbose:
                print(f"[search_web] error: {e}")
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for res in soup.select(".result")[:self.max_search_results]:
            title_elem = res.select_one(".result__a")
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            href = title_elem.get("href", "")
            if href.startswith("/l/?kh=-1&uddg="):
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                real_url = parsed.get("uddg", [""])[0]
            else:
                real_url = href
            snippet_elem = res.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            results.append({"title": title, "url": real_url, "snippet": snippet})
        return results

    def read_page(self, url: str) -> str:
        if self.verbose:
            print(f"[read_page] {url}")
        time.sleep(self.request_delay)
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            return f"[Ошибка загрузки: {e}]"
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > self.max_page_length_chars:
            text = text[:self.max_page_length_chars] + "\n...[обрезано]"
        return text

    def process_query(self, user_query: str) -> Dict[str, Any]:
        system_prompt = """Ты — автономный поисковый агент. Используй инструменты search_web, read_page и finalize_answer. Собирай информацию и вызови finalize_answer с готовым ответом и URL источников."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        final_answer = None
        sources = []
        iterations = 0

        for it in range(1, self.max_iterations + 1):
            iterations = it
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=2000
                )
            except Exception as e:
                if self.verbose:
                    print(f"[API error] {e}")
                break

            msg = resp.choices[0].message
            if not msg.tool_calls:
                if self.verbose:
                    print("[no tool call, using text response]")
                final_answer = msg.content
                sources = []
                break

            messages.append(msg)
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                if name == "search_web":
                    results = self._duckduckgo_search(args["query"])
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(results, ensure_ascii=False)})
                elif name == "read_page":
                    content = self.read_page(args["url"])
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                elif name == "finalize_answer":
                    final_answer = args["answer"]
                    sources = args.get("sources", [])
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": "OK"})
                    break
                else:
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": f"Unknown tool: {name}"})
            if final_answer is not None:
                break

        if final_answer is None:
            # ask for final answer explicitly
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [{"role": "system", "content": "Сейчас ты должен завершить поиск. Используй инструмент finalize_answer."}],
                    tools=[t for t in self.tools if t["function"]["name"] == "finalize_answer"],
                    tool_choice={"type": "function", "function": {"name": "finalize_answer"}},
                    max_tokens=2000
                )
                msg = resp.choices[0].message
                if msg.tool_calls:
                    args = json.loads(msg.tool_calls[0].function.arguments)
                    final_answer = args["answer"]
                    sources = args.get("sources", [])
                else:
                    final_answer = msg.content or "Не удалось получить ответ."
                    sources = []
            except Exception as e:
                if self.verbose:
                    print(f"[force completion failed] {e}")
                final_answer = "Превышено число итераций. Ответ не собран."
                sources = []

        return {"answer": final_answer, "sources": sources, "iterations": iterations}

    def __call__(self, query: str) -> str:
        result = self.process_query(query)
        return result["answer"]


def _search_agent_tool(query: str) -> str:
    try:
        agent = SearchAgent(verbose=False)
        result = agent.process_query(query)
        return json.dumps({
            "answer": result["answer"],
            "sources": result["sources"],
            "iterations": result["iterations"]
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_tools():
    return [
        ToolEntry("search_agent", {
            "name": "search_agent",
            "description": "Агент для глубокого поиска в интернете (DuckDuckGo + чтение страниц). Возвращает ответ со списком источников.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос на естественном языке"}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }, _search_agent_tool)
    ]