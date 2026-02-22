"""SearchAgent - Автономный поисковый агент на основе LLM с function calling.
Агент получает запрос, самостоятельно формирует поисковые запросы,
анализирует сниппеты, при необходимости читает страницы,
и возвращает итоговый ответ со ссылками на источники.

Поиск выполняется через DuckDuckGo (HTML-парсинг) – не требует ключей.
Чтение страниц – через requests + BeautifulSoup.
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
    """Агент для глубокого поиска в интернете с использованием LLM и function calling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_search_results: int = 5,
        max_page_length: int = 8000,
        request_delay: float = 1.0,
        verbose: bool = False
    ):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        self.model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-8b-instruct:free")
        self.max_search_results = max_search_results
        self.max_page_length = max_page_length
        self.request_delay = request_delay
        self.verbose = verbose

        self.system_prompt = self._build_system_prompt()
        self.tools = self._define_tools()

    def _build_system_prompt(self) -> str:
        return """Ты — SearchAgent, автономный поисковый агент. Твоя задача — находить точную и актуальную информацию в интернете, отвечая на запрос пользователя.

У тебя есть инструменты:

1. search_web(query: str) -> List[Dict] — выполняет поиск в интернете. Возвращает список результатов с полями: title, url, snippet.

2. read_page(url: str) -> str — загружает страницу по URL и возвращает её текстовое содержание (очищенное от HTML). Большие страницы обрезаются.

3. finalize_answer(answer: str, sources: List[str]) — завершает поиск и возвращает итоговый ответ. Вызывай только когда информация собрана достаточно.

ПРАВИЛА РАБОТЫ:
- При получении запроса подумай, какие поисковые запросы нужны.
- Вызывай search_web для каждого запроса и анализируй результаты.
- Если сниппеты недостаточны, вызывай read_page для перспективных ссылок.
- После сбора данных (обычно 2-4 поиска и чтении 1-3 страниц) вызови finalize_answer.
- НЕ ПЕРЕПОЛНЯЙ ИСТОЧНИКИ: выбери 3-5 самых релевантных URL.
- ВСЕГДА вызывай finalize_answer, даже если информация неполная — верни лучший возможный ответ.
- Если произошла ошибка (например, страница недоступна), все равно продолжи и finalize_answer в конце.

ВАЖНО: Ты должен ВСЕГДА завершить поиск вызовом finalize_answer. Не зацикливайся.

Цель: максимально полный и точный ответ на исходный запрос."""

    def _define_tools(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Выполняет поиск в интернете по запросу и возвращает результаты с заголовками, URL и сниппетами",
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
                    "description": "Загружает страницу по URL и возвращает её текстовое содержание",
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
                    "description": "Завершает поиск и возвращает финальный ответ",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string", "description": "Итоговый ответ на запрос пользователя"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Список использованных URL (3-5 самых релевантных)"
                            }
                        },
                        "required": ["answer", "sources"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

    def _duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            time.sleep(self.request_delay)
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] DuckDuckGo error: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:self.max_search_results]:
            title_elem = result.select_one(".result__a")
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
            snippet_elem = result.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            results.append({"title": title, "url": real_url, "snippet": snippet})
        return results

    def search_web(self, query: str) -> List[Dict[str, str]]:
        if self.verbose:
            print(f"[DEBUG] search_web: {query}")
        return self._duckduckgo_search(query)

    def read_page(self, url: str) -> str:
        if self.verbose:
            print(f"[DEBUG] read_page: {url}")
        time.sleep(self.request_delay)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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

    def process_query(self, user_query: str, max_iterations: int = 8) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        iteration = 0
        final_answer = None
        sources = []
        tool_call_count = 0
        max_tool_calls = 10

        while iteration < max_iterations and final_answer is None:
            iteration += 1
            if self.verbose:
                print(f"[DEBUG] Итерация {iteration}")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
                msg = response.choices[0].message
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] LLM error: {e}")
                break

            if msg.tool_calls:
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    fname = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    tool_call_count += 1
                    
                    if fname == "search_web":
                        results = self.search_web(args["query"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(results, ensure_ascii=False)
                        })
                    elif fname == "read_page":
                        content = self.read_page(args["url"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content[:15000]
                        })
                    elif fname == "finalize_answer":
                        final_answer = args["answer"]
                        sources = args.get("sources", [])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Поиск завершён"
                        })
                        break
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Ошибка: неизвестный инструмент {fname}"
                        })
                    
                    if tool_call_count >= max_tool_calls and final_answer is None:
                        if self.verbose:
                            print(f"[DEBUG] Достигнут лимит инструментов ({max_tool_calls}), принудительное завершение")
                        messages.append({
                            "role": "user",
                            "content": "Достигнут лимит поисковых запросов. Пожалуйста, заверши поиск и верни финальный ответ на основе собранной информации, даже если она неполная. Используй инструмент finalize_answer."
                        })
            else:
                if self.verbose:
                    print("[DEBUG] Модель не вызвала инструменты, завершаем")
                if msg.content:
                    final_answer = msg.content
                    sources = []
                break

        if final_answer is None:
            if self.verbose:
                print("[DEBUG] Попытка получить финальный ответ после цикла")
            try:
                messages.append({
                    "role": "user",
                    "content": "Пожалуйста, заверши поиск и верни финальный ответ на основе всей собранной информации. Используй инструмент finalize_answer."
                })
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice={"type": "function", "function": {"name": "finalize_answer"}}
                )
                msg = response.choices[0].message
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == "finalize_answer":
                            args = json.loads(tool_call.function.arguments)
                            final_answer = args["answer"]
                            sources = args.get("sources", [])
                            break
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] Ошибка при финализации: {e}")
            
            if final_answer is None:
                final_answer = "Не удалось получить ответ за разрешённое число итераций."
                sources = []

        return {"answer": final_answer, "sources": sources, "iterations": iteration}


def search_agent_tool(query: str) -> str:
    """Обработчик инструмента для вызова из системы Ouroboros."""
    agent = SearchAgent(verbose=False)
    result = agent.process_query(query, max_iterations=6)
    return json.dumps(result, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "search_agent",
            {
                "name": "search_agent",
                "description": "Интеллектуальный поиск в интернете с агентным анализом. Принимает запрос, сам решает что искать, читает страницы, возвращает ответ с источниками.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Поисковый запрос"}
                    },
                    "required": ["query"]
                }
            },
            search_agent_tool
        )
    ]