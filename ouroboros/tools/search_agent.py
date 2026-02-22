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
    """Агент для глубокого поиска в интернете."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_search_results: int = 5,
        max_page_length_chars: int = 8000,
        request_delay: float = 1.0,
        verbose: bool = False
    ):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
        )
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("OPENROUTER_MODEL", "gpt-4")
        self.max_search_results = max_search_results
        self.max_page_length_chars = max_page_length_chars
        self.request_delay = request_delay
        self.verbose = verbose

        self.system_prompt = self._build_system_prompt()
        self.tools = self._define_tools()

    def _build_system_prompt(self) -> str:
        return """Ты – SearchAgent, автономный поисковый агент. Твоя задача – находить точную и актуальную информацию в интернете, отвечая на запрос пользователя.

У тебя есть три инструмента:

1. search_web(query: str) -> List[Dict]  
   Выполняет поиск в интернете по строке query.  
   Возвращает список результатов, каждый содержит:
   - title – заголовок страницы,
   - url – ссылка на страницу,
   - snippet – краткое описание (сниппет).

2. read_page(url: str) -> str  
   Загружает страницу по указанному URL и возвращает её текстовое содержание (очищенное от HTML).  
   Если страница слишком большая, она обрезается.

3. finalize_answer(answer: str, sources: List[str])  
   Завершает работу агента и возвращает итоговый ответ answer вместе со списком использованных источников (sources – список URL).  
   Этот инструмент нужно вызвать только когда информация собрана полностью.

Правила работы:
- Получив запрос пользователя, проанализируй его. Если нужно – разбей на подзадачи.
- Для каждой подзадачи сформулируй один или несколько поисковых запросов.
- Вызывай search_web для каждого запроса. Изучай полученные сниппеты.
- Если сниппетов недостаточно, вызывай read_page для наиболее перспективных ссылок.
- После того как собрал достаточно данных (возможно, после нескольких итераций), вызови finalize_answer.
- Не вызывай finalize_answer преждевременно, но и не зацикливайся – старайся уложиться в разумное число шагов.
- В финальном ответе обязательно укажи источники (URL), на которые ты опирался.

ВАЖНО: Если ты собрал достаточно информации для ответа, НЕМЕДЛЕННО вызови finalize_answer. Если после 5-7 итераций информация ещё не собрана – всё равно вызови finalize_answer с тем, что есть, указав что информация неполная.
"""

    def _define_tools(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Выполняет поиск в интернете по запросу и возвращает результаты",
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

    def _duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
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
                print(f"[DEBUG] Ошибка при запросе к DuckDuckGo: {e}")
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
                url = parsed.get("uddg", [""])[0]
            else:
                url = href

            snippet_elem = result.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            results.append({"title": title, "url": url, "snippet": snippet})
        return results

    def search_web(self, query: str) -> List[Dict[str, str]]:
        if self.verbose:
            print(f"[DEBUG] Поиск: {query}")
        return self._duckduckgo_search(query)

    def read_page(self, url: str) -> str:
        if self.verbose:
            print(f"[DEBUG] Чтение страницы: {url}")
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
        if len(text) > self.max_page_length_chars:
            text = text[:self.max_page_length_chars] + "\n...[текст обрезан]"
        return text

    def process_query(self, user_query: str, max_iterations: int = 10) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        iteration = 0
        final_answer = None
        sources = []

        while iteration < max_iterations and final_answer is None:
            iteration += 1
            if self.verbose:
                print(f"\n[DEBUG] Итерация {iteration}")

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
                    print(f"[DEBUG] Ошибка LLM: {e}")
                final_answer = f"Ошибка при обращении к модели: {e}"
                sources = []
                break

            if msg.tool_calls:
                if self.verbose:
                    print(f"[DEBUG] Вызов инструментов: {[tc.function.name for tc in msg.tool_calls]}")
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if func_name == "search_web":
                        query = args.get("query", "")
                        if query:
                            results = self.search_web(query)
                            result_str = json.dumps(results, ensure_ascii=False, indent=2)
                        else:
                            result_str = "Ошибка: пустой запрос"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str
                        })

                    elif func_name == "read_page":
                        url = args.get("url", "")
                        if url:
                            page_text = self.read_page(url)
                            if len(page_text) > 10000:
                                page_text = page_text[:10000] + "\n...[текст обрезан]"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": page_text
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Ошибка: пустой URL"
                            })

                    elif func_name == "finalize_answer":
                        final_answer = args.get("answer", "")
                        sources = args.get("sources", [])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Поиск завершён."
                        })
                        break  # завершаем цикл

                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Ошибка: неизвестный инструмент {func_name}"
                        })
            else:
                if self.verbose:
                    print("[DEBUG] Модель ответила текстом без вызова инструментов")
                messages.append(msg)
                final_answer = msg.content or "(пустой ответ)"
                sources = ["Ответ сгенерирован без явных источников"]

        if final_answer is None:
            final_answer = "Не удалось получить ответ за допустимое число итераций. Возможно, информация не найдена или модель не смогла обработать запрос."
            sources = []

        return {
            "answer": final_answer,
            "sources": sources,
            "iterations": iteration
        }


# Интеграция в систему Ouroboros
def search_agent_tool(query: str) -> str:
    """
    Инструмент-обёртка для вызова SearchAgent из Ouroboros.
    Возвращает JSON строку с полями 'answer' и 'sources'.
    """
    try:
        agent = SearchAgent(verbose=False)
        result = agent.process_query(query)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "answer": "", "sources": []}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            "search_agent",
            {
                "name": "search_agent",
                "description": "Автономный поисковый агент на основе LLM. Выполняет глубокий поиск, анализирует страницы и возвращает ответ с источниками. Использует DuckDuckGo, не требует API ключей.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Поисковый запрос"}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                },
            },
            search_agent_tool
        )
    ]