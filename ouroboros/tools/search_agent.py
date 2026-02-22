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

load_dotenv()


class SearchAgent:
    """
    Агент для глубокого поиска в интернете.
    Использует LLM с function calling для планирования и синтеза информации.
    """

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
        """
        Инициализация агента.

        :param api_key: API ключ для OpenAI-совместимой модели (OpenRouter/z.ai)
        :param base_url: Базовый URL API (например, https://openrouter.ai/api/v1)
        :param model: Название модели (должна поддерживать function calling)
        :param max_search_results: Максимальное количество результатов на один поиск
        :param max_page_length_chars: Максимальная длина текста страницы (в символах)
        :param request_delay: Задержка между HTTP-запросами (сек)
        :param verbose: Подробный вывод отладки
        """
        self.client = openai.OpenAI(
            api_key=api_key),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        # Используем бесплатную модель по умолчанию, если не задана
        self.model = model or os.getenv("OUROBOROS_MODEL", "arcee-ai/trinity-large-preview:free")
        self.max_search_results = max_search_results
        self.max_page_length_chars = max_page_length_chars
        self.request_delay = request_delay
        self.verbose = verbose

        # Системный промпт
        self.system_prompt = self._build_system_prompt()

        # Инструменты для function calling
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

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

- ОГРАНИЧЕНИЕ ИТЕРАЦИЙ: максимум 3 поисковых запроса (search_web) и максимум 2 прочитанные страницы (read_page). После этого ты ОБЯЗАН вызвать finalize_answer.
- НЕ ЗАЦИКЛИВАЙСЯ: если поиск даёт плохие результаты или не возвращает нужной информации, не повторяй те же запросы. Собери что есть, и вызови finalize_answer.
- ПРЕЖДЕВРЕМЕННОЕ ЗАВЕРШЕНИЕ: если уже после первого или второго поиска информация кажется достаточной, немедленно вызови finalize_answer.
- НЕ ПЫТАЙСЯ СОВЕРШЕНСТВОВАТЬ: твоя цель – дать хороший ответ, а не идеальный. Лучше завершить, чем бесконечно искать.
- ВЫЗОВ FINALIZE_ANSWER – это единственный способ вернуть результат. Без этого вызова пользователь не получит ответ.

Алгоритм:
1. Проанализируй запрос. При необходимости разбей на подзадачи.
2. Выполни 1-3 поисковых запроса через search_web.
3. Изучи сниппеты. Если нужно deeper – прочти до 2 страниц через read_page.
4. Как только собрал достаточно информации (или исчерпал лимит) – вызови finalize_answer с полным ответом и списком источников.

Помни: ты должен дать максимально полный и точный ответ на исходный запрос пользователя, но в РАЗУМНЫЕ СРОКИ.
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
                            "query": {"type": "string"}
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
                            "url": {"type": "string"}
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
                            "answer": {"type": "string"},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string"}
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
    # Инструменты для работы с интернетом
    # -------------------------------------------------------------------------

    def _duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        """
        Парсинг DuckDuckGo (HTML-версия) для получения результатов поиска.
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            time.sleep(self.request_delay)  # вежливость
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

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })
        return results

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """
        Публичный метод поиска.
        """
        if self.verbose:
            print(f"[DEBUG] Поиск: {query}")
        return self._duckduckgo_search(query)

    def read_page(self, url: str) -> str:
        """
        Загружает страницу, очищает от HTML и возвращает текст.
        """
        if self.verbose:
            print(f"[DEBUG] Чтение страницы: {url}")
        time.sleep(self.request_delay)  # вежливость

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            return f"[Ошибка загрузки страницы: {e}]"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "navheader", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > self.max_page_length_chars:
            text = text[:self.max_page_length_chars] + "\n...[текст обрезан]"
        return text

    # -------------------------------------------------------------------------
    # Основной цикл обработки запроса
    # -------------------------------------------------------------------------

    def process_query(self, user_query: str, max_iterations: int = 15) -> Dict[str, Any]:
        """
        Запускает агента для обработки запроса пользователя.
        Возвращает словарь с ключами 'answer' и 'sources'.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        iteration = 0
        final_answer = None
        sources = []
        search_calls = 0
        read_calls = 0

        while iteration < max_iterations and final_answer is None:
            iteration += 1
            if self.verbose:
                print(f"\n[DEBUG] Итерация {iteration}")

            # Вызов LLM с инструментами
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                if self.verbose:
                    print(f"[DEBUG] Вызов инструментов: {[tc.function.name for tc in msg.tool_calls]}")
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if func_name == "search_web":
                        search_calls += 1
                        if search_calls > 3:
                            # Превысили лимит поисков, принудительно завершаем
                            result_str = "Лимит поисковых запросов исчерпан. Пожалуйста, вызови finalize_answer."
                        else:
                            query = args["query"]
                            results = self.search_web(query)
                            result_str = json.dumps(results, ensure_ascii=False, indent=2)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str
                        })

                    elif func_name == "read_page":
                        read_calls += 1
                        if read_calls > 2:
                            result_str = "Лимит чтения страниц исчерпан. Пожалуйста, вызови finalize_answer."
                        else:
                            url = args["url"]
                            page_text = self.read_page(url)
                            if len(page_text) > 10000:
                                page_text = page_text[:10000] + "\n...[текст обрезан]"
                            result_str = page_text
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str
                        })

                    elif func_name == "finalize_answer":
                        final_answer = args["answer"]
                        sources = args.get("sources", [])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Поиск завершён, ответ сохранён."
                        })
                        break

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
                final_answer = msg.content
                sources = ["Ответ сгенерирован без явных источников"]

        if final_answer is None:
            final_answer = "Не удалось получить ответ за допустимое число итераций. Возможно, информации по запросу недостаточно или поиск не дал результатов."
            sources = []

        return {
            "answer": final_answer,
            "sources": sources,
            "iterations": iteration
        }


def search_agent_tool(query: str) -> str:
    """
    Инструмент-обёртка для использования SearchAgent как инструмента Ouroboros.
    """
    try:
        agent = SearchAgent(verbose=False)
        result = agent.process_query(query)
        return json.dumps({
            "answer": result["answer"],
            "sources": result["sources"],
            "iterations": result["iterations"]
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Интеграция с системой инструментов Ouroboros
# -----------------------------------------------------------------------------

from ouroboros.tools.registry import ToolContext, ToolEntry


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("search_agent", {
            "name": "search_agent",
            "description": "Интеллектуальный поиск в интернете с агентным анализом. Принимает запрос, сам решает что искать, читает страницы, возвращает ответ с источниками.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    }
                },
                "required": ["query"]
            },
            "handler": search_agent_tool
        })
    ]