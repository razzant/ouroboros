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
        """
        Инициализация агента.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("OPENROUTER_MODEL") or "StepFun/Step-3.5-Flash:free"
        self.max_search_results = max_search_results
        self.max_page_length_chars = max_page_length_chars
        self.request_delay = request_delay
        self.verbose = verbose

        if not self.api_key:
            raise ValueError("OpenAI/OpenRouter API key not found in environment (OPENAI_API_KEY or OPENROUTER_API_KEY)")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.system_prompt = self._build_system_prompt()
        self.tools = self._define_tools()

    def _build_system_prompt(self) -> str:
        return """Ты – SearchAgent, автономный поисковый агент. Твоя задача – находить точную и актуальную информацию в интернете, отвечая на запрос пользователя.

У тебя есть два инструмента:

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
   Этот инструмент нужно вызвать только когда информация собрана достаточно для полного ответа.

Правила работы:
- Получив запрос пользователя, проанализируй его. Если нужно – разбей на подзадачи.
- Для каждой подзадачи сформулируй один или несколько поисковых запросов.
- Вызывай search_web для каждого запроса. Изучай полученные сниппеты.
- Если сниппетов недостаточно, вызывай read_page для наиболее перспективных ссылок.
- После того как собрал достаточно данных (возможно, после нескольких итераций), вызови finalize_answer.
- Не вызывай finalize_answer преждевременно, но и не зацикливайся – старайся уложиться в разумное число шагов.
- В финальном ответе обязательно укажи источники (URL), на которые ты опирался.
- Если по запросу не удаётся найти информацию, всё равно вызови finalize_answer с объяснением, что ничего найдено не было.
- Ты можешь вызвать finalize_answer в любой момент, когда считаешь ответ готовым.

Важно: используй инструменты последовательно, анализируй результаты каждого шага перед следующим.
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
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос"
                            }
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
                            "url": {
                                "type": "string",
                                "description": "URL страницы для загрузки"
                            }
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
                            "answer": {
                                "type": "string",
                                "description": "Финальный ответ на запрос пользователя"
                            },
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
        # Каждый результат находится в <div class="result">
        for result in soup.select(".result")[:self.max_search_results]:
            title_elem = result.select_one(".result__a")
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            # Ссылка может быть в атрибуте href и содержать перенаправление
            href = title_elem.get("href", "")
            # Извлекаем реальный URL из параметра uddg
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
        # Удаляем скрипты, стили, навигацию
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Обрезаем, если слишком длинно
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
                    print(f"[DEBUG] Ошибка вызова модели: {e}")
                final_answer = f"Ошибка вызова модели: {e}"
                sources = []
                break

            # Если модель хочет вызвать инструменты
            if msg.tool_calls:
                if self.verbose:
                    print(f"[DEBUG] Вызов инструментов: {[tc.function.name for tc in msg.tool_calls]}")
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"[DEBUG] Ошибка парсинга аргументов: {e}")
                        result_content = f"Ошибка парсинга аргументов: {e}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content
                        })
                        continue

                    if func_name == "search_web":
                        query = args.get("query", "")
                        if not query:
                            result_content = "Ошибка: поисковый запрос пустой"
                        else:
                            results = self.search_web(query)
                            result_content = json.dumps(results, ensure_ascii=False, indent=2)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content
                        })

                    elif func_name == "read_page":
                        url = args.get("url", "")
                        if not url:
                            result_content = "Ошибка: URL не указан"
                        else:
                            page_text = self.read_page(url)
                            # Обрезаем, если слишком длинно (на всякий случай)
                            if len(page_text) > 10000:
                                page_text = page_text[:10000] + "\n...[текст обрезан]"
                            result_content = page_text
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content
                        })

                    elif func_name == "finalize_answer":
                        final_answer = args.get("answer", "")
                        sources = args.get("sources", [])
                        # Добавляем результат вызова (можно просто подтверждение)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Поиск завершён, ответ сохранён."
                        })
                        break  # можно выйти из цикла, но лучше дать модели завершить

                    else:
                        # Неизвестный инструмент
                        result_content = f"Ошибка: неизвестный инструмент {func_name}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_content
                        })
            else:
                # Модель не вызвала инструменты – возможно, она хочет завершить без поиска? 
                # Проверим, есть ли ответ. Если да, считаем его финальным.
                if msg.content:
                    if self.verbose:
                        print("[DEBUG] Модель ответила текстом без вызова инструментов – считаем финальным ответом")
                    final_answer = msg.content
                    sources = ["Ответ сгенерирован без явных источников (модель не использовала поиск)"]
                    break
                else:
                    # Если нет ни工具, ни текста, возможно, модель ничего не сказала. Продолжим?
                    if self.verbose:
                        print("[DEBUG] Модель не ответила и не вызвала инструменты – продолжаем")
                    messages.append(msg)

        if final_answer is None:
            # Принудительное завершение: берем последний ответ модели, если есть
            if messages and messages[-1]["role"] == "assistant" and messages[-1].get("content"):
                final_answer = messages[-1]["content"] + "\n[Принудительное завершение: достигнут лимит итераций]"
                sources = ["Лимит итераций исчерпан, ответ взят из последнего сообщения модели"]
            else:
                final_answer = "Не удалось получить ответ за допустимое число итераций. Возможно, модель не смогла обработать запрос."
                sources = []

        return {
            "answer": final_answer,
            "sources": sources,
            "iterations": iteration
        }


# -----------------------------------------------------------------------------
# Интеграция в систему Ouroboros
# -----------------------------------------------------------------------------

def search_agent_tool(query: str) -> Dict[str, Any]:
    """
    Обёртка для вызова SearchAgent как инструмента Ouroboros.
    Возвращает JSON с ответом и источниками.
    """
    try:
        agent = SearchAgent(verbose=False)
        result = agent.process_query(query)
        return result
    except Exception as e:
        return {
            "answer": f"Ошибка при выполнении поиска: {str(e)}",
            "sources": [],
            "iterations": 0,
            "error": True
        }


def get_tools() -> List[ToolEntry]:
    """
    Возвращает список инструментов для регистрации в Ouroboros.
    """
    return [
        ToolEntry(
            name="search_agent",
            description="Автономный поисковый агент на основе LLM. Принимает запрос, самостоятельно ищет в интернете, читает страницы и возвращает ответ с источниками. Использует DuckDuckGo, не требует API ключей для поиска. Для работы нужен OPENROUTER_API_KEY и OPENROUTER_MODEL.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос пользователя"
                    }
                },
                "required": ["query"]
            },
            handler=lambda params: json.dumps(search_agent_tool(params["query"]), ensure_ascii=False)
        )
    ]


if __name__ == "__main__":
    # Пример использования
    agent = SearchAgent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=os.getenv("OPENROUTER_MODEL", "StepFun/Step-3.5-Flash:free"),
        verbose=True
    )

    query = "Какие последние новости в области open source больших языковых моделей?"
    result = agent.process_query(query)

    print("\n" + "="*60)
    print("ОТВЕТ:")
    print(result["answer"])
    print("\nИСТОЧНИКИ:")
    for src in result["sources"]:
        print(f"- {src}")
    print(f"\nИтераций: {result['iterations']}")
