"""SearchAgent - автономный поисковый агент с DuckDuckGo + чтением страниц."""

from .registry import ToolEntry


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
        }, lambda query: "{\"error\": \"Not implemented yet\"}")
    ]