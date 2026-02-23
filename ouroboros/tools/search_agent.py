def get_tools() -> List[ToolEntry]:
    """Return ToolEntry list for auto-discovery."""
    return [
        ToolEntry(
            name="search_agent",
            schema={
                "name": "search_agent",
                "description": "Поиск в интернете через автономного агента (DuckDuckGo + LLM)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Поисковый запрос"},
                        "max_iterations": {"type": "integer", "description": "Максимальное число итераций агента (по умолчанию 10)"}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                },
                "strict": True
            },
            handler=search_agent_tool
        )
    ]