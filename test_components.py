#!/usr/bin/env python3
"""Тест отдельных компонентов SearchAgent без LLM цикла."""
import os
import sys
sys.path.insert(0, '/home/ivan/.ouroboros/repo')

from ouroboros.tools.search_agent import SearchAgent

# Протестируем только поиск DuckDuckGo
agent = SearchAgent(verbose=True)
print("\n=== Тест DuckDuckGo поиска ===")
results = agent.search_web("какие есть ИИ модели с бесплатным api")
print(f"Найдено: {len(results)} результатов")
for r in results[:3]:
    print(f"- {r['title']} | {r['url']}")

# Проверим создание клиента OpenAI
print("\n=== Проверка OpenAI клиента ===")
try:
    # Простой запрос без tools, просто проверим доступность модели
    resp = agent.client.chat.completions.create(
        model=agent.model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10
    )
    print(f"Модель ответила: {resp.choices[0].message.content[:50]}")
except Exception as e:
    print(f"Ошибка LLM: {e}")