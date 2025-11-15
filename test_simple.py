#!/usr/bin/env python3
"""Simple test to see if the agent is working."""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import wikipediaapi

load_dotenv()

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    user_agent='DeepAgents-Research-Bot/1.0',
    language='en'
)

@tool
def wikipedia_search(query: str, sentences: int = 3) -> dict:
    """Search Wikipedia for information on a topic."""
    page = wiki.page(query)

    if not page.exists():
        return {"found": False, "query": query}

    summary = page.summary.split('. ')[:sentences]
    summary_text = '. '.join(summary) + '.'

    return {
        "found": True,
        "title": page.title,
        "summary": summary_text,
        "url": page.fullurl,
    }

# Configure local LLM
lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
model_name = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")

print(f"Using: {lm_studio_url} with model: {model_name}")
print()

local_llm = ChatOpenAI(
    base_url=lm_studio_url,
    api_key="not-needed",
    temperature=0.7,
    model=model_name,
)

# Create simple agent
agent = create_deep_agent(
    model=local_llm,
    tools=[wikipedia_search],
    system_prompt="You are a helpful assistant. Use wikipedia_search to find information.",
)

print("Running simple test: searching for 'Python programming language'")
print("=" * 80)
print()

# Run the agent
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Search Wikipedia for 'Python programming language' and tell me what you find."
        }
    ]
})

print()
print("=" * 80)
print("Agent Response:")
print("=" * 80)
print()

# Print all messages to see what happened
for i, msg in enumerate(result["messages"]):
    print(f"Message {i} ({msg.__class__.__name__}):")
    if hasattr(msg, 'content'):
        print(msg.content)
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"  Tool calls: {msg.tool_calls}")
    print()

print("=" * 80)
print("Final message:")
print(result["messages"][-1].content)
