"""
Wikipedia Research Agent with Metrics Tracking

This version shows exactly what DeepAgents is doing under the hood.
"""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import wikipediaapi
from collections import defaultdict

load_dotenv()

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    user_agent='DeepAgents-Research-Bot/1.0',
    language='en'
)

# Metrics tracking
metrics = {
    "tool_calls": defaultdict(int),
    "total_tool_calls": 0,
    "agents_spawned": 0,
    "model_calls": 0,
}


@tool
def wikipedia_search(query: str, sentences: int = 10) -> dict:
    """Search Wikipedia for information on a topic."""
    metrics["tool_calls"]["wikipedia_search"] += 1
    metrics["total_tool_calls"] += 1

    page = wiki.page(query)
    if not page.exists():
        return {"found": False, "query": query}

    summary = page.summary.split('. ')[:sentences]
    summary_text = '. '.join(summary) + '.'
    sections = [section.title for section in page.sections[:5]]
    links = list(page.links.keys())[:10]

    return {
        "found": True,
        "title": page.title,
        "summary": summary_text,
        "url": page.fullurl,
        "sections": sections,
        "related_topics": links,
    }


@tool
def wikipedia_get_section(page_title: str, section_title: str) -> dict:
    """Get detailed content from a specific section of a Wikipedia page."""
    metrics["tool_calls"]["wikipedia_get_section"] += 1
    metrics["total_tool_calls"] += 1

    page = wiki.page(page_title)
    if not page.exists():
        return {"found": False, "error": f"Page '{page_title}' not found"}

    section = page.section_by_title(section_title)
    if section is None:
        return {
            "found": False,
            "error": f"Section '{section_title}' not found",
            "available_sections": [s.title for s in page.sections]
        }

    return {
        "found": True,
        "page_title": page_title,
        "section_title": section_title,
        "content": section.text[:3000],
    }


RESEARCH_INSTRUCTIONS = """You are an expert research analyst and writer.

Your job is to:
1. Conduct thorough research on the given topic using Wikipedia tools
2. Search for multiple aspects of the topic to get comprehensive information
3. Synthesize the information into a well-structured, polished report

Use the wikipedia_search and wikipedia_get_section tools to gather information.

Write a comprehensive report with:
- Executive Summary
- Introduction
- Main Body (organized by theme)
- Key Insights
- Sources (list Wikipedia articles consulted with URLs)
"""


def print_metrics():
    """Print metrics about what happened during execution."""
    print()
    print("=" * 80)
    print("DEEP AGENTS METRICS")
    print("=" * 80)
    print()

    print("🤖 AGENTS:")
    print(f"  Main agent: 1")
    print(f"  Subagents spawned: {metrics['agents_spawned']}")
    print(f"  Total agents: {1 + metrics['agents_spawned']}")
    print()

    print("🔧 BUILT-IN TOOLS (provided by DeepAgents):")
    print("  - write_todos (planning)")
    print("  - ls, read_file, write_file, edit_file (file system)")
    print("  - glob, grep (search)")
    print("  - task (spawn subagents)")
    print()

    print("🛠️  CUSTOM TOOLS (provided by you):")
    print("  - wikipedia_search")
    print("  - wikipedia_get_section")
    print()

    print("📊 TOOL USAGE:")
    for tool_name, count in sorted(metrics["tool_calls"].items()):
        print(f"  {tool_name}: {count} calls")
    print(f"  Total custom tool calls: {metrics['total_tool_calls']}")
    print()

    print("🧠 HOW DEEPAGENTS WORKS:")
    print("  1. TodoListMiddleware - Adds write_todos tool for planning")
    print("  2. FilesystemMiddleware - Adds file system tools (ls, read_file, etc)")
    print("  3. SubAgentMiddleware - Adds task tool to spawn subagents")
    print("  4. SummarizationMiddleware - Auto-summarizes when context gets large")
    print()
    print("  Your custom tools (wikipedia_search, wikipedia_get_section) are")
    print("  added alongside these built-in tools!")
    print()


def main():
    """Run the Wikipedia research agent with metrics."""

    print("=" * 80)
    print("Wikipedia Research Agent - WITH METRICS")
    print("=" * 80)
    print()

    # Configure local LLM
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")

    print(f"LM Studio: {lm_studio_url}")
    print(f"Model: {model_name}")
    print()

    local_llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="not-needed",
        temperature=0.7,
        model=model_name,
    )

    # Create the deep agent
    agent = create_deep_agent(
        model=local_llm,
        tools=[wikipedia_search, wikipedia_get_section],
        system_prompt=RESEARCH_INSTRUCTIONS,
    )

    topic = "Quantum computing"

    print(f"Topic: {topic}")
    print()
    print("Running agent...")
    print("=" * 80)
    print()

    # Run the agent with streaming to see what's happening
    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Research '{topic}' using Wikipedia and write a comprehensive report."
                }
            ]
        },
        stream_mode="values"
    ):
        # Count messages to track model calls
        if "messages" in chunk:
            metrics["model_calls"] = len([m for m in chunk["messages"] if hasattr(m, 'content')])

            # Print the latest message
            latest = chunk["messages"][-1]
            if hasattr(latest, 'content') and latest.content:
                print(f"\n[{latest.__class__.__name__}]")
                if len(latest.content) > 500:
                    print(latest.content[:500] + "...")
                else:
                    print(latest.content)

            # Check for tool calls
            if hasattr(latest, 'tool_calls') and latest.tool_calls:
                for tc in latest.tool_calls:
                    print(f"  → Calling tool: {tc['name']}")

    # Get final result
    final_response = chunk["messages"][-1].content

    print()
    print("=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print()
    print(final_response)

    # Save the report
    output_file = "research_report_with_metrics.md"
    with open(output_file, "w") as f:
        f.write(final_response)

    print()
    print(f"✅ Report saved to: {output_file}")

    # Print metrics
    print_metrics()


if __name__ == "__main__":
    main()
