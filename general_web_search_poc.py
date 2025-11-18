"""
Proof of Concept: General Web Search Research Agent (Local LLM + DuckDuckGo)

This agent researches any topic using web search and crawling, then writes a comprehensive report.
Uses LM Studio for local LLM inference and DuckDuckGo for web search.

Requirements:
    pip install deepagents ddgs requests beautifulsoup4 python-dotenv langchain-openai

Setup:
    1. Start LM Studio with local server enabled (default: http://localhost:1234)
    2. Load a model in LM Studio
    3. (Optional) Create .env file to customize LM_STUDIO_URL
"""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

# Load environment variables
load_dotenv()


@tool
def ddg_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of search results with title, snippet, and URL
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return [{
                    "error": f"No results found for query: {query}",
                    "suggestion": "Try different keywords or rephrase your search"
                }]

            # Format results
            formatted_results = []
            for idx, result in enumerate(results, 1):
                formatted_results.append({
                    "rank": idx,
                    "title": result.get("title", "No title"),
                    "snippet": result.get("body", "No description"),
                    "url": result.get("href", ""),
                })

            return formatted_results

    except Exception as e:
        return [{
            "error": f"Search failed: {str(e)}",
            "suggestion": "Try again or rephrase your query"
        }]


@tool
def crawl_webpage(url: str, max_chars: int = 5000) -> dict:
    """Crawl a webpage and extract its main text content.

    Args:
        url: The URL to crawl
        max_chars: Maximum characters to return (default: 5000)

    Returns:
        Dictionary with page title, content, and URL
    """
    try:
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get title
        title = soup.title.string if soup.title else "No title"

        # Get main text content
        # Try to find main content area first
        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = ' '.join(text.split())

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        return {
            "success": True,
            "url": url,
            "title": title,
            "content": text,
            "char_count": len(text)
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "url": url,
            "error": "Request timed out (10s limit)"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "url": url,
            "error": f"Failed to fetch page: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"Failed to parse page: {str(e)}"
        }


# System prompt to guide the agent with iterative search refinement
RESEARCH_INSTRUCTIONS = """You are a research assistant with web search capabilities.

Research the topic using ddg_search and crawl_webpage. If your first search doesn't give good results, try different keywords or queries.

Provide a concise answer with the right amount of detail. Include:
- Direct answer to the question
- Key facts and relevant details
- Sources with URLs

Be direct and to the point. Do not mention saving files.
"""


def main():
    """Run the web search research agent with local LLM."""

    # Get LM Studio URL from environment or use default
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")

    print("=" * 80)
    print(f"Web Search Research: {model_name}")
    print("=" * 80)
    print()

    local_llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="not-needed",  # LM Studio doesn't require API key
        temperature=0.7,
        model=model_name,
    )

    # Example topic - you can change this!
    topic = "What are the permit requirements for building a backyard ADU in Los Angeles County in 2025"

    print(f"Topic: {topic}")
    print()

    # ========================================================================
    # PART 1: Ask LLM directly (without tools/agent)
    # ========================================================================
    print("PART 1: Direct LLM (no web search)...")
    print()

    try:
        direct_response = local_llm.invoke([
            {
                "role": "user",
                "content": f"Answer this question concisely with the right amount of detail: {topic}"
            }
        ])

        direct_answer = direct_response.content

        # Save direct response
        direct_output = "direct_llm_response.md"
        with open(direct_output, "w", encoding="utf-8") as f:
            f.write(f"# Direct LLM Response (No Web Search)\n\n")
            f.write(f"**Query:** {topic}\n\n")
            f.write(direct_answer)

        print(f"✅ Saved: {direct_output} ({len(direct_answer.split())} words)")
        print()

    except Exception as e:
        print(f"❌ Failed to get direct LLM response: {e}")
        print()

    # ========================================================================
    # PART 2: DeepAgent with web search tools
    # ========================================================================
    print("PART 2: DeepAgent with web search...")
    print()

    # Create the deep agent with web search tools
    agent = create_deep_agent(
        model=local_llm,
        tools=[ddg_search, crawl_webpage],
        system_prompt=RESEARCH_INSTRUCTIONS,
    )

    # Run the agent with streaming to show progress
    try:
        print("Tool calls:")

        result = None
        for chunk in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Research and answer: {topic}"
                    }
                ]
            },
            stream_mode="values"
        ):
            if "messages" in chunk:
                msg = chunk["messages"][-1]

                # Show tool calls as they happen
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})

                        if tool_name == 'ddg_search':
                            print(f"  🔍 Searching: {tool_args.get('query', 'N/A')}")
                        elif tool_name == 'crawl_webpage':
                            print(f"  📄 Crawling: {tool_args.get('url', 'N/A')}")

            result = chunk

        print()

        if not result:
            print("❌ No result received")
            return

        # Extract the actual report content from all AI messages
        report_content = []
        for msg in result["messages"]:
            if hasattr(msg, 'content') and msg.content:
                content = msg.content
                # Check if this looks like report content (has markdown headers)
                if any(marker in content for marker in ["##", "# ", "Executive Summary", "Introduction", "Sources"]):
                    if len(content) > 200:  # Substantial content
                        report_content.append(content)

        # Use the longest content (likely the full report)
        if report_content:
            final_response = max(report_content, key=len)
        else:
            # Fallback to last message
            final_response = result["messages"][-1].content

        # Save the report to a file
        output_file = "agent_web_search_report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# DeepAgent Web Search Report\n\n")
            f.write(f"**Query:** {topic}\n\n")
            f.write(final_response)

        print(f"✅ Saved: {output_file} ({len(final_response.split())} words)")
        print()
        print("Compare the two reports:")
        print(f"  1. direct_llm_response.md (no tools)")
        print(f"  2. agent_web_search_report.md (with web search)")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Make sure LM Studio is running with a loaded model")
        print("  - Check server is enabled in Settings")
        print(f"  - Verify URL: {lm_studio_url}")
        print()
        raise


if __name__ == "__main__":
    main()
