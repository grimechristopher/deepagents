"""
Proof of Concept: Validated Web Search Agent with Fact-Checking

This agent researches topics, extracts claims, and validates them with a fact-checker subagent.
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

# Load environment variables
load_dotenv()

# ===== TOOLS =====

@tool
def ddg_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for a query and return results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No results found for: {query}"

            formatted = "\n\n".join([
                f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                for r in results
            ])
            return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def crawl_webpage(url: str) -> str:
    """Fetch and extract text content from a webpage."""
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())

        # Truncate if too long
        return text[:8000] if len(text) > 8000 else text

    except Exception as e:
        return f"Error crawling {url}: {str(e)}"


# ===== VALIDATION SUBAGENT =====

validation_subagent = {
    "name": "fact-checker",
    "description": "Validates claims by finding supporting and contradicting evidence",
    "system_prompt": """Validate claims thoroughly.

For each claim:
- Search for supporting evidence
- Search for contradictions
- Crawl sources if snippets are insufficient
- If you find conflicts, search more to resolve them

Return:
CLAIM: [claim]
SUPPORTING: [evidence with sources]
CONTRADICTING: [if found, with sources]
CONFIDENCE: HIGH / MEDIUM / LOW
VERDICT: CONFIRMED / LIKELY TRUE / UNCERTAIN / LIKELY FALSE
NOTES: [important caveats or conflicting details]
NEEDS_MORE_RESEARCH: [YES if LOW confidence or unresolved conflicts, NO otherwise]""",
    "tools": [ddg_search, crawl_webpage],
}


def main():
    """Run the validated search agent with local LLM."""

    # Get LM Studio URL from environment or use default
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")

    print("=" * 80)
    print(f"Validated Search Agent: {model_name}")
    print("=" * 80)
    print()

    local_llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="not-needed",  # LM Studio doesn't require API key
        temperature=0.7,
        model=model_name,
    )

    # Example query - you can change this!
    query = "How to catch Bulbasaur in Pokemon Go"

    print(f"Query: {query}")
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
                "content": f"Answer this question concisely with the right amount of detail: {query}"
            }
        ])

        direct_answer = direct_response.content

        # Save direct response
        direct_output = "unvalidated_llm_response.md"
        with open(direct_output, "w", encoding="utf-8") as f:
            f.write(f"# Unvalidated LLM Response (No Web Search, No Fact-Checking)\n\n")
            f.write(f"**Query:** {query}\n\n")
            f.write(direct_answer)

        print(f"✅ Saved: {direct_output} ({len(direct_answer.split())} words)")
        print()

    except Exception as e:
        print(f"❌ Failed to get direct LLM response: {e}")
        print()

    # ========================================================================
    # PART 2: Validated search agent
    # ========================================================================
    print("PART 2: Validated search with fact-checking...")
    print()

    # ===== MAIN AGENT =====

    agent = create_deep_agent(
        model=local_llm,
        tools=[ddg_search, crawl_webpage],
        subagents=[validation_subagent],
        system_prompt="""Research workflow:

1. Search and crawl as needed
2. Extract key claims from findings
3. Validate all claims with fact-checker subagent
4. For LOW confidence claims, search more and revalidate
5. Present validated findings with confidence levels

Cite sources when relevant. Be direct and to the point."""
    )

    try:
        result = None
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            if "messages" in chunk:
                msg = chunk["messages"][-1]

                # Track tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')

                        if tool_name == 'ddg_search':
                            query_text = tool_call['args'].get('query', 'N/A')
                            print(f"  🔍 Searching: {query_text}")

                        elif tool_name == 'crawl_webpage':
                            url = tool_call['args'].get('url', 'N/A')
                            print(f"  📄 Crawling: {url[:70]}...")

                        elif tool_name == 'task':
                            print(f"  ✓ Validating claim...")

                result = chunk

        print()

        # Extract final response
        if result:
            final_response = None
            for msg in result["messages"]:
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Look for substantial content without tool calls
                    if len(content) > 100 and not hasattr(msg, 'tool_calls'):
                        final_response = content
        else:
            final_response = None

        # Save output
        if result and final_response:
            output_file = "validated_search_report.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Validated Search Report\n\n")
                f.write(f"**Query:** {query}\n\n")
                f.write(final_response)

            print(f"✅ Saved: {output_file} ({len(final_response.split())} words)")
            print()
            print("Compare the two reports:")
            print(f"  1. unvalidated_llm_response.md (no tools, no validation)")
            print(f"  2. validated_search_report.md (with fact-checking)")
            print()
        else:
            print("⚠️  No validated response generated")
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
