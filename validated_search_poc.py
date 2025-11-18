"""
Proof of Concept: Validated Web Search Agent with Fact-Checking

This agent researches topics, extracts claims, and validates them with a fact-checker subagent.
Uses Azure OpenAI (GPT-4o) and DuckDuckGo for web search.

Requirements:
    pip install deepagents ddgs requests beautifulsoup4 python-dotenv langchain-openai

Setup:
    1. Create .env file with Azure OpenAI credentials:
       AZURE_OPENAI_API_KEY=your-api-key
       AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
       AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
       AZURE_OPENAI_API_VERSION=2024-08-01-preview
"""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import AzureChatOpenAI
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
    """Run the validated search agent with Azure OpenAI."""

    # Get Azure OpenAI credentials from environment
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    # Validate required credentials
    if not azure_api_key or not azure_endpoint:
        raise ValueError(
            "Missing Azure OpenAI credentials. Please set:\n"
            "  AZURE_OPENAI_API_KEY\n"
            "  AZURE_OPENAI_ENDPOINT\n"
            "in your .env file"
        )

    print("=" * 80)
    print(f"Validated Search Agent: Azure OpenAI ({azure_deployment})")
    print("=" * 80)
    print()

    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        temperature=0.7,
    )

    # Example query - you can change this!
    query = "Who were in the main cast of Gilligans Island, and what are some notable facts about each actor?"

    print(f"Query: {query}")
    print()

    # ========================================================================
    # PART 1: Ask LLM directly (without tools/agent)
    # ========================================================================
    print("PART 1: Direct LLM (no web search)...")
    print()

    try:
        direct_response = azure_llm.invoke([
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
        model=azure_llm,
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
        tool_call_counts = {
            'ddg_search': 0,
            'crawl_webpage': 0,
            'task': 0,
            'other': 0
        }

        print("  Starting agent stream...")
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            if "messages" in chunk:
                msg = chunk["messages"][-1]

                # Track tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = getattr(tool_call, 'name', tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else 'unknown')

                        if tool_name == 'ddg_search':
                            tool_call_counts['ddg_search'] += 1
                            args = getattr(tool_call, 'args', tool_call.get('args', {}) if isinstance(tool_call, dict) else {})
                            query_text = args.get('query', 'N/A') if isinstance(args, dict) else 'N/A'
                            print(f"  🔍 Searching: {query_text}")

                        elif tool_name == 'crawl_webpage':
                            tool_call_counts['crawl_webpage'] += 1
                            args = getattr(tool_call, 'args', tool_call.get('args', {}) if isinstance(tool_call, dict) else {})
                            url = args.get('url', 'N/A') if isinstance(args, dict) else 'N/A'
                            print(f"  📄 Crawling: {url[:70]}...")

                        elif tool_name == 'task':
                            tool_call_counts['task'] += 1
                            print(f"  ✓ Validating claim...")

                        else:
                            tool_call_counts['other'] += 1
                            print(f"  🔧 Tool: {tool_name}")

                result = chunk

        print()

        # Recount all tool calls from the final result to ensure accuracy
        if result and "messages" in result:
            final_counts = {
                'ddg_search': 0,
                'crawl_webpage': 0,
                'task': 0,
                'other': 0
            }

            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = getattr(tool_call, 'name', tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else 'unknown')
                        if tool_name == 'ddg_search':
                            final_counts['ddg_search'] += 1
                        elif tool_name == 'crawl_webpage':
                            final_counts['crawl_webpage'] += 1
                        elif tool_name == 'task':
                            final_counts['task'] += 1
                        else:
                            final_counts['other'] += 1

            print("  Tool Call Summary:")
            print(f"    - DuckDuckGo searches: {final_counts['ddg_search']}")
            print(f"    - Webpage crawls: {final_counts['crawl_webpage']}")
            print(f"    - Validation tasks (subagent): {final_counts['task']}")
            if final_counts['other'] > 0:
                print(f"    - Other tools: {final_counts['other']}")
            print(f"    - Total tool calls: {sum(final_counts.values())}")
        else:
            print("  Tool Call Summary:")
            print(f"    - No tool calls detected")

        print()
        print(f"  Agent completed. Result: {result is not None}")

        # Extract final response
        if result:
            final_response = None
            # Look through messages in reverse to find the last AI message with content
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and msg.content and isinstance(msg.content, str):
                    content = msg.content.strip()
                    # Skip if it's just a tool call message
                    has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0
                    if len(content) > 100 and not has_tool_calls:
                        final_response = content
                        break

            # Debug: print message info if no response found
            if not final_response:
                print("  Debug: Message types in result:")
                for i, msg in enumerate(result["messages"]):
                    msg_type = type(msg).__name__
                    has_content = hasattr(msg, 'content') and msg.content
                    content_len = len(msg.content) if has_content else 0
                    has_tools = hasattr(msg, 'tool_calls') and msg.tool_calls
                    print(f"    [{i}] {msg_type}: content_len={content_len}, has_tool_calls={has_tools}")
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
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("  - Verify Azure OpenAI credentials in .env file")
        print("  - Check endpoint URL format")
        print("  - Confirm deployment name matches your Azure resource")
        print()


if __name__ == "__main__":
    main()
