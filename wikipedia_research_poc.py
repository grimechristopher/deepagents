"""
Proof of Concept: Wikipedia Research Agent (Local LLM + Wikipedia)

This agent researches a topic using Wikipedia and writes a comprehensive report.
Uses LM Studio for local LLM inference.

Requirements:
    pip install deepagents wikipedia-api python-dotenv langchain-openai

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
import wikipediaapi

# Load environment variables
load_dotenv()

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    user_agent='DeepAgents-Research-Bot/1.0',
    language='en'
)


@tool
def wikipedia_search(query: str, sentences: int = 10) -> dict:
    """Search Wikipedia for information on a topic.

    Args:
        query: The topic to search for on Wikipedia
        sentences: Number of sentences to return from the summary (default: 10)

    Returns:
        Dictionary with page title, summary, URL, and related links
    """
    page = wiki.page(query)

    if not page.exists():
        # Try to find similar pages
        return {
            "found": False,
            "query": query,
            "suggestion": "Page not found. Try rephrasing your search query or search for related terms."
        }

    # Get summary (first N sentences)
    summary = page.summary.split('. ')[:sentences]
    summary_text = '. '.join(summary) + '.'

    # Get section titles (page.sections is already a list of Section objects)
    sections = [section.title for section in page.sections[:5]]

    # Get links to related pages
    links = list(page.links.keys())[:10]  # First 10 related links

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
    """Get detailed content from a specific section of a Wikipedia page.

    Args:
        page_title: The title of the Wikipedia page
        section_title: The title of the section to retrieve

    Returns:
        Dictionary with section content
    """
    page = wiki.page(page_title)

    if not page.exists():
        return {
            "found": False,
            "error": f"Page '{page_title}' not found"
        }

    # Find the section
    section = page.section_by_title(section_title)

    if section is None:
        return {
            "found": False,
            "error": f"Section '{section_title}' not found in page '{page_title}'",
            "available_sections": [s.title for s in page.sections]
        }

    return {
        "found": True,
        "page_title": page_title,
        "section_title": section_title,
        "content": section.text[:3000],  # Limit to 3000 chars to avoid context overflow
    }


# System prompt to guide the agent
RESEARCH_INSTRUCTIONS = """You are an expert research analyst and writer.

Use wikipedia_search to research the topic, then write a complete markdown report directly to the user.

IMPORTANT: Write the ENTIRE report in your final response. Do NOT use write_file. Do NOT say you saved anything.

## Report Format

Write a complete report with these sections:

# [Topic Name]

## Executive Summary
Brief overview of key findings

## Introduction
Background and context

## Main Findings
Detailed information organized by subtopics

## Key Insights
Important takeaways

## Sources
- List Wikipedia articles with URLs

Remember: Return the COMPLETE report text in your response.
"""


def main():
    """Run the Wikipedia research agent with local LLM."""

    print("=" * 80)
    print("Wikipedia Research Agent - Local LLM + Wikipedia")
    print("=" * 80)
    print()

    # Get LM Studio URL from environment or use default
    lm_studio_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

    print(f"Connecting to LM Studio at: {lm_studio_url}")
    print()

    # Configure local LLM (LM Studio with OpenAI-compatible API)
    # Get the model name from environment or use the one you have loaded
    model_name = os.getenv("LM_STUDIO_MODEL", "qwen2.5-14b-instruct")

    print(f"Using model: {model_name}")
    print()

    local_llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="not-needed",  # LM Studio doesn't require API key
        temperature=0.7,
        model=model_name,
    )

    # Create the deep agent with Wikipedia tools
    agent = create_deep_agent(
        model=local_llm,
        tools=[wikipedia_search, wikipedia_get_section],
        system_prompt=RESEARCH_INSTRUCTIONS,
    )

    # Example topic - you can change this!
    topic = "The confirmed physical attributes of Jesus Christ"

    print(f"Researching topic: {topic}")
    print()
    print("The agent will:")
    print("  1. Plan the research approach")
    print("  2. Search Wikipedia for information")
    print("  3. Explore related topics and sections")
    print("  4. Take notes and organize findings")
    print("  5. Write a comprehensive report")
    print()
    print("=" * 80)
    print()
    print("NOTE: This uses your local LLM, so performance depends on:")
    print("  - Model size and capabilities")
    print("  - Hardware (GPU/CPU)")
    print("  - Context window size")
    print()
    print("For best results, use a model with:")
    print("  - Good instruction following (e.g., Llama 3, Mistral)")
    print("  - Tool/function calling support")
    print("  - At least 8B parameters")
    print()
    print("=" * 80)
    print()

    # Run the agent
    try:
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Research '{topic}' using Wikipedia and write me a comprehensive report. Save the final report as 'research_report.md'."
                }
            ]
        })

        print()
        print("=" * 80)
        print("Research Complete!")
        print("=" * 80)
        print()

        # Extract the actual report content from all AI messages
        report_content = []
        for msg in result["messages"]:
            if hasattr(msg, 'content') and msg.content:
                # Check if this looks like report content (has sections/headers)
                content = msg.content
                if any(marker in content for marker in ["##", "**Executive Summary**", "Introduction", "Sources"]):
                    # This looks like the actual report
                    if len(content) > 200:  # Substantial content
                        report_content.append(content)

        # If we found report content, use it; otherwise use the last message
        if report_content:
            # Use the longest content (likely the full report)
            final_response = max(report_content, key=len)
            print("✅ Found comprehensive report in agent messages!")
        else:
            # Fallback to last message
            final_response = result["messages"][-1].content
            print("⚠️  Using last message (report may be incomplete)")

        print()
        print("Report preview (first 500 chars):")
        print("-" * 80)
        print(final_response[:500] + "..." if len(final_response) > 500 else final_response)
        print()

        # Save the report to a file
        output_file = "research_report.md"
        with open(output_file, "w") as f:
            f.write(final_response)

        print(f"✅ Report saved to: {output_file}")
        print(f"   ({len(final_response)} characters, {len(final_response.split())} words)")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR!")
        print("=" * 80)
        print()
        print(f"Failed to run agent: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure LM Studio is running")
        print("  2. Make sure a model is loaded in LM Studio")
        print("  3. Check that the server is enabled (Settings > Server)")
        print(f"  4. Verify the URL is correct: t{lm_studio_url}")
        print()
        raise


if __name__ == "__main__":
    main()
