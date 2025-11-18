"""
Proof of Concept: Wolfram Alpha Math Agent with Query Rewriting

This agent solves mathematical problems using Wolfram Alpha with automatic query formatting.
Uses Azure OpenAI (GPT-4o) for both query rewriting and the main agent.

Requirements:
    pip install deepagents langchain-openai langchain-community python-dotenv wolframalpha

Setup:
    1. Create .env file with Azure OpenAI credentials:
       AZURE_OPENAI_API_KEY=your-api-key
       AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
       AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
       AZURE_OPENAI_API_VERSION=2024-12-01-preview
       WOLFRAM_ALPHA_APPID=your-wolfram-app-id
"""

import os
import httpx
import xmltodict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from deepagents import create_deep_agent

# Load environment variables
load_dotenv()


# ===== TOOLS =====

def make_rewrite_tool(llm):
    """Factory function to create rewrite tool with specific LLM."""
    @tool
    def rewrite_for_wolfram(natural_language_question: str) -> str:
        """Convert a natural language math question into proper Wolfram Alpha syntax.

        This tool reformats user questions into the exact syntax Wolfram Alpha expects.
        Always use this BEFORE calling wolfram_query.
        """

        prompt = f"""Convert this natural language math question to Wolfram Alpha syntax.

Rules:
- Equations: "solve [equation] for [variable]"
  Example: "solve 2x + 10 = 300 for x"
- Integrals: "integrate [expression] from [a] to [b]"
  Example: "integrate x^2 from 0 to 5"
- Derivatives: "derivative of [expression]"
  Example: "derivative of sin(x)"
- Limits: "limit of [expression] as [variable] approaches [value]"
  Example: "limit of 1/x as x approaches 0"

Question: {natural_language_question}

Output only the Wolfram query with no additional text:"""

        response = llm.invoke(prompt)
        return response.content.strip()

    return rewrite_for_wolfram


def make_wolfram_tool(appid: str):
    """Factory function to create Wolfram tool with API key."""
    @tool
    def wolfram_query(formatted_query: str) -> str:
        """Execute a query to Wolfram Alpha with properly formatted mathematical syntax.

        IMPORTANT: Only use this tool with output from rewrite_for_wolfram.
        Do not pass raw natural language to this tool.
        """
        try:
            # Direct API call to avoid wolframalpha library Content-Type bug
            url = "https://api.wolframalpha.com/v2/query"
            params = {
                "input": formatted_query,
                "appid": appid,
                "output": "json"
            }

            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()

                data = resp.json()
                result = data.get("queryresult", {})

                if not result.get("success"):
                    return f"Wolfram Alpha could not understand the query: {formatted_query}"

                # Extract plaintext results from pods
                outputs = []
                for pod in result.get("pods", []):
                    title = pod.get("title", "")
                    for subpod in pod.get("subpods", []):
                        plaintext = subpod.get("plaintext", "")
                        if plaintext:
                            outputs.append(f"{title}: {plaintext}")

                if outputs:
                    return "\n".join(outputs)
                else:
                    return "No plaintext results found"

        except Exception as e:
            return f"Error querying Wolfram Alpha: {str(e)}"

    return wolfram_query


# System prompt for the agent
WOLFRAM_INSTRUCTIONS = """You are a mathematical assistant powered by Wolfram Alpha.

WORKFLOW (follow this exactly):
1. When the user asks a math question, FIRST call rewrite_for_wolfram with their question
2. Take the reformatted query from rewrite_for_wolfram
3. THEN call wolfram_query with that reformatted query
4. Present the result clearly to the user

Always use both tools in sequence. Never skip the rewrite step.

Example flow:
User: "What is X in 2x + 10 = 300"
→ Call rewrite_for_wolfram("What is X in 2x + 10 = 300")
→ Get back: "solve 2x + 10 = 300 for x"
→ Call wolfram_query("solve 2x + 10 = 300 for x")
→ Present result to user

Be concise and direct in your explanations.
"""


def main():
    """Run the Wolfram Alpha math agent with Azure OpenAI."""

    # Get Azure OpenAI credentials from environment
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Validate required credentials
    if not azure_api_key or not azure_endpoint:
        raise ValueError(
            "Missing Azure OpenAI credentials. Please set:\n"
            "  AZURE_OPENAI_API_KEY\n"
            "  AZURE_OPENAI_ENDPOINT\n"
            "in your .env file"
        )

    # Check for Wolfram Alpha API key
    if not os.getenv("WOLFRAM_ALPHA_APPID"):
        raise ValueError(
            "Missing WOLFRAM_ALPHA_APPID environment variable.\n"
            "Please set it in your .env file"
        )

    print("=" * 80)
    print(f"Wolfram Alpha Math Agent: Azure OpenAI ({azure_deployment})")
    print("=" * 80)
    print()

    # Initialize Azure OpenAI LLM
    azure_llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        azure_deployment=azure_deployment,
        api_version=azure_api_version,
        temperature=0.7,
    )

    # Get Wolfram Alpha API key
    wolfram_appid = os.getenv("WOLFRAM_ALPHA_APPID")

    # Example questions - you can change these!
    test_questions = [
        "What is X in 2x + 10 = 300",
        "What is the integral of x^2 from 0 to 5?",
        "Find the derivative of sin(x) * cos(x)",
    ]

    for question_idx, question in enumerate(test_questions, 1):
        print(f"Question {question_idx}/{len(test_questions)}: {question}")
        print()

        # ====================================================================
        # PART 1: Ask LLM directly (without tools/agent)
        # ====================================================================
        print("PART 1: Direct LLM (no Wolfram Alpha)...")
        print()

        try:
            direct_response = azure_llm.invoke([
                {
                    "role": "user",
                    "content": f"Solve this math problem: {question}"
                }
            ])

            direct_answer = direct_response.content

            # Save direct response
            direct_output = f"direct_llm_wolfram_response{question_idx}.md"
            with open(direct_output, "w", encoding="utf-8") as f:
                f.write(f"# Direct LLM Response (No Wolfram Alpha)\n\n")
                f.write(f"**Question:** {question}\n\n")
                f.write(direct_answer)

            print(f"✅ Saved: {direct_output} ({len(direct_answer.split())} words)")
            print()

        except Exception as e:
            print(f"❌ Failed to get direct LLM response: {e}")
            print()

        # ====================================================================
        # PART 2: DeepAgent with Wolfram Alpha tools
        # ====================================================================
        print("PART 2: DeepAgent with Wolfram Alpha...")
        print()

        # Create tools with Azure LLM and Wolfram API key
        rewrite_for_wolfram = make_rewrite_tool(azure_llm)
        wolfram_query = make_wolfram_tool(wolfram_appid)

        # Create the deep agent
        agent = create_deep_agent(
            model=azure_llm,
            tools=[rewrite_for_wolfram, wolfram_query],
            system_prompt=WOLFRAM_INSTRUCTIONS,
        )

        try:
            print("Tool calls:")

            result = None
            for chunk in agent.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
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

                            if tool_name == 'rewrite_for_wolfram':
                                print(f"  🔄 Rewriting query: {tool_args.get('natural_language_question', 'N/A')[:60]}...")
                            elif tool_name == 'wolfram_query':
                                print(f"  🧮 Wolfram Alpha: {tool_args.get('formatted_query', 'N/A')}")

                result = chunk

            print()

            if not result:
                print("❌ No result received")
                continue

            # Extract final response
            final_response = result["messages"][-1].content

            # Save the response
            output_file = f"wolfram_agent_response{question_idx}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Wolfram Alpha Agent Response\n\n")
                f.write(f"**Question:** {question}\n\n")
                f.write(final_response)

            print(f"✅ Saved: {output_file} ({len(final_response.split())} words)")
            print()
            print("Compare the two responses:")
            print(f"  1. direct_llm_wolfram_response{question_idx}.md (no tools)")
            print(f"  2. wolfram_agent_response{question_idx}.md (with Wolfram Alpha)")
            print()

        except Exception as e:
            print(f"❌ Error: {e}")
            print()
            print("Troubleshooting:")
            print("  - Verify Azure OpenAI credentials in .env file")
            print("  - Check endpoint URL format")
            print("  - Confirm deployment name matches your Azure resource")
            print("  - Verify WOLFRAM_ALPHA_APPID in .env file")
            print()
            raise

        if question_idx < len(test_questions):
            print("-" * 80)
            print()


if __name__ == "__main__":
    main()
