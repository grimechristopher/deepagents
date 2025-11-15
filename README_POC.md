# Wikipedia Research Agent - Proof of Concept

A simple proof of concept showing how to use DeepAgents to research a topic and write a comprehensive report.

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install deepagents tavily-python python-dotenv
```

### 2. Get API Keys

1. **Tavily API Key** (Free tier available):
   - Go to https://www.tavily.com/
   - Sign up for a free account
   - Get your API key from the dashboard

2. **Anthropic API Key** (Required for Claude):
   - Go to https://console.anthropic.com/
   - Create an account and get your API key

### 3. Set Up Environment Variables

Create a `.env` file in this directory:

```bash
TAVILY_API_KEY=your_tavily_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 4. Run the POC

```bash
python wikipedia_research_poc.py
```

## What It Does

The agent will:

1. **Plan** - Uses the built-in `write_todos` tool to create a research plan
2. **Research** - Searches the web multiple times from different angles
3. **Organize** - Saves notes using the built-in file system tools
4. **Synthesize** - Writes a polished report saved as `research_report.md`

## Example Output

After running, you'll get:
- A markdown report saved as `research_report.md`
- Console output showing the agent's thought process
- Intermediate research notes (if the agent chose to save them)

## Customize the Topic

Edit line 110 in `wikipedia_research_poc.py`:

```python
topic = "Your topic here"
```

## How It Works

### Built-in Capabilities

The agent automatically has access to:
- ✅ **Planning** - `write_todos` tool for task decomposition
- ✅ **File System** - `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- ✅ **Memory Management** - Automatic context summarization
- ✅ **Sub-agents** - Can spawn specialized agents if needed

### Custom Tool

We only need to add ONE custom tool:
- `web_search` - Searches the web using Tavily

### The Magic

DeepAgents handles all the complexity:
- Automatic task planning
- Context management
- File operations
- Report synthesis

## Next Steps

### Add More Tools

```python
@tool
def wikipedia_fetch(page_title: str) -> str:
    """Fetch a Wikipedia page directly."""
    # Implementation using wikipedia-api or similar
    pass

agent = create_deep_agent(
    tools=[web_search, wikipedia_fetch],
    system_prompt=RESEARCH_INSTRUCTIONS,
)
```

### Add Subagents for Specialized Research

```python
fact_checker_subagent = {
    "name": "fact-checker",
    "description": "Verifies facts and checks source reliability",
    "system_prompt": "You are a fact-checking expert...",
    "tools": [web_search],
}

agent = create_deep_agent(
    tools=[web_search],
    system_prompt=RESEARCH_INSTRUCTIONS,
    subagents=[fact_checker_subagent],
)
```

### Stream the Output

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]},
    stream_mode="values"
):
    if "messages" in chunk:
        print(chunk["messages"][-1].content)
```

## Estimated Costs

Using Claude Sonnet 4:
- Input: ~$3 per million tokens
- Output: ~$15 per million tokens

A typical research report generation:
- ~50k-100k input tokens (~$0.15-$0.30)
- ~5k-10k output tokens (~$0.08-$0.15)
- **Total: ~$0.25-$0.50 per report**

## Troubleshooting

### "No API key found"
Make sure your `.env` file is in the same directory and contains both API keys.

### "Module not found"
Run `pip install deepagents tavily-python python-dotenv`

### Agent doesn't save the report
The agent might need clearer instructions. Add to your prompt: "IMPORTANT: You must save the final report as 'research_report.md' using the write_file tool."

## Performance Tips

1. **Be specific** - Give detailed topics for better results
2. **Use subagents** - For complex topics, define specialized subagents
3. **Monitor tokens** - Check the output to see token usage
4. **Iterate** - Run multiple times and refine your system prompt

## Time to Build

- **Setup**: 5 minutes
- **First run**: 1-2 minutes (depends on topic complexity)
- **Total**: ~10 minutes to get your first research report!
