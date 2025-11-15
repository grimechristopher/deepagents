# Wikipedia Research Agent - Local LLM Version

**100% Local & Free!** No API keys needed. Uses LM Studio + Wikipedia.

## Quick Start (5-10 minutes)

### 1. Install Dependencies

```bash
pip install deepagents wikipedia-api python-dotenv langchain-openai
```

### 2. Setup LM Studio

1. **Download LM Studio**: https://lmstudio.ai/
2. **Load a Model**: Download and load a model that supports function/tool calling
   - Recommended: `Llama 3.2 3B Instruct` (fast, good quality)
   - Better: `Llama 3.1 8B Instruct` (slower, better quality)
   - Advanced: `Mistral 7B Instruct v0.3` (excellent tool calling)
3. **Enable Server**:
   - Go to the "Local Server" tab in LM Studio
   - Click "Start Server"
   - Default URL: `http://localhost:1234`
   - Keep LM Studio running!

### 3. Run the POC

```bash
python wikipedia_research_poc.py
```

That's it! No API keys, no cloud services, runs 100% on your machine.

## What You Get

The agent will:
1. ✅ **Plan** the research using built-in todo list
2. ✅ **Search** Wikipedia for the main topic
3. ✅ **Explore** related topics and dive into sections
4. ✅ **Take notes** using the file system
5. ✅ **Write** a comprehensive markdown report
6. ✅ **Save** it as `research_report.md`

All using your local LLM!

## Features

### Two Wikipedia Tools

**`wikipedia_search(query)`**
- Searches Wikipedia for a topic
- Returns summary, sections, related topics
- Great for getting an overview

**`wikipedia_get_section(page_title, section_title)`**
- Gets detailed content from a specific section
- Perfect for deep dives into particular aspects

### Built-in DeepAgents Capabilities

- **Planning**: `write_todos` - breaks down the research task
- **File System**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Memory Management**: Automatic context summarization
- **Sub-agents**: Can spawn specialized agents if needed

## Customize

### Change the Topic

Edit line 197 in `wikipedia_research_poc.py`:

```python
topic = "Your topic here"
```

### Change LM Studio URL

Create a `.env` file:

```bash
LM_STUDIO_URL=http://localhost:1234/v1
```

### Use a Different Local LLM Server

Any OpenAI-compatible API works:

```python
local_llm = ChatOpenAI(
    base_url="http://your-server:port/v1",
    api_key="not-needed",
    temperature=0.7,
)
```

Works with:
- **LM Studio** (recommended, easiest)
- **Ollama** (with OpenAI compatibility layer)
- **vLLM**
- **Text Generation WebUI**
- Any other OpenAI-compatible local server

## Performance Tips

### Model Selection

**For Fast Results (2-5 minutes)**:
- Llama 3.2 3B Instruct
- Phi-3 Mini
- Qwen 2.5 3B

**For Better Quality (5-15 minutes)**:
- Llama 3.1 8B Instruct
- Mistral 7B Instruct v0.3
- Qwen 2.5 7B

**For Best Quality (15-30 minutes)**:
- Llama 3.1 70B Instruct (needs powerful GPU)
- Mixtral 8x7B

### Hardware Requirements

**Minimum** (3B models):
- 8GB RAM
- CPU only works but is slow
- GPU with 4GB VRAM recommended

**Recommended** (7-8B models):
- 16GB RAM
- GPU with 8GB VRAM
- Much faster inference

**Optimal** (70B models):
- 32GB+ RAM
- GPU with 24GB+ VRAM (or multiple GPUs)
- Best quality results

### Important: Tool/Function Calling

Not all models support tool calling well! For best results:

✅ **Good Tool Calling Support**:
- Llama 3.1/3.2 series
- Mistral Instruct v0.3
- Qwen 2.5 series
- Hermes 2 Pro series

❌ **Poor Tool Calling**:
- Base models (not instruct-tuned)
- Older models (pre-2024)
- Models without function calling training

If your model doesn't support tool calling well, the agent may struggle to use the Wikipedia tools effectively.

## Troubleshooting

### "Connection refused" or "Failed to connect"

**Problem**: Can't connect to LM Studio

**Solutions**:
1. Make sure LM Studio is running
2. Click "Start Server" in the Local Server tab
3. Check the port (default: 1234)
4. Try accessing http://localhost:1234/v1/models in your browser

### "Model doesn't follow instructions"

**Problem**: Agent doesn't use tools or plan properly

**Solutions**:
1. Use an instruct-tuned model (must have "Instruct" in name)
2. Try a larger model (8B instead of 3B)
3. Check if model supports function calling
4. Increase temperature to 0.7-0.9

### "Wikipedia page not found"

**Problem**: Agent searches for topics that don't exist

**Solutions**:
1. Use more specific topic names
2. Try related search terms
3. Check Wikipedia directly first
4. The agent will get suggestions for alternative searches

### Agent is very slow

**Problem**: Takes 30+ minutes to complete

**Solutions**:
1. Use a smaller model (3B instead of 7B)
2. Use GPU acceleration if available
3. Reduce context window size in LM Studio
4. Use quantized models (Q4 or Q5)

### Agent doesn't save the report

**Problem**: No `research_report.md` file created

**Solutions**:
1. Check if the model supports tool calling
2. Look for intermediate notes (agent may have saved those)
3. Try a different model with better instruction following
4. Make the prompt more explicit about saving

## Example Output

After running, you'll see:

```
================================================================================
Wikipedia Research Agent - Local LLM + Wikipedia
================================================================================

Connecting to LM Studio at: http://localhost:1234/v1

Researching topic: Quantum computing

The agent will:
  1. Plan the research approach
  2. Search Wikipedia for information
  3. Explore related topics and sections
  4. Take notes and organize findings
  5. Write a comprehensive report
...
```

And you'll get a file `research_report.md` with a comprehensive report!

## Cost

**FREE!** Runs 100% locally:
- ✅ No API costs
- ✅ No internet required (after model download)
- ✅ Complete privacy
- ✅ Unlimited usage

The only cost is electricity for running your computer.

## Privacy

Since everything runs locally:
- No data sent to external APIs
- Wikipedia queries are the only network activity
- Your research stays on your machine
- Perfect for sensitive topics

## Next Steps

### Add More Tools

```python
@tool
def arxiv_search(query: str):
    """Search academic papers on ArXiv."""
    # Implementation
    pass

agent = create_deep_agent(
    model=local_llm,
    tools=[wikipedia_search, wikipedia_get_section, arxiv_search],
)
```

### Use Subagents

```python
fact_checker = {
    "name": "fact-checker",
    "description": "Verifies facts across multiple Wikipedia sources",
    "system_prompt": "You are a fact-checking expert...",
    "tools": [wikipedia_search, wikipedia_get_section],
    "model": local_llm,
}

agent = create_deep_agent(
    model=local_llm,
    tools=[wikipedia_search, wikipedia_get_section],
    subagents=[fact_checker],
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

## Comparison: Local vs Cloud

| Feature | Local LLM | Cloud (Claude/GPT) |
|---------|-----------|-------------------|
| Cost | Free | $0.25-$0.50 per report |
| Speed | Slower (2-30 min) | Fast (1-2 min) |
| Privacy | 100% private | Sent to API |
| Quality | Good-Excellent | Excellent |
| Setup | Requires setup | Just API keys |
| Internet | Optional | Required |

Local LLMs are perfect when you value privacy, have unlimited time, or want to avoid API costs!
