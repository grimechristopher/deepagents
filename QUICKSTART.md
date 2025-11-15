# Quick Start - Wikipedia Research POC

## Option 1: Using Virtual Environment (Recommended)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install deepagents wikipedia-api python-dotenv langchain-openai

# 3. Run the POC
python wikipedia_research_poc.py
```

## Option 2: Using the Install Script

```bash
# 1. Run the install script (it will create venv for you)
./INSTALL.sh

# 2. Activate the venv
source venv/bin/activate

# 3. Run the install script again (it will install deps)
./INSTALL.sh

# 4. Run the POC
python wikipedia_research_poc.py
```

## Option 3: Global Install (Not Recommended)

```bash
# Install directly to your system Python
pip install deepagents wikipedia-api python-dotenv langchain-openai

# Run the POC
python wikipedia_research_poc.py
```

## Prerequisites

Before running, make sure:

1. **LM Studio is running**
   - Download from: https://lmstudio.ai/
   - Load a model (recommend: Llama 3.2 3B Instruct)
   - Go to "Local Server" tab
   - Click "Start Server"

2. **Server is accessible**
   - Default URL: http://localhost:1234
   - Test it: `curl http://localhost:1234/v1/models`

## Troubleshooting

### "ModuleNotFoundError: No module named 'dotenv'"
You forgot to install dependencies or activate your venv!

```bash
# If using venv:
source venv/bin/activate
pip install python-dotenv

# If not using venv:
pip install python-dotenv
```

### "ModuleNotFoundError: No module named 'deepagents'"
Install deepagents:

```bash
pip install deepagents
```

### "ModuleNotFoundError: No module named 'wikipediaapi'"
Install wikipedia-api:

```bash
pip install wikipedia-api
```

### Still having issues?
Install everything at once:

```bash
pip install deepagents wikipedia-api python-dotenv langchain-openai
```
