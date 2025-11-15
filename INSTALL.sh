#!/bin/bash
# Installation script for Wikipedia Research POC

echo "=============================================================================="
echo "Installing Wikipedia Research Agent - Local LLM Version"
echo "=============================================================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment detected. Creating one..."
    echo ""

    # Create venv
    python3 -m venv venv

    echo "Virtual environment created!"
    echo ""
    echo "To activate it, run:"
    echo "  source venv/bin/activate"
    echo ""
    echo "Then run this script again."
    exit 0
fi

echo "Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo ""

pip install deepagents wikipedia-api python-dotenv langchain-openai

echo ""
echo "=============================================================================="
echo "Installation Complete!"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Make sure LM Studio is running with a model loaded"
echo "  2. Enable the server in LM Studio (Local Server tab)"
echo "  3. Run: python wikipedia_research_poc.py"
echo ""
