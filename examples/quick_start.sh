#!/bin/bash
# Quick start script for Agentic RAG pipeline

echo "=================================="
echo "Agentic RAG Pipeline - Quick Start"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Warning: .env file not found!"
    echo "Please copy .env.example to .env and add your API keys:"
    echo "  cp .env.example .env"
    echo ""
    exit 1
fi

# Index sample documents
echo ""
echo "Indexing sample documents..."
python3 main.py index data/raw

# Run health check
echo ""
echo "Running health check..."
python3 main.py health

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Try these commands:"
echo "  python3 main.py query           # Interactive query mode"
echo "  python3 main.py query 'What is machine learning?'  # Single query"
echo "  python3 main.py info            # Show collection info"
echo ""
