#!/bin/bash

# Narrative Engine Startup Script

echo "🎭 Starting Narrative Engine..."
echo "================================"

# Check Python version
python3 --version

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null || true

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
else
    echo "Installing dependencies..."
    pip install -q fastapi uvicorn websockets python-multipart
fi

# Start the server
echo ""
echo "🚀 Server starting on http://localhost:8000"
echo "📖 Story interface at http://localhost:8000/play"
echo "📚 API docs at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"

python3 -m backend.main