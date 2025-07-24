# UltraSafe Data Processing and Analysis System Makefile

.PHONY: help setup install run clean clean-cache clean-all test lint format
PYTHON = python3
APP = app.main:app
HOST = 0.0.0.0
PORT = 8000
LOG_LEVEL = INFO
# Default target
help:
	@echo "UltraSafe Data Processing and Analysis System"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  install    - Install Python dependencies"
	@echo "  run        - Run the FastAPI application"
	@echo "  clean      - Remove Python cache files"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black and isort"
	@echo "  help       - Show this help message"

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up UltraSafe project..."
	python3 -m venv venv
	@echo "Virtual environment created. Activate it with: source venv/bin/activate"
	@echo "Then run: make install"

# Install Python dependencies
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed successfully"

# Run the application
run:
	@echo "Starting UltraSafe application..."
	python3 -m uvicorn $(APP) --host $(HOST) --port $(PORT) --reload
	
# Clean Python cache files
clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	@echo "Cache files cleaned"


lint:
	@echo "Running code linting..."
	flake8 app/ scripts/ tests/
	@echo "Linting completed"

# Format code
format:
	@echo "Formatting code..."
	black app/
	isort app/
	@echo "Code formatting completed"

# Create necessary directories
setup-dirs:
	@echo "Creating necessary directories..."
	mkdir -p uploads
	mkdir -p exports
	mkdir -p static/charts
	mkdir -p db
	@echo "Directories created"


