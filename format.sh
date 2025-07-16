#!/bin/bash

# Code Formatting Script
# This script formats all Python code in the repository using Black, isort, and other tools

set -e  # Exit on any error

echo "🎨 Formatting Python Code..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "📦 Installing formatting tools..."
pip install black isort autoflake autopep8 flake8

echo "🧹 Removing unused imports and variables..."
autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive . --exclude=venv/

echo "📏 Fixing line length violations..."
autopep8 --max-line-length=88 --aggressive --aggressive --in-place --recursive . --exclude=venv/

echo "🔧 Formatting with Black (88 character line limit)..."
black --line-length 88 . --exclude="venv/"

echo "📚 Sorting imports with isort..."
isort . --profile black --line-length 88 --skip venv

echo "🔍 Running flake8 check..."
flake8 . --max-line-length=88 --exclude=venv/ || echo "⚠️  Some flake8 issues remain - check output above"

echo "✅ Code formatting complete!"
echo ""
echo "📋 Summary:"
echo "- Removed unused imports and variables (autoflake)"
echo "- Fixed line length violations (autopep8)"
echo "- Applied consistent formatting (black)"
echo "- Sorted imports (isort)"
echo "- Checked for remaining issues (flake8)"
