@echo off
REM Code Formatting Script for Windows
REM This script formats all Python code in the repository using Black, isort, and other tools

echo 🎨 Formatting Python Code...

REM Check if virtual environment exists
if not exist "venv\" (
    echo ❌ Virtual environment not found. Please run setup.bat first.
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo 📦 Installing formatting tools...
pip install black isort autoflake autopep8 flake8

echo 🧹 Removing unused imports and variables...
autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive . --exclude=venv/

echo 📏 Fixing line length violations...
autopep8 --max-line-length=88 --aggressive --aggressive --in-place --recursive . --exclude=venv/

echo 🔧 Formatting with Black (88 character line limit)...
black --line-length 88 . --exclude="venv/"

echo 📚 Sorting imports with isort...
isort . --profile black --line-length 88 --skip venv

echo 🔍 Running flake8 check...
flake8 . --max-line-length=88 --exclude=venv/ || echo ⚠️ Some flake8 issues remain - check output above

echo ✅ Code formatting complete!
echo.
echo 📋 Summary:
echo - Removed unused imports and variables (autoflake)
echo - Fixed line length violations (autopep8)
echo - Applied consistent formatting (black)
echo - Sorted imports (isort)
echo - Checked for remaining issues (flake8)

pause
