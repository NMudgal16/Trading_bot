@echo off
echo ==================================================
echo        AI TRADING BOT PROJECT SETUP
echo ==================================================

echo.
echo Step 1: Creating project structure...
echo.

REM Create directories
mkdir config 2>nul
mkdir data 2>nul
mkdir strategies 2>nul
mkdir models 2>nul
mkdir execution 2>nul
mkdir risk_management 2>nul
mkdir backtesting 2>nul
mkdir dashboard 2>nul
mkdir dashboard\backend 2>nul
mkdir dashboard\frontend 2>nul
mkdir utils 2>nul
mkdir tests 2>nul
mkdir logs 2>nul

echo ✓ Directories created

REM Create __init__.py files
type nul > config\__init__.py
type nul > data\__init__.py
type nul > strategies\__init__.py
type nul > models\__init__.py
type nul > execution\__init__.py
type nul > risk_management\__init__.py
type nul > backtesting\__init__.py
type nul > utils\__init__.py
type nul > tests\__init__.py

echo ✓ __init__.py files created

REM Create main files
type nul > main.py
type nul > requirements.txt
type nul > Dockerfile
type nul > README.md

echo ✓ Main files created

echo.
echo Step 2: Creating basic file contents...
echo.

REM Create requirements.txt
echo # Core dependencies > requirements.txt
echo pandas^>=2.0.0 >> requirements.txt
echo numpy^>=1.24.0 >> requirements.txt
echo yfinance^>=0.2.28 >> requirements.txt
echo requests^>=2.31.0 >> requirements.txt
echo flask^>=2.3.0 >> requirements.txt
echo python-dotenv^>=1.0.0 >> requirements.txt

REM Create main.py
echo print("AI Trading Bot Project") > main.py
echo print("======================") >> main.py
echo print("Project structure created successfully!") >> main.py
echo print("") >> main.py
echo print("Next steps:") >> main.py
echo print("1. python -m venv venv") >> main.py
echo print("2. venv\Scripts\activate") >> main.py
echo print("3. pip install -r requirements.txt") >> main.py

REM Create .gitignore
echo venv/ > .gitignore
echo .env >> .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore
echo *.db >> .gitignore
echo *.log >> .gitignore
echo .vscode/ >> .gitignore

echo ✓ File contents created

echo.
echo ==================================================
echo        SETUP COMPLETE!
echo ==================================================
echo.
echo Next steps:
echo 1. Create virtual environment: python -m venv venv
echo 2. Activate it: venv\Scripts\activate
echo 3. Install packages: pip install -r requirements.txt
echo 4. Test: python main.py
echo.
pause