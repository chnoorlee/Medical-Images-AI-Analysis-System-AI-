@echo off
echo Installing frontend dependencies...
echo.

REM Check if npm is available
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: npm is not found in PATH
    echo Please install Node.js from https://nodejs.org/
    echo Then run this script again
    pause
    exit /b 1
)

echo Installing dependencies...
npm install

if %errorlevel% equ 0 (
    echo.
    echo Dependencies installed successfully!
    echo You can now run: npm run dev
) else (
    echo.
    echo Error installing dependencies
)

pause