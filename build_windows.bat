@echo off
setlocal
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m compileall -q minicut_agent main.py mcp_server.py smartcut_runner.py
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m PyInstaller --noconfirm --clean --windowed --name "MiniCut Studio Agent" --hidden-import PySide6.QtMultimedia --hidden-import PySide6.QtMultimediaWidgets --collect-data tzdata main.py
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m PyInstaller --noconfirm --clean --console --name "MiniCut MCP" mcp_server.py
if errorlevel 1 exit /b %errorlevel%
copy /Y "dist\MiniCut MCP\MiniCut MCP.exe" "dist\MiniCut Studio Agent\MiniCut MCP.exe"

py -3.11 -m PyInstaller --noconfirm --clean --onefile --console --name "MiniCut SmartCut" --collect-all smartcut --collect-all av smartcut_runner.py
if errorlevel 1 exit /b %errorlevel%
copy /Y "dist\MiniCut SmartCut.exe" "dist\MiniCut Studio Agent\MiniCut SmartCut.exe"
copy /Y "THIRD_PARTY_SMARTCUT.txt" "dist\MiniCut Studio Agent\THIRD_PARTY_SMARTCUT.txt"
copy /Y "README.md" "dist\MiniCut Studio Agent\README_AGENT.md"

"dist\MiniCut Studio Agent\MiniCut SmartCut.exe" --help >nul
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build selesai di dist\MiniCut Studio Agent
echo FFmpeg/ffprobe harus tersedia di PATH.
pause
