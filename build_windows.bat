@echo off
setlocal
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\fetch_mpv.ps1"
if errorlevel 1 exit /b %errorlevel%

powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\fetch_ffmpeg.ps1"
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m compileall -q minicut_agent main.py mcp_server.py smartcut_runner.py
if errorlevel 1 exit /b %errorlevel%

py -3.11 -m PyInstaller --noconfirm --clean --windowed --name "MiniCut Studio Agent" --hidden-import PySide6.QtMultimedia --hidden-import PySide6.QtMultimediaWidgets --hidden-import mpv --collect-data tzdata main.py
if errorlevel 1 exit /b %errorlevel%
for /R "mpv-runtime" %%F in (*.dll) do copy /Y "%%F" "dist\MiniCut Studio Agent\"
copy /Y "THIRD_PARTY_MPV.txt" "dist\MiniCut Studio Agent\THIRD_PARTY_MPV.txt"
copy /Y "ffmpeg-runtime\ffmpeg.exe" "dist\MiniCut Studio Agent\ffmpeg.exe"
copy /Y "ffmpeg-runtime\ffprobe.exe" "dist\MiniCut Studio Agent\ffprobe.exe"
copy /Y "THIRD_PARTY_FFMPEG.txt" "dist\MiniCut Studio Agent\THIRD_PARTY_FFMPEG.txt"
if exist "ffmpeg-runtime\FFMPEG_BUILD_LICENSE.txt" copy /Y "ffmpeg-runtime\FFMPEG_BUILD_LICENSE.txt" "dist\MiniCut Studio Agent\FFMPEG_BUILD_LICENSE.txt"

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
"dist\MiniCut Studio Agent\ffmpeg.exe" -version >nul
if errorlevel 1 exit /b %errorlevel%
"dist\MiniCut Studio Agent\ffprobe.exe" -version >nul
if errorlevel 1 exit /b %errorlevel%

echo Build selesai di dist\MiniCut Studio Agent
echo Paket portable sudah membawa FFmpeg dan ffprobe sendiri.
pause
