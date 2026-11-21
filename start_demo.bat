@echo off
pushd "%~dp0"

echo.
echo ============================================================
echo   ShopMate 电商售后 Agent
echo ============================================================
echo.

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" goto NO_VENV

echo 正在启动，稍等...
echo.
echo   地址： http://localhost:8000
echo.
echo   第一次启动要加载检索模型，大约 10-20 秒。
echo   浏览器会自动打开；如果打开早了显示连不上，刷新一下就行。
echo   窗口里出现 Application startup complete 就表示好了。
echo.
echo   关闭服务：在这个窗口按 Ctrl+C。
echo ============================================================
echo.

start "" /b "%PY%" scripts\open_when_ready.py
"%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8000

echo.
echo 服务已停止。
goto END

:NO_VENV
echo [错误] 没有找到虚拟环境 .venv
echo.
echo 请先在这个目录下依次执行两条命令，再重新双击本文件：
echo.
echo     python -m venv .venv
echo     .venv\Scripts\pip install -r requirements.txt
echo.

:END
echo.
pause
popd
