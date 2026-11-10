@echo off
REM End-to-end pipeline: data -> train -> evaluate -> tests -> launch app.
REM Usage: scripts\run_all.bat [demo^|full] [--low-vram] [--no-app]
setlocal
cd /d "%~dp0\.."
set MODE=%1
if "%MODE%"=="" set MODE=demo
set LOWVRAM=
set LAUNCH=1
for %%A in (%*) do (
  if "%%A"=="--low-vram" set LOWVRAM=--low-vram
  if "%%A"=="--no-app" set LAUNCH=0
)
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

echo == [1/7] GPU check
%PY% scripts\check_gpu.py || echo (continuing on CPU)
echo == [2/7] Datasets
%PY% -m src.data.download_data || goto :error
echo == [3/7] YOLO dataset
%PY% -m src.data.prepare_yolo_dataset || goto :error
echo == [4/7] Water model (%MODE%)
%PY% -m src.water.train --mode %MODE% || goto :error
%PY% -m src.water.evaluate || goto :error
echo == [5/7] Waste model (%MODE%)
%PY% -m src.waste.train --mode %MODE% %LOWVRAM% || goto :error
%PY% -m src.waste.evaluate || goto :error
echo == [6/7] Tests
%PY% -m pytest -q || goto :error
if "%LAUNCH%"=="1" (
  echo == [7/7] Launching Streamlit on http://localhost:8501
  %PY% -m streamlit run app\streamlit_app.py
)
goto :eof
:error
echo Pipeline failed with error %errorlevel%
exit /b %errorlevel%
