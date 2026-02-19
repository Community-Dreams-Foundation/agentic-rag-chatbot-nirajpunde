@echo off
REM Windows equivalent of "make sanity" - run from project root
cd /d "%~dp0"
echo == Sanity Check ==
python scripts/run_sanity.py
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
python scripts/verify_output.py artifacts/sanity_output.json
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
echo OK: sanity check passed
