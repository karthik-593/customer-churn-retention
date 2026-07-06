@echo off
REM tiny shim so `.\make <target>` works in PowerShell/cmd without GNU make.
REM mirrors the Makefile. usage:  .\make all   |   .\make score   |   .\make monitor   |   .\make serve
setlocal
if not defined PY set PY=python
set T=%~1
if "%T%"=="" set T=all

if "%T%"=="features"    ( %PY% -m src.features & exit /b )
if "%T%"=="train"       ( %PY% -m src.train & exit /b )
if "%T%"=="export"      ( %PY% -m src.export_model & exit /b )
if "%T%"=="score"       ( %PY% -m src.score 2017-02-01 & exit /b )
if "%T%"=="serve"       ( %PY% -m uvicorn api.main:app --reload & exit /b )
if "%T%"=="monitor"     goto monitor
if "%T%"=="calibration" ( %PY% -m monitoring.calibration & exit /b )
if "%T%"=="smoke"       goto smoke
if "%T%"=="clean"       goto clean
if "%T%"=="all"         goto all
echo unknown target "%T%" -- try: features train export score serve monitor calibration all smoke clean
exit /b 1

:all
%PY% -m src.features || exit /b 1
%PY% -m src.train || exit /b 1
%PY% -m src.export_model || exit /b 1
%PY% -m src.score 2017-02-01 || exit /b 1
exit /b

:monitor
REM leading monitors: tier-0 gate first (non-zero exit stops here), then tier-1 drift
%PY% -m monitoring.data_quality 2017-02-01 || exit /b 1
%PY% -m monitoring.score_drift 2017-02-01 || exit /b 1
exit /b

:smoke
%PY% -m tests.smoke_test || exit /b 1
%PY% -m tests.smoke_train || exit /b 1
%PY% -m tests.smoke_score || exit /b 1
%PY% -m tests.smoke_full_scorer || exit /b 1
%PY% -m tests.smoke_api || exit /b 1
%PY% -m tests.smoke_data_quality || exit /b 1
%PY% -m tests.smoke_score_drift || exit /b 1
%PY% -m tests.smoke_calibration || exit /b 1
exit /b

:clean
if exist mlflow.db del /q mlflow.db
if exist mlruns rmdir /s /q mlruns
if exist mlartifacts rmdir /s /q mlartifacts
if exist api\model rmdir /s /q api\model
if exist reports rmdir /s /q reports
exit /b
