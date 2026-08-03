@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem MLflow UI launcher for Bike-Sharing-Demand.
rem Usage:
rem   start_mlflow.bat
rem   start_mlflow.bat 5001
rem   start_mlflow.bat 5001 --no-browser
rem Optional override:
rem   set BIKE_SHARING_PYTHON=C:\caminho\para\python.exe

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE="

if defined BIKE_SHARING_PYTHON if exist "%BIKE_SHARING_PYTHON%" (
    set "PYTHON_EXE=%BIKE_SHARING_PYTHON%"
)

if not defined PYTHON_EXE if exist "%USERPROFILE%\miniforge3\envs\Bike-Sharing\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\miniforge3\envs\Bike-Sharing\python.exe"
)

if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\Bike-Sharing\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\Bike-Sharing\python.exe"
)

if not defined PYTHON_EXE if /I "%CONDA_DEFAULT_ENV%"=="Bike-Sharing" if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
)

if not defined PYTHON_EXE (
    for /F "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
set "MLRUNS_DIR=%PROJECT_DIR%mlruns"
set "MLRUNS_PATH=%MLRUNS_DIR:\=/%"
set "MLRUNS_URI=file:///%MLRUNS_PATH%"
set "HOST=127.0.0.1"
set "PORT=%~1"
set "OPEN_BROWSER=1"

if not defined PORT set "PORT=5000"
if /I "%~2"=="--no-browser" set "OPEN_BROWSER=0"
set "MLFLOW_URL=http://%HOST%:%PORT%"

echo(%PORT%| findstr /R "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo [ERRO] Porta invalida: "%PORT%".
    echo Uso: start_mlflow.bat [porta] [--no-browser]
    exit /b 2
)

if not defined PYTHON_EXE (
    echo [ERRO] Nenhum interpretador Python foi encontrado.
    echo Ative o ambiente Bike-Sharing ou defina BIKE_SHARING_PYTHON.
    exit /b 3
)

if not exist "%PYTHON_EXE%" (
    echo [ERRO] O interpretador configurado nao existe: %PYTHON_EXE%
    exit /b 3
)

pushd "%PROJECT_DIR%" >nul
"%PYTHON_EXE%" -c "from src.environment import require_environment; require_environment(); import mlflow" >nul 2>&1
set "ENV_CHECK=%ERRORLEVEL%"
popd >nul
if not "%ENV_CHECK%"=="0" (
    echo [ERRO] O interpretador encontrado nao pertence ao ambiente Bike-Sharing
    echo        ou nao possui as dependencias auditadas do projeto.
    echo        Defina BIKE_SHARING_PYTHON com o caminho correto.
    exit /b 4
)

if not exist "%MLRUNS_DIR%" mkdir "%MLRUNS_DIR%"

powershell -NoProfile -Command ^
    "if (Get-NetTCPConnection -LocalAddress '%HOST%' -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    echo [ERRO] A porta %PORT% ja esta ocupada.
    echo Tente outra porta, por exemplo: start_mlflow.bat 5001
    exit /b 5
)

echo.
echo Iniciando MLflow para Bike-Sharing-Demand...
echo Ambiente : Bike-Sharing
echo Python   : %PYTHON_EXE%
echo Backend  : %MLRUNS_URI%
echo Interface: %MLFLOW_URL%
echo.
echo Para encerrar, pressione Ctrl+C ou feche a janela "MLflow - Bike Sharing".

start "MLflow - Bike Sharing" /D "%PROJECT_DIR%" ^
    "%PYTHON_EXE%" -m mlflow server ^
    --backend-store-uri "%MLRUNS_URI%" ^
    --default-artifact-root "%MLRUNS_URI%" ^
    --no-serve-artifacts ^
    --host "%HOST%" ^
    --port "%PORT%"

for /L %%I in (1,1,30) do (
    powershell -NoProfile -Command ^
        "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%MLFLOW_URL%' -TimeoutSec 1; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 goto :ready
    timeout /T 1 /NOBREAK >nul
)

echo [ERRO] O servidor nao respondeu em %MLFLOW_URL% apos 30 segundos.
echo Consulte a janela "MLflow - Bike Sharing" para ver o erro de inicializacao.
exit /b 6

:ready
echo [OK] MLflow esta disponivel em %MLFLOW_URL%
if "%OPEN_BROWSER%"=="1" start "" "%MLFLOW_URL%"
exit /b 0
