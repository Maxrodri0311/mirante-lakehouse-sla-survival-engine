@echo off
echo ================================================================================
echo   Mirante Lakehouse SLA Survival Engine
echo   Automated Execution, Testing and Quantitative Benchmarks
echo ================================================================================
echo.

if not exist "data\lakehouse\silver\fact_pipeline_execution" (
    echo [1/4] Bootstrapping Calibrated Lakehouse Telemetry (2,500 runs)...
    python src/data_generator.py --runs 2500
    if %ERRORLEVEL% NEQ 0 (echo [ERROR] Data generator failed && exit /b %ERRORLEVEL%)
) else (
    echo [1/4] Lakehouse Silver telemetry partition already present.
)

echo.
echo [2/4] Executing Interactive Executive TUI Dashboard...
python src/interface.py
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Interface failed && exit /b %ERRORLEVEL%)

echo.
echo [3/4] Running Automated Pytest Suite (38 tests)...
python -m pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Pytest suite failed && exit /b %ERRORLEVEL%)

echo.
echo [4/4] Running Quantitative Latency Benchmarks (50 iterations)...
python tests/benchmark.py
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Benchmark failed && exit /b %ERRORLEVEL%)

echo.
echo ================================================================================
echo   Execution Complete: All Tests and Latency Targets Passed!
echo ================================================================================