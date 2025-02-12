@echo off
setlocal enabledelayedexpansion

:: Set environment variables
set PYTHON_PATH=python
set SCRIPT_PATH=src\tokenizer.py
set LOG_DIR=C:\Users\ASUS\Desktop\SPARSA-LM-Base 0.1\logs
set VOCAB_SIZE=8192
set MIN_FREQ=2
set DATASET_CONFIG=C:\Users\ASUS\Desktop\SPARSA-LM-Base 0.1\config\dataset_config.yaml
set CACHE_DIR=C:\Users\ASUS\Desktop\SPARSA-LM-Base 0.1\data\processed\cache

:: Create logs directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Check Python installation
%PYTHON_PATH% --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please ensure Python is installed and in PATH.
    exit /b 1
)

:: Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo Error: Tokenizer script not found at %SCRIPT_PATH%.
    exit /b 1
)

:: Check CUDA availability
echo Checking hardware configuration...
%PYTHON_PATH% -c "import torch; print('Using ' + ('CUDA' if torch.cuda.is_available() else 'CPU') + ' for tokenizer training')"
if errorlevel 1 (
    echo Error: Unable to determine hardware configuration. Ensure PyTorch is installed correctly.
    exit /b 1
)
echo.

:: Run the tokenizer script with specified parameters
echo Starting tokenizer training with the following configuration:
echo - Vocabulary Size: %VOCAB_SIZE%
echo - Minimum Frequency: %MIN_FREQ%
echo - Dataset Config: %DATASET_CONFIG%
echo - Cache Directory: %CACHE_DIR%
echo - Log Directory: %LOG_DIR%
echo.

:: Run command and display output while logging
%PYTHON_PATH% %SCRIPT_PATH% --vocab-size %VOCAB_SIZE% --min-frequency %MIN_FREQ% --dataset-config "%DATASET_CONFIG%" --cache-dir "%CACHE_DIR%" --log-dir "%LOG_DIR%" > "%LOG_DIR%\tokenizer_output.log" 2>&1 & type "%LOG_DIR%\tokenizer_output.log"

if errorlevel 1 (
    echo Error: Tokenizer training failed. Check %LOG_DIR%\tokenizer.log for details.
    exit /b 1
) else (
    echo Tokenizer training completed successfully. Logs available in %LOG_DIR%
)

endlocal
