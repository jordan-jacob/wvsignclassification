@echo off
setlocal
cd /d "%~dp0"

echo.
echo  WV Sign Mapper -- Build executable
echo  ====================================
echo.
echo  This will create dist\WVSignMapper\WVSignMapper.exe
echo  The folder will be ~1.5-2.5 GB (PyTorch dominates).
echo  Build time: 5-15 minutes depending on hardware.
echo.

pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: Could not install PyInstaller. Check your Python environment.
    pause
    exit /b 1
)

pyinstaller sign_mapper.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: Build failed. See output above for details.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Build complete!
echo.
echo   Executable folder : dist\WVSignMapper\
echo   Launch with       : dist\WVSignMapper\WVSignMapper.exe
echo.
echo   Zip dist\WVSignMapper\ to distribute to users.
echo   They just unzip and double-click WVSignMapper.exe.
echo  ============================================================
echo.
pause
