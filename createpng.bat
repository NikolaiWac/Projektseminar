@echo off
setlocal EnableDelayedExpansion

:: ---------------------------------------------------------
:: PNG Generator Script (Fixed for Parentheses Issue)
:: ---------------------------------------------------------

echo.
echo  --- PNG Batch Generator ---
echo.

:: 1. Get User Input
:AskCount
set /p "count=Enter the number of files to create: "
:: Basic validation
echo %count%| findstr /r "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo Invalid number. Please enter a positive integer.
    goto AskCount
)

:AskRes
set /p "width=Enter Width (px): "
set /p "height=Enter Height (px): "

:: Define output directory (Same folder as batch file + "data")
set "dataDir=%~dp0data"
if not exist "%dataDir%" mkdir "%dataDir%"

echo.
echo Generating %count% files with resolution %width%x%height%...
echo Saving to: "%dataDir%"
echo ---------------------------------------------------------

:: 2. Create a temporary PowerShell script
:: We write line-by-line to prevent Batch from misinterpreting the ")" characters.
set "psScript=%temp%\MakePngs.ps1"
if exist "%psScript%" del "%psScript%"

echo param($count, $width, $height, $targetDir) >> "%psScript%"
echo Add-Type -AssemblyName System.Drawing >> "%psScript%"
echo for ($i=1; $i -le $count; $i++) { >> "%psScript%"
echo     $modulo = $i %% 2 >> "%psScript%"
echo     if ($modulo -eq 1) { $suffix = 0 } else { $suffix = 1 } >> "%psScript%"
echo     $filename = "$i $suffix.png" >> "%psScript%"
echo     $path = Join-Path -Path $targetDir -ChildPath $filename >> "%psScript%"
echo     try { >> "%psScript%"
echo         $bmp = New-Object System.Drawing.Bitmap([int]$width, [int]$height) >> "%psScript%"
echo         $g = [System.Drawing.Graphics]::FromImage($bmp) >> "%psScript%"
echo         $g.Clear([System.Drawing.Color]::White) >> "%psScript%"
echo         $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png) >> "%psScript%"
echo         $g.Dispose() >> "%psScript%"
echo         $bmp.Dispose() >> "%psScript%"
echo         Write-Host "Created: $filename" >> "%psScript%"
echo     } catch { >> "%psScript%"
echo         Write-Error "Failed to create $filename : $_" >> "%psScript%"
echo     } >> "%psScript%"
echo } >> "%psScript%"

:: 3. Execute the temporary script
powershell -ExecutionPolicy Bypass -File "%psScript%" -count %count% -width %width% -height %height% -targetDir "%dataDir%"

:: 4. Cleanup
if exist "%psScript%" del "%psScript%"

echo.
echo ---------------------------------------------------------
echo Done! Files have been created in the "data" folder.
pause