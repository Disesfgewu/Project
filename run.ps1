# run_model.ps1
# Define the path of the Python file
$scriptPath = "C:\Users\marti\Desktop\ai\ntut_AICUP\src\ano_main.py"

# Check if Python is available
Write-Host "Checking Python environment..."
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python is not installed or not added to the PATH environment variable." -ForegroundColor Red
    exit 1
}

# Execute each parameter set line by line
Write-Host "Running parameters: epoch=100, batch_size=32..."
python $scriptPath --epoch 100 --batch_size 32
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=100, batch_size=32!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=100, batch_size=32! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=75, batch_size=64..."
python $scriptPath --epoch 75 --batch_size 64
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=75, batch_size=64!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=75, batch_size=64! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=100, batch_size=64..."
python $scriptPath --epoch 100 --batch_size 64
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=100, batch_size=64!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=100, batch_size=64! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=150, batch_size=64..."
python $scriptPath --epoch 150 --batch_size 64
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=150, batch_size=64!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=150, batch_size=64! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=200, batch_size=64..."
python $scriptPath --epoch 200 --batch_size 64
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=200, batch_size=64!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=200, batch_size=64! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=75, batch_size=128..."
python $scriptPath --epoch 75 --batch_size 128
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=75, batch_size=128!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=75, batch_size=128! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=100, batch_size=128..."
python $scriptPath --epoch 100 --batch_size 128
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=100, batch_size=128!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=100, batch_size=128! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=150, batch_size=128..."
python $scriptPath --epoch 150 --batch_size 128
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=150, batch_size=128!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=150, batch_size=128! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=200, batch_size=128..."
python $scriptPath --epoch 200 --batch_size 128
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=200, batch_size=128!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=200, batch_size=128! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=75, batch_size=256..."
python $scriptPath --epoch 75 --batch_size 256
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=75, batch_size=256!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=75, batch_size=256! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=100, batch_size=256..."
python $scriptPath --epoch 100 --batch_size 256
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=100, batch_size=256!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=100, batch_size=256! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=150, batch_size=256..."
python $scriptPath --epoch 150 --batch_size 256
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=150, batch_size=256!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=150, batch_size=256! Please check the error message." -ForegroundColor Red
}

Write-Host "Running parameters: epoch=200, batch_size=256..."
python $scriptPath --epoch 200 --batch_size 256
if ($LASTEXITCODE -eq 0) {
    Write-Host "Script ran successfully with parameters epoch=200, batch_size=256!" -ForegroundColor Green
} else {
    Write-Host "Script failed with parameters epoch=200, batch_size=256! Please check the error message." -ForegroundColor Red
}
