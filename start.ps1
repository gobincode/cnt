$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$server = Join-Path $project "server"

if (-not (Test-Path -LiteralPath (Join-Path $project "node_modules"))) {
    throw "Frontend dependencies are missing. Run npm install from $project first."
}

Push-Location $server
try {
    python -c "import fastapi, uvicorn, pydicom, multipart, torch"
    if ($LASTEXITCODE -ne 0) {
        throw "Backend dependencies are missing. Run python -m pip install -r server/requirements.txt first."
    }
} finally {
    Pop-Location
}

$api = Start-Process -FilePath "python" -ArgumentList @(
    "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $server -PassThru -NoNewWindow

try {
    Push-Location $project
    Write-Host "Starting KneeScan. Open the URL printed by Vite. Ctrl+C stops the demo."
    npm run dev
} finally {
    Pop-Location
    if (-not $api.HasExited) {
        Stop-Process -Id $api.Id
    }
}
