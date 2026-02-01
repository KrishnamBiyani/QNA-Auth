$python = "C:\Python312\python.exe"
Write-Host "Starting Server..."
$serverProcess = Start-Process $python -ArgumentList "server/app.py" -PassThru -WindowStyle Minimized
Write-Host "Server PID: $($serverProcess.Id)"

Write-Host "Waiting for server to be ready..."
for ($i=0; $i -lt 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method Get -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "Server is Ready!"
            break
        }
    } catch {
        # Ignore connection errors
    }
    Start-Sleep -Seconds 1
}

Write-Host "Running Test Script..."
& $python test_enrollment.py

Write-Host "Cleaning up..."
Stop-Process -Id $serverProcess.Id -Force
