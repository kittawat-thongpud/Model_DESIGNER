<#
server.ps1 - manage Model DESIGNER server session via pmux (psmux)

Usage (in PowerShell):
  .\server.ps1 start     - start server in background pmux session
  .\server.ps1 attach    - attach to running session (Ctrl+B then D to detach)
  .\server.ps1 logs      - tail server log file
  .\server.ps1 stop      - kill server session
  .\server.ps1 status    - show if session is running
  .\server.ps1 restart   - restart server without git pull
  .\server.ps1 update    - git pull latest code, build frontend if present, and restart
#>

param (
    [Parameter(Position=0)]
    [string]$Action = "help"
)

# === Configuration ===
$SESSION     = "model-designer"
$APP_DIR     = $PSScriptRoot
$LOG_FILE    = Join-Path $APP_DIR "server.log"
$LAUNCHER    = Join-Path $APP_DIR "pmux_launcher.ps1"

# Command to run the app inside the launcher
$CMD = "Set-Location '$APP_DIR'; python run.py *>&1 | Tee-Object -FilePath '$LOG_FILE'"

# === Helpers ===

function Ensure-PmuxAvailable {
    if (-not (Get-Command pmux -ErrorAction SilentlyContinue)) {
        Write-Host "[ERROR] 'pmux' command not found in PATH. Install pmux or add it to PATH." -ForegroundColor Red
        exit 1
    }
}

function Test-PmuxSession {
    # returns $true when session exists
    pmux has-session -t $SESSION 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Start-PmuxSession {
    param(
        [string]$SessionName,
        [string]$AppDir,
        [string]$LogFile,
        [string]$LauncherPath
    )

    # Create launcher script that pmux will execute
$launcherContent = @"
Set-Location '$AppDir'

$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

python run.py *>&1 | Tee-Object -FilePath '$LogFile'
"@.Trim()

    try {
        $launcherContent | Set-Content -LiteralPath $LauncherPath -Encoding utf8 -Force
    } catch {
        Write-Host "[ERROR] Failed to write launcher script: $LauncherPath" -ForegroundColor Red
        throw
    }

    $pmuxArg = "powershell -NoExit -File `"$LauncherPath`""

    pmux new-session -d -s $SessionName $pmuxArg
    return $LASTEXITCODE
}

# === Usage / help ===
if ($Action -match "^(--help|-h|help)$" -or [string]::IsNullOrWhiteSpace($Action)) {
    Write-Host "Usage: .\server.ps1 {start|attach|logs|stop|status|restart|update}" -ForegroundColor Cyan
    exit 0
}

# Ensure pmux exists for commands that need it
switch ($Action) {
    'start'    { Ensure-PmuxAvailable }
    'restart'  { Ensure-PmuxAvailable }
    'update'   { Ensure-PmuxAvailable }
    default    { } 
}

# === Main control flow ===
if ($Action -eq "start") {
    if (Test-PmuxSession) {
        Write-Host "[WARN] Session '$SESSION' is already running." -ForegroundColor Yellow
        Write-Host "   Use: .\server.ps1 attach  - to view it"
        Write-Host "   Use: .\server.ps1 stop    - to stop it first"
        exit 0
    }

    $rc = Start-PmuxSession -SessionName $SESSION -AppDir $APP_DIR -LogFile $LOG_FILE -LauncherPath $LAUNCHER
    Start-Sleep -Seconds 1

    if (Test-PmuxSession) {
        Write-Host "[OK] Server started in pmux session '$SESSION'." -ForegroundColor Green
        Write-Host "   App: http://localhost:8000"
        Write-Host "   Log: $LOG_FILE"
        Write-Host "   Use: .\server.ps1 attach  - to view live output"
        exit 0
    } else {
        Write-Host "[ERROR] Failed to start pmux session. Check log: $LOG_FILE" -ForegroundColor Red
        exit 1
    }
}
elseif ($Action -eq "attach") {
    if (Test-PmuxSession) {
        Write-Host "[INFO] Attaching to session '$SESSION' (press Ctrl+B then D to detach)..." -ForegroundColor Cyan
        pmux attach-session -t $SESSION
        exit $LASTEXITCODE
    } else {
        Write-Host "[ERROR] No session '$SESSION' running. Use: .\server.ps1 start" -ForegroundColor Red
        exit 1
    }
}
elseif ($Action -eq "logs") {
    if (Test-Path $LOG_FILE) {
        Write-Host "[INFO] Tailing $LOG_FILE (Ctrl+C to stop)..." -ForegroundColor Cyan
        Get-Content $LOG_FILE -Wait -Tail 200
        exit 0
    } else {
        Write-Host "[ERROR] Log file not found: $LOG_FILE" -ForegroundColor Red
        exit 1
    }
}
elseif ($Action -eq "stop") {
    if (Test-PmuxSession) {
        pmux kill-session -t $SESSION
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Session '$SESSION' stopped." -ForegroundColor Yellow
            exit 0
        } else {
            Write-Host "[ERROR] Failed to stop session '$SESSION'." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[WARN] No session '$SESSION' found." -ForegroundColor Yellow
        exit 0
    }
}
elseif ($Action -eq "status") {
    if (Test-PmuxSession) {
        Write-Host "[OK] Session '$SESSION' is running." -ForegroundColor Green
        Write-Host "   Use: .\server.ps1 attach  - to view"
        Write-Host "   Use: .\server.ps1 logs    - to tail log"
        Write-Host "   Use: .\server.ps1 stop    - to stop"
        exit 0
    } else {
        Write-Host "[INFO] Session '$SESSION' is NOT running." -ForegroundColor DarkGray
        Write-Host "   Use: .\server.ps1 start   - to start"
        exit 0
    }
}
elseif ($Action -eq "restart") {
    if (Test-PmuxSession) {
        Write-Host "[INFO] Stopping session '$SESSION'..." -ForegroundColor Cyan
        pmux kill-session -t $SESSION
        Start-Sleep -Seconds 1
    }

    $rc = Start-PmuxSession -SessionName $SESSION -AppDir $APP_DIR -LogFile $LOG_FILE -LauncherPath $LAUNCHER
    Start-Sleep -Seconds 1

    if (Test-PmuxSession) {
        Write-Host "[OK] Server restarted in pmux session '$SESSION'." -ForegroundColor Green
        Write-Host "   Use: .\server.ps1 logs  - to verify"
        exit 0
    } else {
        Write-Host "[ERROR] Failed to restart. Check: $LOG_FILE" -ForegroundColor Red
        exit 1
    }
}
elseif ($Action -eq "update") {
    Write-Host "[INFO] Pulling latest code from git..." -ForegroundColor Cyan
    try {
        git -C $APP_DIR pull
    } catch {
        Write-Host "[WARN] git pull returned an error or git not found. Continuing." -ForegroundColor Yellow
    }

    $FRONTEND_DIR = Join-Path $APP_DIR "frontend"
    if (Test-Path $FRONTEND_DIR) {
        if ((Get-Command node -ErrorAction SilentlyContinue) -and (Get-Command npm -ErrorAction SilentlyContinue)) {
            $nodeVer = (& node --version) 2>$null
            $npmVer  = (& npm --version) 2>$null
            Write-Host "[INFO] Node: $nodeVer   npm: $npmVer"
            Write-Host "[INFO] Installing npm dependencies..." -ForegroundColor Cyan
            npm --prefix "$FRONTEND_DIR" install
            Write-Host "[INFO] Building frontend..." -ForegroundColor Cyan
            npm --prefix "$FRONTEND_DIR" run build
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Frontend build failed. Aborting restart." -ForegroundColor Red
                exit 1
            } else {
                Write-Host "[OK] Frontend built successfully." -ForegroundColor Green
            }
        } else {
            Write-Host "[WARN] Node.js/npm not found. Skipping frontend build." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[INFO] No frontend/ directory found - skipping build." -ForegroundColor Yellow
    }

    if (Test-PmuxSession) {
        Write-Host "[INFO] Restarting server..." -ForegroundColor Cyan
        pmux kill-session -t $SESSION
        Start-Sleep -Seconds 1
    }

    $rc = Start-PmuxSession -SessionName $SESSION -AppDir $APP_DIR -LogFile $LOG_FILE -LauncherPath $LAUNCHER
    Start-Sleep -Seconds 1

    if (Test-PmuxSession) {
        Write-Host "[OK] Server updated and restarted." -ForegroundColor Green
        Write-Host "   Use: .\server.ps1 logs  - to verify"
        exit 0
    } else {
        Write-Host "[ERROR] Failed to restart after update. Check: $LOG_FILE" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "[ERROR] Unknown command: $Action" -ForegroundColor Red
    Write-Host "Usage: .\server.ps1 {start|attach|logs|stop|status|restart|update}" -ForegroundColor Cyan
    exit 2
}