# ============================================================
# AI Camera - Setup Script (PowerShell)
# Kiem tra moi truong va cai dat dependencies
# ============================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         AI Camera - Setup Script                       " -ForegroundColor Cyan
Write-Host "         Kiem tra moi truong va cai dat dependencies    " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# BUOC 1: Kiem tra Python da cai dat chua
# ============================================================
Write-Host "[1/4] Kiem tra Python..." -ForegroundColor Yellow

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "   [LOI] Khong tim thay Python!" -ForegroundColor Red
    Write-Host "   Vui long cai dat Python 3.11.9 tu:" -ForegroundColor Red
    Write-Host "   https://www.python.org/downloads/release/python-3119/" -ForegroundColor White
    Write-Host "   Nho tick 'Add Python to PATH' khi cai dat." -ForegroundColor White
    Read-Host "Nhan Enter de thoat"
    exit 1
}

# ============================================================
# BUOC 2: Kiem tra phien ban Python
# ============================================================
Write-Host "[2/4] Kiem tra phien ban Python..." -ForegroundColor Yellow

$pythonVersion = (python --version 2>&1) -replace "Python ", ""
$versionParts = $pythonVersion.Split(".")
$major = [int]$versionParts[0]
$minor = [int]$versionParts[1]

Write-Host "   Phien ban hien tai: Python $pythonVersion"

# Kiem tra Python 3.x
if ($major -ne 3) {
    Write-Host "   [LOI] Can Python 3.x, hien tai la Python $major.x" -ForegroundColor Red
    Write-Host "   Vui long cai dat Python 3.11.9" -ForegroundColor Red
    Read-Host "Nhan Enter de thoat"
    exit 1
}

# Kiem tra Python >= 3.13
if ($minor -ge 13) {
    Write-Host ""
    Write-Host "   ========================================================" -ForegroundColor DarkYellow
    Write-Host "   [CANH BAO] Python $pythonVersion co the gap loi tuong thich!" -ForegroundColor DarkYellow
    Write-Host "   ========================================================" -ForegroundColor DarkYellow
    Write-Host ""
    Write-Host "   Thu vien mediapipe va opencv-python chua ho tro day du" -ForegroundColor White
    Write-Host "   Python 3.13+. De xuat:" -ForegroundColor White
    Write-Host ""
    Write-Host "   -> Tai Python 3.11.9 tai:" -ForegroundColor Green
    Write-Host "      https://www.python.org/downloads/release/python-3119/" -ForegroundColor White
    Write-Host ""
    Write-Host "   Nho tick 'Add Python to PATH' khi cai dat." -ForegroundColor White
    Write-Host ""

    $continue = Read-Host "   Ban co muon tiep tuc cai dat khong? (y/n)"
    if ($continue -ne "y") {
        Write-Host "   Da huy cai dat." -ForegroundColor Gray
        exit 1
    }
    Write-Host "   Tiep tuc cai dat voi Python $pythonVersion..." -ForegroundColor Yellow
    Write-Host ""
}

# Kiem tra Python < 3.10
if ($minor -lt 10) {
    Write-Host "   [LOI] Can Python 3.10 tro len, hien tai la Python $pythonVersion" -ForegroundColor Red
    Write-Host "   Vui long cai dat Python 3.11.9" -ForegroundColor Red
    Read-Host "Nhan Enter de thoat"
    exit 1
}

Write-Host "   [OK] Python $pythonVersion tuong thich." -ForegroundColor Green
Write-Host ""

# ============================================================
# BUOC 3: Tao moi truong ao (venv)
# ============================================================
Write-Host "[3/4] Thiet lap moi truong ao..." -ForegroundColor Yellow

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "   [OK] Moi truong ao da ton tai." -ForegroundColor Green
} else {
    Write-Host "   Dang tao moi truong ao .venv..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   [LOI] Khong the tao moi truong ao!" -ForegroundColor Red
        Read-Host "Nhan Enter de thoat"
        exit 1
    }
    Write-Host "   [OK] Da tao moi truong ao .venv" -ForegroundColor Green
}
Write-Host ""

# ============================================================
# BUOC 4: Cai dat dependencies
# ============================================================
Write-Host "[4/4] Cai dat cac thu vien can thiet..." -ForegroundColor Yellow
Write-Host ""

# Kich hoat venv
& .\.venv\Scripts\Activate.ps1

# Nang cap pip
Write-Host "   Dang nang cap pip..."
python -m pip install --upgrade pip --quiet 2>$null

# Cai dat backend dependencies
Write-Host "   Dang cai dat Backend dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "   [LOI] Khong the cai dat Backend dependencies!" -ForegroundColor Red
    Read-Host "Nhan Enter de thoat"
    exit 1
}
Write-Host "   [OK] Backend dependencies da cai dat xong." -ForegroundColor Green
Write-Host ""

# Cai dat frontend dependencies
if (Test-Path "frontend\package.json") {
    Write-Host "   Dang cai dat Frontend dependencies..."

    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Host "   [CANH BAO] Khong tim thay npm! Bo qua cai dat Frontend." -ForegroundColor DarkYellow
        Write-Host "   Cai dat Node.js tu: https://nodejs.org/" -ForegroundColor White
    } else {
        Push-Location frontend
        npm install
        Pop-Location
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   [LOI] Khong the cai dat Frontend dependencies!" -ForegroundColor Red
            Read-Host "Nhan Enter de thoat"
            exit 1
        }
        Write-Host "   [OK] Frontend dependencies da cai dat xong." -ForegroundColor Green
    }
} else {
    Write-Host "   [THONG BAO] Khong tim thay frontend/package.json, bo qua Frontend." -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  [HOAN TAT] Cai dat thanh cong!                       " -ForegroundColor Green
Write-Host "                                                        " -ForegroundColor Green
Write-Host "  Chay ung dung bang lenh:  .\run.ps1                   " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Nhan Enter de thoat"
