# ============================================================
# AI Camera - Run Script (PowerShell)
# Khoi chay Backend + Frontend
# ============================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         AI Camera - Run Script                         " -ForegroundColor Cyan
Write-Host "         Khoi chay Backend + Frontend                   " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Kiem tra moi truong ao ton tai
# ============================================================
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[LOI] Khong tim thay moi truong ao .venv!" -ForegroundColor Red
    Write-Host "Vui long chay .\setup.ps1 truoc." -ForegroundColor Red
    Write-Host ""
    Read-Host "Nhan Enter de thoat"
    exit 1
}

# ============================================================
# Kich hoat moi truong ao
# ============================================================
if ($env:VIRTUAL_ENV) {
    Write-Host "[OK] Moi truong ao da duoc kich hoat: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "[INFO] Dang kich hoat moi truong ao..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
    Write-Host "[OK] Da kich hoat moi truong ao." -ForegroundColor Green
}
Write-Host ""

# ============================================================
# Kiem tra dependencies da cai chua
# ============================================================
$checkResult = python -c "import fastapi; import cv2; import mediapipe" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[CANH BAO] Mot so thu vien chua duoc cai dat." -ForegroundColor DarkYellow
    Write-Host "Dang chay cai dat tu dong..." -ForegroundColor Yellow
    pip install -r requirements.txt --quiet
    Write-Host "[OK] Da cai dat xong." -ForegroundColor Green
    Write-Host ""
}

# ============================================================
# Khoi chay Backend (terminal rieng)
# ============================================================
Write-Host "[BACKEND] Khoi chay Backend API..." -ForegroundColor Magenta
Write-Host "          URL:     http://127.0.0.1:8000" -ForegroundColor White
Write-Host "          Swagger: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""

$projectRoot = (Get-Location).Path

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$projectRoot'; & '.\`.venv\Scripts\Activate.ps1'; Write-Host ''; Write-Host '========================================' -ForegroundColor Cyan; Write-Host '  BACKEND - AI Camera API' -ForegroundColor Cyan; Write-Host '  http://127.0.0.1:8000' -ForegroundColor Cyan; Write-Host '========================================' -ForegroundColor Cyan; Write-Host ''; uvicorn main:app --reload"
)

# Cho Backend khoi dong truoc
Write-Host "   Cho Backend khoi dong (3 giay)..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# ============================================================
# Khoi chay Frontend (terminal rieng)
# ============================================================
if (Test-Path "frontend\package.json") {
    Write-Host "[FRONTEND] Khoi chay Frontend..." -ForegroundColor Magenta
    Write-Host "           URL: http://localhost:3000" -ForegroundColor White
    Write-Host ""

    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "Set-Location '$projectRoot\frontend'; Write-Host ''; Write-Host '========================================' -ForegroundColor Green; Write-Host '  FRONTEND - AI Camera UI' -ForegroundColor Green; Write-Host '  http://localhost:3000' -ForegroundColor Green; Write-Host '========================================' -ForegroundColor Green; Write-Host ''; npm run dev"
    )
} else {
    Write-Host "[THONG BAO] Khong tim thay frontend/package.json, bo qua Frontend." -ForegroundColor Gray
}

# ============================================================
# Thong bao va mo trinh duyet
# ============================================================
Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Dang khoi chay...                                     " -ForegroundColor Green
Write-Host "                                                        " -ForegroundColor Green
Write-Host "  Backend:  http://127.0.0.1:8000                       " -ForegroundColor White
Write-Host "  Swagger:  http://127.0.0.1:8000/docs                  " -ForegroundColor White
Write-Host "  Frontend: http://localhost:3000                        " -ForegroundColor White
Write-Host "                                                        " -ForegroundColor Green
Write-Host "  Dong cua so nay se KHONG tat Backend/Frontend.        " -ForegroundColor Yellow
Write-Host "  De dung, dong tung cua so terminal tuong ung.         " -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""

# Mo trinh duyet sau 5 giay
Write-Host "Mo trinh duyet sau 5 giay..." -ForegroundColor Gray
Start-Sleep -Seconds 5
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "Hoan tat! Ban co the dong cua so nay." -ForegroundColor Green
