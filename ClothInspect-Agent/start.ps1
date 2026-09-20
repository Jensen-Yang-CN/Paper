# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    ClothInspect-Agent 服装智能质检系统 V1.0 —— 一键启动
.DESCRIPTION
    自动完成：环境探测 → 依赖检查 → 前端构建（首次）→ 启动后端并打开浏览器。
.EXAMPLE
    .\start.ps1
    .\start.ps1 -Port 8080 -NoBrowser
    .\start.ps1 -Python "E:\Miniconda\envs\label-studio\python.exe"
#>
param(
    [int]$Port = 8000,
    [string]$Python = "",
    [switch]$NoBrowser,
    [switch]$Rebuild,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

function Write-Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Write-Ok($t)   { Write-Host "  [OK]   $t" -ForegroundColor Green }
function Write-Warn2($t){ Write-Host "  [提示] $t" -ForegroundColor Yellow }
function Write-Err2($t) { Write-Host "  [错误] $t" -ForegroundColor Red }

Write-Host ""
Write-Host "  ClothInspect-Agent 服装智能质检系统 V1.0" -ForegroundColor White
Write-Host "  视觉模型与多模态大模型协同推理" -ForegroundColor DarkGray
Write-Host "  ------------------------------------------------" -ForegroundColor DarkGray

# ---------------------------------------------------------- 1. 定位 Python
Write-Step "1/4 检测运行环境"
$candidates = @()
if ($Python) { $candidates += $Python }
if ($env:CLOTHINSPECT_PYTHON) { $candidates += $env:CLOTHINSPECT_PYTHON }
$candidates += @(
    "E:\Miniconda\envs\label-studio\python.exe",
    "$env:USERPROFILE\miniconda3\envs\label-studio\python.exe",
    "$env:USERPROFILE\anaconda3\envs\label-studio\python.exe"
)

$py = $null
foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) {
        $ok = & $c -c "import ultralytics, fastapi" 2>&1
        if ($LASTEXITCODE -eq 0) { $py = $c; break }
    }
}
if (-not $py) {
    $sysPy = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($sysPy) {
        & $sysPy -c "import ultralytics, fastapi" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $py = $sysPy }
    }
}

if (-not $py) {
    Write-Err2 "未找到同时安装 ultralytics 与 fastapi 的 Python 环境。"
    Write-Host ""
    Write-Host "  请任选一种方式解决：" -ForegroundColor White
    Write-Host "  1) 指定已有环境：.\start.ps1 -Python `"E:\Miniconda\envs\label-studio\python.exe`""
    Write-Host "  2) 安装依赖到当前环境："
    Write-Host "       cd backend; pip install -r requirements.txt"
    Write-Host "     （ultralytics + torch 体积较大，建议复用已有环境）"
    Write-Host ""
    exit 1
}
Write-Ok "Python: $py"
$pyver = & $py -c "import sys; print(sys.version.split()[0])"
Write-Ok "版本: $pyver"

# ---------------------------------------------------------- 2. 权重与配置
Write-Step "2/4 检查模型权重与配置"
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $Root ".env.example") $envFile
    Write-Warn2 "已生成 .env（从 .env.example 复制）"
}
$keyLine = Select-String -Path $envFile -Pattern '^\s*DASHSCOPE_API_KEY\s*=\s*(.+)$' -ErrorAction SilentlyContinue
if ($keyLine -and $keyLine.Matches[0].Groups[1].Value.Trim().Length -gt 0) {
    Write-Ok "DASHSCOPE_API_KEY 已配置 -> 使用真实 qwen3-vl-plus 复核"
} else {
    Write-Warn2 "未配置 DASHSCOPE_API_KEY -> 将使用离线启发式复核（仅演示流程）"
    Write-Host "         如需真实大模型复核，请编辑 $envFile 填写密钥" -ForegroundColor DarkGray
}

$weightFound = @(
    (Join-Path $Root "..\模型权重\Final_Model_V6_best.pt"),
    (Join-Path $Backend "assets\weights\Final_Model_V6_best.pt"),
    (Join-Path $Backend "assets\weights\best.pt")
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($weightFound) {
    Write-Ok "YOLOv11 权重: $weightFound"
} else {
    Write-Warn2 "未找到 YOLOv11 权重，检测接口将返回 503。"
    Write-Host "         请把 Final_Model_V6_best.pt 放到 backend\assets\weights\" -ForegroundColor DarkGray
    Write-Host "         或在 .env 中设置 YOLO_WEIGHTS=<完整路径>" -ForegroundColor DarkGray
}

# ---------------------------------------------------------- 3. 前端构建
Write-Step "3/4 准备前端"
$dist = Join-Path $Frontend "dist\index.html"
$needBuild = $Rebuild -or (-not (Test-Path $dist))
if (-not $needBuild) {
    Write-Ok "前端已构建（dist/index.html 存在）"
} else {
    if (-not $SkipInstall -and -not (Test-Path (Join-Path $Frontend "node_modules"))) {
        Write-Host "  安装前端依赖（首次较慢，约 1 分钟）..." -ForegroundColor DarkGray
        Push-Location $Frontend
        $env:npm_config_cache = Join-Path $Root ".npm-cache"
        npm install --no-fund --no-audit --ignore-scripts
        Pop-Location
    }
    Write-Host "  构建前端..." -ForegroundColor DarkGray
    Push-Location $Frontend
    $env:npm_config_cache = Join-Path $Root ".npm-cache"
    npm run build
    Pop-Location
    if (Test-Path $dist) { Write-Ok "前端构建完成" } else { Write-Err2 "前端构建失败"; exit 1 }
}

# ---------------------------------------------------------- 4. 启动服务
Write-Step "4/4 启动服务"
$url = "http://127.0.0.1:$Port"
Write-Host "  地址     : $url" -ForegroundColor White
Write-Host "  接口文档 : $url/docs" -ForegroundColor DarkGray
Write-Host "  按 Ctrl+C 停止服务`n" -ForegroundColor DarkGray

if (-not $NoBrowser) {
    # 启动一个独立的隐藏进程，等 6 秒（服务起来了）再打开浏览器。
    # 不用 Start-Job：作业失败会中断整个脚本，退出时还要额外清理。
    try {
        Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList @(
            "-NoProfile", "-Command", "Start-Sleep -Seconds 6; Start-Process '$url'"
        ) | Out-Null
        Write-Host "  浏览器将在 6 秒后自动打开" -ForegroundColor DarkGray
    } catch {
        Write-Warn2 "自动打开浏览器失败，请手动访问 $url"
    }
}

$env:PYTHONIOENCODING = "utf-8"
Push-Location $Backend
try {
    & $py run.py --port $Port
} finally {
    Pop-Location
}
