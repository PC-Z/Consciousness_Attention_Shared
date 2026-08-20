[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeRoot = Join-Path $projectRoot '.runtime'
$uvRoot = Join-Path $runtimeRoot 'uv'
$pythonRoot = Join-Path $runtimeRoot 'python'
$cacheRoot = Join-Path $runtimeRoot 'cache'
$tempRoot = Join-Path $runtimeRoot 'temp'
$venvRoot = Join-Path $projectRoot '.envs\attention'

New-Item -ItemType Directory -Force -Path $uvRoot,$pythonRoot,$cacheRoot,$tempRoot | Out-Null

$env:UV_UNMANAGED_INSTALL = $uvRoot
$env:UV_NO_MODIFY_PATH = '1'
$env:UV_PYTHON_INSTALL_DIR = $pythonRoot
$env:UV_PYTHON_BIN_DIR = Join-Path $pythonRoot 'bin'
$env:UV_PYTHON_INSTALL_REGISTRY = '0'
$env:UV_CACHE_DIR = $cacheRoot
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:PIP_CACHE_DIR = Join-Path $cacheRoot 'pip'
$env:JUPYTER_CONFIG_DIR = Join-Path $runtimeRoot 'jupyter'
$env:JUPYTER_DATA_DIR = Join-Path $runtimeRoot 'jupyter-data'
$env:IPYTHONDIR = Join-Path $runtimeRoot 'ipython'

$uvExe = Join-Path $uvRoot 'uv.exe'
if (-not (Test-Path -LiteralPath $uvExe)) {
    $installer = Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1'
    Invoke-Expression $installer
}
if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "uv was not installed at the project-local path: $uvExe"
}

& $uvExe python install 3.11
if ($Force -and (Test-Path -LiteralPath $venvRoot)) {
    $resolved = (Resolve-Path -LiteralPath $venvRoot).Path
    $expectedParent = (Resolve-Path -LiteralPath (Join-Path $projectRoot '.envs')).Path
    if (-not $resolved.StartsWith($expectedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace unexpected environment path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $venvRoot 'Scripts\python.exe'))) {
    & $uvExe venv --python 3.11 --managed-python $venvRoot
}

$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
& $uvExe pip install --python $pythonExe --editable "$projectRoot[dev,notebook]"
& $pythonExe -m ipykernel install --prefix $runtimeRoot --name attention --display-name attention
& $pythonExe -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)"
Write-Host "attention environment ready: $venvRoot"
