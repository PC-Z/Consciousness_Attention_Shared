$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimeRoot = Join-Path $projectRoot '.runtime'
$env:JUPYTER_CONFIG_DIR = Join-Path $runtimeRoot 'jupyter'
$env:JUPYTER_DATA_DIR = Join-Path $runtimeRoot 'jupyter-data'
$env:JUPYTER_PATH = Join-Path $runtimeRoot 'share\jupyter'
$env:IPYTHONDIR = Join-Path $runtimeRoot 'ipython'
$env:TEMP = Join-Path $runtimeRoot 'temp'
$env:TMP = $env:TEMP
$pythonExe = Join-Path $projectRoot '.envs\attention\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Run .\setup_env.ps1 before starting Jupyter.'
}
& $pythonExe -m jupyterlab --notebook-dir $projectRoot @args
