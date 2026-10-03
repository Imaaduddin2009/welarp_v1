$ErrorActionPreference = "Stop"
python .\run_all.py
if ($LASTEXITCODE -ne 0) {
    throw "The full C360 run failed with exit code $LASTEXITCODE"
}
