import importlib
import subprocess
import sys

REQUIRED = [
    "numpy",
    "pandas",
    "yaml",      # pyyaml
    "anndata",
    "scanpy",
    "pyarrow",
]

PIP_NAMES = {
    "yaml": "pyyaml",
}

missing = []
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(PIP_NAMES.get(pkg, pkg))

if missing:
    print("Installing missing Python packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("All required Python packages are already installed.")
