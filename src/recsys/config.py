from pathlib import Path
from typing import Any

import yaml

# This file is src/recsys/config.py, so parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    """Load the YAML config and turn every entry under `paths` into an absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open() as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (REPO_ROOT / v).resolve() for k, v in cfg.get("paths", {}).items()}
    return cfg