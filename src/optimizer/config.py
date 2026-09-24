"""Load config/plan_config.yaml. Every tunable parameter comes from here — never hard-code."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config/plan_config.yaml")


def project_root() -> Path:
    """Repository root (directory containing config/ and data/)."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "config" / "plan_config.yaml").exists():
            return p
    return Path.cwd()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else project_root() / DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(p)
    cfg["_config_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return cfg
