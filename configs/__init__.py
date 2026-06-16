"""Load experiment configurations by name (conf1, conf2, conf3, ...)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

CONFIGS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIGS_DIR.parent


def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("conf*.py"))


def load_config(name: str) -> dict[str, Any]:
    """
    Load a configuration module by stem name, e.g. 'conf1'.

    Each config module must define a CONFIG dict.
    """
    module_name = name if name.startswith("conf") else f"conf{name}"
    try:
        module = importlib.import_module(f"configs.{module_name}")
    except ModuleNotFoundError as exc:
        available = ", ".join(list_configs()) or "(none)"
        raise ValueError(
            f"Unknown config '{name}'. Available: {available}"
        ) from exc

    if not hasattr(module, "CONFIG"):
        raise ValueError(f"configs.{module_name} must define CONFIG dict.")

    cfg = dict(module.CONFIG)
    cfg.setdefault("config_name", module_name)

    run_dir = Path(cfg.get("run_dir", f"runs/{module_name}"))
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    cfg["run_dir"] = str(run_dir)
    cfg["log_dir"] = str(run_dir / "logs")
    cfg["model_dir"] = str(run_dir / "models")
    cfg["traj_dir"] = str(run_dir / "trajectories")
    cfg["mc_dir"] = str(run_dir / "eval_mc")
    cfg["plot_dir"] = str(run_dir / "plots")
    cfg["report_dir"] = str(run_dir / "reports")
    cfg["stability_dir"] = str(run_dir / "stability_lobes")

    return cfg


def save_config_snapshot(cfg: dict[str, Any], path: Path) -> None:
    """Persist resolved config for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
