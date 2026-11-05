"""Shared helpers for organized PPO run directories.

Canonical layout (created by train_ppo_unified / experiment.py):

    runs/ppo/<mode>_<timestamp>/
      run_manifest.json
      config.yaml          # frozen experiment settings (optional)
      models/
      logs/seed_<n>/...
      evaluation/          # eval_policy.py trajectories
      plots/               # plot_results.py
      mpc/                 # NMPC trajectories (mpc_face_milling_chatgpt_V4)
      comparison/          # compare_mpc_rl.py outputs

A pointer file ``runs/ppo/LATEST`` stores the absolute path of the most recent
completed (or started) run so ``--run latest`` needs no manual paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("runs/ppo")
LATEST_FILENAME = "LATEST"
MANIFEST_FILENAME = "run_manifest.json"
SUBDIRS = ("models", "logs", "evaluation", "plots", "mpc", "comparison")


def default_output_root() -> Path:
    return DEFAULT_OUTPUT_ROOT.resolve()


def latest_pointer_path(output_root: Path | str | None = None) -> Path:
    root = Path(output_root).expanduser().resolve() if output_root else default_output_root()
    return root / LATEST_FILENAME


def write_latest_pointer(run_dir: Path | str, output_root: Path | str | None = None) -> Path:
    """Record ``run_dir`` as the latest run under ``output_root``."""
    run_dir = Path(run_dir).expanduser().resolve()
    root = Path(output_root).expanduser().resolve() if output_root else run_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / LATEST_FILENAME
    pointer.write_text(str(run_dir) + "\n", encoding="utf-8")
    return pointer


def read_latest_pointer(output_root: Path | str | None = None) -> Path | None:
    pointer = latest_pointer_path(output_root)
    if not pointer.is_file():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        root = Path(output_root).expanduser().resolve() if output_root else default_output_root()
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path if path.is_dir() else None


def list_runs(output_root: Path | str | None = None) -> list[Path]:
    """Return run directories that contain a manifest, newest first."""
    root = Path(output_root).expanduser().resolve() if output_root else default_output_root()
    if not root.is_dir():
        return []
    runs = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / MANIFEST_FILENAME).is_file()
    ]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def resolve_run_dir(
    run: str | None = None,
    run_dir: Path | str | None = None,
    output_root: Path | str | None = None,
) -> Path:
    """Resolve a run directory from ``--run-dir`` or ``--run`` (including ``latest``)."""
    if run_dir is not None and run is not None:
        raise ValueError("Pass only one of run_dir or run, not both.")

    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {path}")
        return path

    root = Path(output_root).expanduser().resolve() if output_root else default_output_root()
    token = (run or "latest").strip()
    if token.lower() == "latest":
        pointed = read_latest_pointer(root)
        if pointed is not None:
            return pointed
        runs = list_runs(root)
        if not runs:
            raise FileNotFoundError(
                f"No runs found under {root}. Train first or pass an explicit --run-dir."
            )
        return runs[0]

    candidate = Path(token).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    under_root = (root / token).resolve()
    if under_root.is_dir():
        return under_root
    raise FileNotFoundError(
        f"Cannot resolve run {token!r}. Expected a directory or a name under {root}."
    )


def run_subdir(run_dir: Path | str, name: str) -> Path:
    if name not in SUBDIRS:
        raise ValueError(f"Unknown run subdirectory {name!r}; expected one of {SUBDIRS}.")
    path = Path(run_dir).expanduser().resolve() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(run_dir: Path | str) -> Path:
    return run_subdir(run_dir, "models")


def logs_dir(run_dir: Path | str) -> Path:
    return run_subdir(run_dir, "logs")


def evaluation_dir(run_dir: Path | str) -> Path:
    return run_subdir(run_dir, "evaluation")


def plots_dir(run_dir: Path | str) -> Path:
    return run_subdir(run_dir, "plots")


def mpc_dir(run_dir: Path | str) -> Path:
    return run_subdir(run_dir, "mpc")


def comparison_dir(run_dir: Path | str) -> Path:
    return run_subdir(run_dir, "comparison")


def load_json(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_run_manifest(run_dir: Path | str) -> dict[str, Any]:
    path = Path(run_dir).expanduser().resolve() / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {MANIFEST_FILENAME} in {run_dir}")
    return load_json(path)


def load_training_config(run_dir: Path | str, seed: int | None = None) -> dict[str, Any] | None:
    """Load a seed training_config.json; defaults to the first available seed."""
    log_root = Path(run_dir).expanduser().resolve() / "logs"
    if seed is not None:
        path = log_root / f"seed_{seed}" / "training_config.json"
        return load_json(path) if path.is_file() else None
    if not log_root.is_dir():
        return None
    candidates = sorted(log_root.glob("seed_*/training_config.json"))
    if not candidates:
        return None
    return load_json(candidates[0])


def ensure_run_layout(run_dir: Path | str) -> Path:
    """Create the standard subdirectories under an existing run directory."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRS:
        (run_dir / name).mkdir(exist_ok=True)
    return run_dir
