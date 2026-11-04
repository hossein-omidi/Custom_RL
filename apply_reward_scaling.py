#!/usr/bin/env python
"""Safely add running-only reward scaling to the exact mpc-correction environment."""
from __future__ import annotations
import argparse, shutil, sys
from datetime import datetime
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one anchor for {label}; found {count}. No files changed.")
    return text.replace(old, new, 1)


def patch_reward(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "running_reward_scale" in text:
        raise RuntimeError(f"{path} already contains running_reward_scale; refusing duplicate patch.")
    text = replace_once(
        text,
        "        productivity_weight: float = 30.0,\n        negative_ap_weight: float = 2.0,",
        "        productivity_weight: float = 30.0,\n        running_reward_scale: float = 1.0,\n        negative_ap_weight: float = 2.0,",
        "productive reward constructor parameter",
    )
    text = replace_once(
        text,
        "        self.productivity_weight = float(productivity_weight)\n        self.negative_ap_weight = float(negative_ap_weight)",
        "        self.productivity_weight = float(productivity_weight)\n        self.running_reward_scale = float(running_reward_scale)\n        if not np.isfinite(self.running_reward_scale) or self.running_reward_scale < 0.0:\n            raise ValueError(\"running_reward_scale must be finite and non-negative.\")\n        self.negative_ap_weight = float(negative_ap_weight)",
        "productive reward scale initialization",
    )
    old_reward = """        reward = (
            self.alive_bonus
            + productivity_term
            - w_cost
            - wdot_cost
            - omega_cost
            - self.negative_ap_weight * negative_ap_cost
            - ap_action_cost
            - self.action_weight * action_cost
        )"""
    new_reward = """        running_reward_unscaled = (
            self.alive_bonus
            + productivity_term
            - w_cost
            - wdot_cost
            - omega_cost
            - self.negative_ap_weight * negative_ap_cost
            - ap_action_cost
            - self.action_weight * action_cost
        )
        # Scale only dense running terms. Terminal completion/failure adjustments
        # remain in their configured physical reward units and are not scaled.
        reward = self.running_reward_scale * running_reward_unscaled"""
    text = replace_once(text, old_reward, new_reward, "running reward composition")
    text = replace_once(
        text,
        '            "productivity": float(productivity_term),\n            "omega_cost": float(omega_cost),',
        '            "productivity": float(productivity_term),\n'
        '            "running_reward_unscaled": float(running_reward_unscaled),\n'
        '            "running_reward_scale": float(self.running_reward_scale),\n'
        '            "running_reward_scaled": float(self.running_reward_scale * running_reward_unscaled),\n'
        '            "productivity_scaled": float(self.running_reward_scale * productivity_term),\n'
        '            "vibration_w_cost_scaled": float(self.running_reward_scale * w_cost),\n'
        '            "vibration_wdot_cost_scaled": float(self.running_reward_scale * wdot_cost),\n'
        '            "omega_cost_scaled": float(self.running_reward_scale * omega_cost),\n'
        '            "omega_cost": float(omega_cost),',
        "running reward diagnostics",
    )
    return text


def patch_registration(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if '"running_reward_scale"' in text:
        raise RuntimeError(f"{path} already accepts running_reward_scale; refusing duplicate patch.")
    text = replace_once(
        text,
        '        "productivity_weight",\n        "negative_ap_weight",',
        '        "productivity_weight",\n        "running_reward_scale",\n        "negative_ap_weight",',
        "registration reward kwargs allowlist",
    )
    stale_comment = """# Production-first reward balance (numerically verified).
#
# Design: material removal is the FIRST-CLASS objective.  There is NO bonus for
# merely reaching the pass end; instead, failing to reach it (chatter or timeout)
# is penalized.  So the return is  (sum of dense production) - (failure penalty),
# and the optimum is the deepest/most-productive operating point that stays
# stable -- not "idle until the pass completes".
#
#   * pass_completion_bonus = 0   -> reaching the end earns nothing (no idling
#                                    incentive); production must be earned per step.
#   * termination/truncation_penalty = 500 -> penalty for NOT reaching the end.
#   * productivity_weight = 240   -> the normalized MRR proxy divides by the
#                                    ACTION bound ap_max (=12), so s_ap = ap/12 is
#                                    throttled to <=1/8 over the feasible depth
#                                    (<=~1.5 mm).  Scaling P by the same factor
#                                    (30 -> 240 = 30 * 12/1.5) restores a full-range
#                                    production gradient WITHOUT restricting the
#                                    action space.  IMPORTANT: P is coupled to
#                                    ap_max here; if ap_max changes, rescale P so
#                                    P/ap_max stays ~= 20 per mm.
#   * omega_cost_weight = 0.3     -> mild actuator/speed cost, kept well below the
#                                    productive MRR so the high-rpm (better-lobe)
#                                    regime is not avoided and a fast productive
#                                    pass never goes net-negative.
"""
    current_comment = """# Runtime reward defaults for the current roughing study.
# The unified PPO trainer records these coefficients explicitly in its run
# configuration. Its running_reward_scale scales only dense per-step terms;
# pass-completion and failure/truncation adjustments remain unscaled.
# Current runtime values: productivity=240, omega_cost=3, completion_bonus=1000,
# termination/truncation_penalty=500, displacement_weight=0.6,
# velocity_weight=0.02. Keep this comment aligned with the constants below.
"""
    text = replace_once(text, stale_comment, current_comment, "stale reward documentation")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true", help="Installer already created a full backup.")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    reward_path = root / "custom_rl/rewards/plate_rewards.py"
    registration_path = root / "custom_rl/envs/registration.py"
    if not reward_path.is_file() or not registration_path.is_file():
        raise SystemExit("Expected custom_rl/rewards/plate_rewards.py and custom_rl/envs/registration.py under repo root.")
    try:
        reward_new = patch_reward(reward_path)
        registration_new = patch_registration(registration_path)
    except Exception as exc:
        raise SystemExit(f"Preflight failed; no files changed: {exc}")
    if args.dry_run:
        print("Reward-scale patch preflight: PASS (2 exact source anchors found). No files changed.")
        return 0
    backup = None
    if not args.no_backup:
        backup = root / f".ppo_single_stage_backup_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(reward_path, backup / "plate_rewards.py")
        shutil.copy2(registration_path, backup / "registration.py")
    reward_path.write_text(reward_new, encoding="utf-8", newline="\n")
    registration_path.write_text(registration_new, encoding="utf-8", newline="\n")
    print(f"Reward-scale patch applied. Backups: {backup if backup else 'provided by installer'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
