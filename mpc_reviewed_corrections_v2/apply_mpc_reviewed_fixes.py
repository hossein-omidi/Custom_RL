#!/usr/bin/env python3
"""Safely apply reviewed corrections to the expected Custom_RL source version.

Run from the repository root:
    python path/to/apply_mpc_reviewed_fixes.py --dry-run
    python path/to/apply_mpc_reviewed_fixes.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

BASELINE_BLOB_SHA = {
    "custom_rl/plants/plate.py": "7f38dcb3e9f0804a281c6077dfe652e472a3aab2",
    "custom_rl/envs/ode_control_env.py": "94ccb2c7bc82a689d7ac05a751d33a4dade5bc20",
    "custom_rl/eval/pipeline.py": "288fc4df0710e10a605acd77ac5c538cc255ca10",
    "scripts/validate_stochastic_setup.py": "83642f3b1648a13ee5b5aa4c3424555f7ca18ecd",
}

def git_blob_sha(text: str) -> str:
    data = text.encode("utf-8")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()

def find_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if (root / "custom_rl/plants/plate.py").is_file():
            return root
        raise RuntimeError(f"--root does not contain custom_rl/plants/plate.py: {root}")
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "custom_rl/plants/plate.py").is_file():
            return candidate
    raise RuntimeError(
        "Could not find the repository root. Run from the Custom_RL root "
        "or pass --root C:\\path\\to\\Custom_RL."
    )

def replace_once(source: str, old: str, new: str, relpath: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{relpath}: expected exactly one matching source block, found {count}. "
            "No files have been changed. Your checkout may differ from the reviewed revision."
        )
    return source.replace(old, new, 1)

def build_updates(root: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    originals: dict[str, bytes] = {}
    updated: dict[str, bytes] = {}
    normalized_text: dict[str, str] = {}

    # Verify every source file before preparing any write.
    for relpath, expected_sha in BASELINE_BLOB_SHA.items():
        path = root / relpath
        if not path.is_file():
            raise RuntimeError(f"Required source file is missing: {relpath}")
        raw = path.read_bytes()
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{relpath} is not UTF-8; refusing to modify it.") from exc
        normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
        actual_sha = git_blob_sha(normalized)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Source version mismatch for {relpath}\n"
                f"  expected Git blob: {expected_sha}\n"
                f"  found Git blob:    {actual_sha}\n"
                "No files have been changed. This protects local edits and prevents "
                "applying corrections to an unreviewed version."
            )
        originals[relpath] = raw
        normalized_text[relpath] = normalized

    # PlatePlant: one Gaussian modal-acceleration disturbance is sampled at
    # begin_step() and held across every RK4 stage/substep in that env step.
    p = "custom_rl/plants/plate.py"
    s = normalized_text[p]
    s = replace_once(
        s,
        "        self.dynamics_uncertainty_std = float(max(dynamics_uncertainty_std, 0.0))\n",
        "        self.dynamics_uncertainty_std = float(dynamics_uncertainty_std)\n"
        "        if not np.isfinite(self.dynamics_uncertainty_std) or self.dynamics_uncertainty_std < 0.0:\n"
        "            raise ValueError(\"dynamics_uncertainty_std must be finite and non-negative.\")\n",
        p,
    )
    s = replace_once(
        s,
        "        self._rng: np.random.Generator | None = None\n",
        "        self._rng: np.random.Generator | None = None\n"
        "        # One zero-order-held modal-acceleration disturbance per environment step.\n"
        "        self._step_acceleration_disturbance = np.zeros(self.K, dtype=np.float64)\n",
        p,
    )
    begin_old = """    def begin_step(self, t: float, u: np.ndarray) -> None:
        \"\"\"Prepare continuous path/phase kinematics before RK integration.

        ODEControlEnv calls this once per accepted environment step. The action
        is still held constant during the RK substeps, but the cutter position
        and tooth phase are propagated from the last accepted values instead of
        recomputing them as omega*t from the start of the episode.
        \"\"\"
        u_phys = self._scale_action(u)
"""
    begin_new = """    def begin_step(self, t: float, u: np.ndarray) -> None:
        \"\"\"Prepare continuous path/phase kinematics and sample this env-step disturbance.

        ODEControlEnv calls this once per accepted environment step. The action
        is held constant during RK substeps, and accepted cutter kinematics are
        propagated from the previous step. The Gaussian modal-acceleration
        offset is also sampled once here and held across all RK4 stages/substeps.
        \"\"\"
        u_phys = self._scale_action(u)

        if self.dynamics_uncertainty_std > 0.0 and self._rng is not None:
            self._step_acceleration_disturbance = (
                self.dynamics_uncertainty_std * self._rng.standard_normal(self.K)
            )
        else:
            self._step_acceleration_disturbance.fill(0.0)
"""
    s = replace_once(s, begin_old, begin_new, p)
    noise_old = """        if self.dynamics_uncertainty_std > 0.0 and self._rng is not None:
            x_dot = x_dot.copy()
            x_dot[1::2] += (
                self.dynamics_uncertainty_std
                * self._rng.standard_normal(self.K)
            )
"""
    noise_new = """        if self.dynamics_uncertainty_std > 0.0:
            x_dot = x_dot.copy()
            x_dot[1::2] += self._step_acceleration_disturbance
"""
    s = replace_once(s, noise_old, noise_new, p)
    s = replace_once(
        s,
        "        # 2. Reset force module history and internal kinematic states\n"
        "        f_nonlinear2.reset_state_history()\n",
        "        # 2. Reset force module history and internal kinematic states\n"
        "        self._step_acceleration_disturbance.fill(0.0)\n"
        "        f_nonlinear2.reset_state_history()\n",
        p,
    )
    updated[p] = s.encode("utf-8")

    # ODEControlEnv: validate noise standard deviations and keep known path
    # coordinates out of sensor measurement noise.
    p = "custom_rl/envs/ode_control_env.py"
    s = normalized_text[p]
    s = replace_once(
        s,
        "        obs_noise_std:\n"
        "            Standard deviation of additive observation noise.\n",
        "        obs_noise_std:\n"
        "            Standard deviation of noise on normalized sensor channels; "
        "cutter-path coordinates remain noise-free.\n",
        p,
    )
    s = replace_once(
        s,
        "        self.auto_history_module = bool(auto_history_module)\n\n"
        "        if self.dt <= 0.0:\n",
        "        self.auto_history_module = bool(auto_history_module)\n\n"
        "        if not np.isfinite(self.process_noise_std) or self.process_noise_std < 0.0:\n"
        "            raise ValueError(\"process_noise_std must be finite and non-negative.\")\n"
        "        if not np.isfinite(self.obs_noise_std) or self.obs_noise_std < 0.0:\n"
        "            raise ValueError(\"obs_noise_std must be finite and non-negative.\")\n\n"
        "        if self.dt <= 0.0:\n",
        p,
    )
    obs_old = """    def _add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        \"\"\"Add observation noise if requested.\"\"\"
        obs = np.asarray(obs, dtype=np.float64)
        if self.obs_noise_std > 0.0:
            return obs + self.obs_noise_std * self.np_random.standard_normal(obs.shape)
        return obs
"""
    obs_new = """    def _add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        \"\"\"Noise measured sensor channels; leave known cutter-path coordinates exact.\"\"\"
        obs = np.asarray(obs, dtype=np.float64)
        if self.obs_noise_std <= 0.0:
            return obs

        if hasattr(self.plant, \"n_sensors\"):
            # Plate observations are [all sensor displacements, all sensor
            # velocities, normalized cutter x, normalized cutter y].
            sensor_channels = 2 * int(self.plant.n_sensors)
        else:
            # Generic ODE plants without an explicit sensor count retain the
            # previous behavior: treat their complete observation as measured.
            sensor_channels = obs.size
        sensor_channels = min(max(sensor_channels, 0), obs.size)

        noisy_obs = obs.copy()
        if sensor_channels:
            noisy_obs[:sensor_channels] += (
                self.obs_noise_std
                * self.np_random.standard_normal(sensor_channels)
            )
        return noisy_obs
"""
    s = replace_once(s, obs_old, obs_new, p)
    updated[p] = s.encode("utf-8")

    # Shared train/evaluation kwargs: reject invalid std instead of silently
    # clamping a negative value to zero.
    p = "custom_rl/eval/pipeline.py"
    s = normalized_text[p]
    s = replace_once(
        s,
        "    _translate_legacy_depth_kwargs(extra)\n\n"
        "    kwargs: dict[str, Any] = {\n",
        "    _translate_legacy_depth_kwargs(extra)\n\n"
        "    dynamics_uncertainty_std = float(dynamics_uncertainty_std)\n"
        "    if not np.isfinite(dynamics_uncertainty_std) or dynamics_uncertainty_std < 0.0:\n"
        "        raise ValueError(\"dynamics_uncertainty_std must be finite and non-negative.\")\n\n"
        "    kwargs: dict[str, Any] = {\n",
        p,
    )
    s = replace_once(
        s,
        '"dynamics_uncertainty_std": float(max(dynamics_uncertainty_std, 0.0)),\n',
        '"dynamics_uncertainty_std": dynamics_uncertainty_std,\n',
        p,
    )
    updated[p] = s.encode("utf-8")

    # Regression tests for the corrected stochastic and observation semantics.
    p = "scripts/validate_stochastic_setup.py"
    s = normalized_text[p]
    tests_to_insert = r"""

def check_disturbance_is_sampled_once_per_control_step() -> None:
    # One Gaussian acceleration offset must span every RK4 stage/substep.
    register_envs()
    env = gym.make(
        ENV_ID,
        reward_id="sparse",
        randomize_y0=False,
        dynamics_uncertainty_std=0.1,
        dt=1.0e-4,
        n_substeps=3,
        max_episode_steps=10,
    )

    class CountingRNG:
        def __init__(self, generator):
            self.generator = generator
            self.draws = 0

        def standard_normal(self, size=None):
            self.draws += 1
            return self.generator.standard_normal(size)

    env.reset(seed=13)
    plant = env.unwrapped.plant
    counter = CountingRNG(env.unwrapped.np_random)
    # Preserve the counting wrapper when ODEControlEnv re-binds the Gym RNG.
    plant.bind_rng = lambda _rng: setattr(plant, "_rng", counter)

    action = np.full(env.action_space.shape, -1.0, dtype=np.float64)
    samples = []
    for expected_draws in (1, 2):
        _, _, terminated, truncated, _ = env.step(action)
        assert counter.draws == expected_draws, (
            f"expected one disturbance draw per env step, got {counter.draws}"
        )
        sample = plant._step_acceleration_disturbance.copy()
        assert np.all(np.isfinite(sample))
        assert np.any(sample != 0.0)
        samples.append(sample)
        if terminated or truncated:
            break

    assert len(samples) == 2, "the test needs two non-terminal environment steps"
    assert not np.array_equal(samples[0], samples[1])
    env.close()
    print("[ok] one sample-and-hold acceleration disturbance per environment step")


def check_observation_noise_excludes_path_coordinates() -> None:
    register_envs()
    clean = gym.make(
        ENV_ID, reward_id="sparse", randomize_y0=False,
        dynamics_uncertainty_std=0.0, obs_noise_std=0.0, max_episode_steps=5,
    )
    noisy = gym.make(
        ENV_ID, reward_id="sparse", randomize_y0=False,
        dynamics_uncertainty_std=0.0, obs_noise_std=0.2, max_episode_steps=5,
    )
    try:
        obs_clean, _ = clean.reset(seed=23)
        obs_noisy, _ = noisy.reset(seed=23)
        sensor_channels = 2 * noisy.unwrapped.plant.n_sensors
        assert 0 < sensor_channels < obs_noisy.size
        assert np.allclose(obs_clean[sensor_channels:], obs_noisy[sensor_channels:])
        assert not np.allclose(obs_clean[:sensor_channels], obs_noisy[:sensor_channels])
    finally:
        clean.close()
        noisy.close()
    print("[ok] observation noise affects sensor channels, not cutter-path coordinates")


def check_noise_parameter_validation() -> None:
    register_envs()
    invalid_kwargs = (
        {"dynamics_uncertainty_std": np.nan},
        {"process_noise_std": -0.1},
        {"obs_noise_std": np.inf},
    )
    for kwargs in invalid_kwargs:
        try:
            gym.make(
                ENV_ID, reward_id="sparse", randomize_y0=False,
                max_episode_steps=5, **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid noise setting was accepted: {kwargs}")
    print("[ok] non-finite/negative noise standard deviations are rejected")
"""
    s = replace_once(
        s,
        "\n\ndef check_reward_terms() -> None:\n",
        tests_to_insert + "\n\ndef check_reward_terms() -> None:\n",
        p,
    )
    s = replace_once(
        s,
        '    assert kwargs["dynamics_uncertainty_std"] == 0.0\n'
        '    assert resolve_ppo_model_path(Path("models/ppo_plate"), 99) is None\n',
        '    assert kwargs["dynamics_uncertainty_std"] == 0.0\n'
        '    for bad_std in (-1.0, np.nan, np.inf):\n'
        '        try:\n'
        '            plate_env_kwargs(dynamics_uncertainty_std=bad_std)\n'
        '        except ValueError:\n'
        '            pass\n'
        '        else:\n'
        '            raise AssertionError(f"invalid dynamics uncertainty accepted: {bad_std}")\n'
        '    assert resolve_ppo_model_path(Path("models/ppo_plate"), 99) is None\n',
        p,
    )
    s = replace_once(
        s,
        '    check_stochastic_differs_with_uncertainty()\n'
        '    check_reward_terms()\n',
        '    check_stochastic_differs_with_uncertainty()\n'
        '    check_disturbance_is_sampled_once_per_control_step()\n'
        '    check_observation_noise_excludes_path_coordinates()\n'
        '    check_noise_parameter_validation()\n'
        '    check_reward_terms()\n',
        p,
    )
    updated[p] = s.encode("utf-8")

    return originals, updated

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Repository root (optional).")
    parser.add_argument("--dry-run", action="store_true", help="Verify source versions and patch anchors without writing.")
    args = parser.parse_args()

    try:
        root = find_repo_root(args.root)
        originals, updated = build_updates(root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("PASS: exact reviewed source versions and all patch anchors matched.")
        print("Would update:")
        for relpath in updated:
            print(f"  {relpath}")
        print("No files were changed.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / f".mpc_reviewed_backup_{stamp}"
    try:
        for relpath, raw in originals.items():
            backup = backup_root / relpath
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(raw)

        written: list[str] = []
        try:
            for relpath, normalized_bytes in updated.items():
                source_raw = originals[relpath]
                newline = "\r\n" if b"\r\n" in source_raw else "\n"
                normalized_text = normalized_bytes.decode("utf-8")
                final_bytes = normalized_text.replace("\n", newline).encode("utf-8")
                (root / relpath).write_bytes(final_bytes)
                written.append(relpath)
        except Exception:
            for relpath in written:
                (root / relpath).write_bytes(originals[relpath])
            raise
    except Exception as exc:
        print(f"ERROR while writing files: {exc}", file=sys.stderr)
        print(f"Inspect backups at: {backup_root}")
        return 3

    print("Applied reviewed corrections.")
    print(f"Byte-exact backups: {backup_root}")
    for relpath in updated:
        print(f"  updated {relpath}")
    print("\nNext:")
    print("  python -m compileall custom_rl scripts/validate_stochastic_setup.py")
    print("  python scripts/validate_stochastic_setup.py")
    print("  git diff --check")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
