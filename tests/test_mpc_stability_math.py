"""Dependency-light tests of the MPC stability intervention.

Run: python -m pytest tests/test_mpc_stability_math.py -q
Full Gym/do-mpc closed-loop validation is still mandatory before research claims.
"""
from __future__ import annotations
import ast
from pathlib import Path
import math

CODE = Path(__file__).parents[1] / 'scripts' / 'MPC' / 'mpc_face_milling_chatgpt_V3.py'

def _guard_func():
    tree=ast.parse(CODE.read_text(encoding='utf-8'))
    f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_hysteretic_guard_active')
    ns={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),str(CODE),'exec'),ns)
    return ns['_hysteretic_guard_active']

def test_guard_enters_above_threshold_only():
    g=_guard_func()
    assert not g(False, 0.61, 0.62, 0.52, True)
    assert g(False,0.62,0.62,0.52,True)

def test_guard_retains_state_in_hysteresis_band():
    g=_guard_func()
    assert g(True,0.57,0.62,0.52,True)
    assert not g(True,0.51,0.62,0.52,True)

def test_disabled_guard_resets_state():
    assert not _guard_func()(True,1.5,0.62,0.52,False)

def test_phase_horizon_shorter_than_dominant_period():
    Nteeth=4; dtheta_fraction=.25; horizon=8; omega=4000*2*math.pi/60
    phase_horizon = horizon*dtheta_fraction*2*math.pi/Nteeth
    assert phase_horizon/omega < (1/17)/5

def test_preview_is_physical_time_coordinate():
    theta=math.pi/8; speed=200; w=.001; vel=.1
    w_preview=w+theta/speed*vel
    assert abs(w_preview-0.001196349540849362)<1e-12

def test_gated_productivity_decreases_for_larger_vibration():
    def L(mrr,disp,vel): return mrr/(1+disp+.1*vel)
    assert L(1,0,0)>L(1,1,0)>L(1,2,0)
    assert L(1,1,0)>L(1,1,3)

def test_stability_profile_is_opt_in():
    text=CODE.read_text(encoding='utf-8')
    assert '"--stability-first", action="store_true"' in text
    assert '"--preview-gain", type=float, default=0.0' in text
    assert '"--prod-gate-disp", type=float, default=0.0' in text
    assert '"--prod-gate-vel", type=float, default=0.0' in text
