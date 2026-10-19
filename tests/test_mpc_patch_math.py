"""Dependency-light checks for the MPC-only patch, run with python -m unittest.

These tests do NOT claim to replace a full do-mpc closed-loop regression. They
check the actuator-timing projection and delay discretization without Gym.
"""
from __future__ import annotations
import ast
import sys
import types
import unittest
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'scripts' / 'MPC' / 'mpc_face_milling_chatgpt_V3.py'
NAMES = {'cheb_lobatto_D', '_project_realized_rate_limits', 'delay_channel_error'}
module = ast.parse(SRC.read_text(encoding='utf8'))
ns = {'np': np, 'PhasePlant': type('PhasePlant', (), {})}
exec(compile(ast.Module(body=[x for x in module.body if isinstance(x, ast.FunctionDef) and x.name in NAMES], type_ignores=[]), str(SRC), 'exec'), ns)

class TestMPCPatch(unittest.TestCase):
    def test_chebyshev_exact_on_quadratics(self):
        for M in (4, 8, 16, 28):
            xi, D = ns['cheb_lobatto_D'](M)
            np.testing.assert_allclose(D @ xi, np.ones(M+1), atol=2e-12)
            np.testing.assert_allclose(D @ (xi**2), 2*xi, atol=2e-12)

    def test_transport_delay_matches_exact_phase_shift(self):
        # First three participation-selected modes, 400 to 4000 rpm.
        pp = types.SimpleNamespace(p=np.pi/2, omega_min=400*np.pi/30,
                                   omega_max=4000*np.pi/30,
                                   wn=np.array([107.4, 258.8, 654.6]))
        e24 = ns['delay_channel_error'](pp, 24, n_om=20)
        e26 = ns['delay_channel_error'](pp, 26, n_om=20)
        self.assertLess(e26, 1e-6)
        self.assertLess(e26, e24)

    def test_slew_uses_previous_realized_hold_and_bounds(self):
        pp = types.SimpleNamespace(omega_min=40., ap_min=0.0, ap_max=18.0, env_dt=0.001)
        previous = np.array([200., 2.0])
        cmd = np.array([500., 6.0])
        dt_prev = 0.008
        u, n, dt = ns['_project_realized_rate_limits'](cmd, previous, pp,
               dtheta=0.4, omega_hi=600., omega_rate=1000., ap_rate=40.,
               ap_fixed=None, previous_dt=dt_prev)
        np.testing.assert_allclose(u, [208., 2.32], atol=1e-12)
        self.assertLessEqual(abs(u[0]-previous[0])/dt_prev,1000.+1e-10)
        self.assertLessEqual(abs(u[1]-previous[1])/dt_prev,40.+1e-10)
        self.assertEqual(n, max(1, round(0.4/(u[0]*pp.env_dt))))
        self.assertAlmostEqual(dt, n*pp.env_dt)

    def test_fixed_depth_and_startup(self):
        pp = types.SimpleNamespace(omega_min=40., ap_min=0., ap_max=10., env_dt=0.001)
        prev = np.array([200.,0.5])
        u, _, _ = ns['_project_realized_rate_limits'](np.array([50.,7.]), prev,
            pp, 0.4, 500., 1000., 5., 0.5)
        np.testing.assert_allclose(u,[198.,0.5])

    def test_no_extra_mode_or_plant_edits(self):
        tree = ast.parse(SRC.read_text())
        f = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'build_model')
        self.assertIn('set_rhs("qh"',ast.get_source_segment(SRC.read_text(),f))
        self.assertIn('collocation', SRC.read_text())

if __name__=='__main__': unittest.main()
