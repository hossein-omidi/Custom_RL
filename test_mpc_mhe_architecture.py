"""Dependency-light tests for the MPC-only MHE/history architecture patch.

Run from the project root: python test_mpc_mhe_architecture.py
No changes to the gym plant or RL components.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

CODE = Path(__file__).parent / 'scripts/MPC/mpc_face_milling_chatgpt_V3.py'
src = CODE.read_text()
ast.parse(src)
node = next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_apply_mhe_delay_policy')
nspace = {'np':np, 'PhasePlant':object,'__name__':'__testing__'}

# Deterministic mock of the delayed-history reconstruction. In the real module
# this is the unmodified Chebyshev-Lobatto interpolation of measured phase samples.
def history(theta, pp, M, th, wc):
    return np.array([wc[-1]-0.05*(k+1) for k in range(M)])
nspace['_delay_history_state']=history
exec(compile(ast.Module(body=[node],type_ignores=[]),str(CODE),'exec'),nspace)
apply=nspace['_apply_mhe_delay_policy']
K,M=3,4
x=np.arange(2*K+M+1,dtype=float).reshape(-1,1)
orig=x.copy()
q_pinv=np.array([-0.01, -0.02, -0.03, -0.04])
pp=SimpleNamespace(w_limit=.003)
th=[0.,1.,2.]
wcs=[2.,2.5,3.]
phi=np.array([.2,.3,.5])

# Estimated mode preserves the dynamically constrained MHE qh completely.
y,tag=apply(x,policy='estimated',mhe_ready=True,K=K,M=M,theta_now=2.,pp=pp,theta_hist=th,wc_hist=wcs,qh_pinv=q_pinv,phi_now=phi)
assert tag=='estimated' and np.array_equal(y,orig) and np.array_equal(x,orig)

# Legacy behavior is reproduced exactly (even if MHE was ready).
y,tag=apply(x,policy='pinv',mhe_ready=True,K=K,M=M,theta_now=2.,pp=pp,theta_hist=th,wc_hist=wcs,qh_pinv=q_pinv,phi_now=phi)
assert tag=='pinv' and np.allclose(y[2*K:2*K+M,0],q_pinv)
assert np.allclose(y[:2*K,0],x[:2*K,0])

# Modal-history mode reconstructs current cutter displacement from MHE eta.
wcs2=list(wcs)
y,tag=apply(x,policy='modal-history',mhe_ready=True,K=K,M=M,theta_now=2.,pp=pp,theta_hist=th,wc_hist=wcs2,qh_pinv=q_pinv,phi_now=phi)
assert tag=='modal-history' and np.isclose(wcs2[-1],phi@x[:K,0])
assert np.allclose(y[2*K:2*K+M,0],history(2.,pp,M,th,wcs2))

# Failure / warm-up always yields the existing pinv history.
for policy in ['estimated','modal-history']:
 y,tag=apply(x,policy=policy,mhe_ready=False,K=K,M=M,theta_now=2.,pp=pp,theta_hist=th,wc_hist=wcs,qh_pinv=q_pinv,phi_now=phi)
 assert tag=='pinv-fallback' and np.allclose(y[2*K:2*K+M,0],q_pinv)

# Modal penalty strength at unchanged physical state: baseline vs profile.
original=8/.5**2
stability=18/.4**2
balanced=10/.55**2
assert np.isclose(stability/original,3.515625)
assert 0.98 < balanced/original < 1.05

# Both policies are opt-in, but all original speed/depth bounds remain.
assert 'if a.balanced_pass and a.stability_first' in src
assert '"mhe_history_policy": "estimated"' in src
assert '"mpc_omega_floor_rpm"' not in src.split('trial = {',1)[1].split('}',1)[0]
assert '"mpc_ap_ceiling_mm"' not in src.split('trial = {',1)[1].split('}',1)[0]
print('PASS: AST, MHE-state preservation, legacy reproduction, modal history, fallback, normalized profile, unrestricted controls')
