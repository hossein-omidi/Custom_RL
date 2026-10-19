"""Dependency-light checks for the MPC-only diagnostic extensions."""
import ast
import argparse
from pathlib import Path
import numpy as np
from types import SimpleNamespace

FILE=Path(__file__).parent/'scripts/MPC/mpc_face_milling_chatgpt_V3.py'
source=FILE.read_text()
root=ast.parse(source)
names={'_mpc_speed_floor','_mpc_depth_ceiling','_interval_safety_ratio','_project_realized_rate_limits'}
functions=[n for n in root.body if isinstance(n,ast.FunctionDef) and n.name in names]
# Postpone evaluation of toolchain type hints; import only mathematical helpers.
for f in functions:
 f.decorator_list=[]
 f.returns=None
 for a in (*f.args.args,*f.args.kwonlyargs): a.annotation=None
ns={'np':np,'argparse':argparse}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(FILE),'exec'),ns)
pp=SimpleNamespace(omega_min=2*np.pi*400/60,omega_max=2*np.pi*4000/60,ap_min=0.,ap_max=12.,env_dt=0.0001)
a=argparse.Namespace(mpc_omega_floor_rpm=2200.0,mpc_ap_ceiling_mm=3.0)
assert abs(ns['_mpc_speed_floor'](pp,a)-2*np.pi*2200/60)<1e-12
assert ns['_mpc_depth_ceiling'](pp,a)==3.
assert ns['_interval_safety_ratio'](np.array([0.,0.]),np.zeros(2),0.001,1.,0.002,0.003)==2/3
assert ns['_interval_safety_ratio'](np.array([0.001,0]),np.array([0.3,0]),0.001,1.,0.,0.003)==(0.0013/0.003)
pr=np.array([250.,2.])
u,nh,dtr=ns['_project_realized_rate_limits'](np.array([200.,6.]),pr,pp,0.392699,418.879,3000.,20.,None,previous_dt=0.003,omega_floor=ns['_mpc_speed_floor'](pp,a),ap_ceiling=3.)
assert u[0]>=ns['_mpc_speed_floor'](pp,a)
assert u[0]>=pr[0]-9-1e-12
assert u[1]<=2.06+1e-12
assert u[1]<=3.+1e-12
assert dtr==nh*pp.env_dt
print('PASS: bounds, previous-hold rate constraints, interval peak risk, syntax')
