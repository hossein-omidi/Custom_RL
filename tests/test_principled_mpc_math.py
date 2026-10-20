"""Dependency-light tests, plus actual symbolic CasADi NMPC objective/constraints.
Requires only numpy and casadi, not do-mpc or gymnasium.
"""
from pathlib import Path
import ast
import unittest
from types import SimpleNamespace
import numpy as np
import casadi as ca

CODE = Path(__file__).resolve().parents[1] / 'scripts' / 'MPC' / 'mpc_face_milling_chatgpt_V4.py'
S = CODE.read_text(encoding='utf-8')
MODULE = ast.parse(S)
SELECT = {'_phase_hold_samples', '_mpc_speed_floor', '_mpc_depth_ceiling',
          '_apply_mhe_delay_policy', 'PhaseSchedule', 'build_mpc', 'mpc_tvp_fun'}
submodule = ast.Module(body=[n for n in MODULE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in SELECT], type_ignores=[])

class FakeSymbols:
    def __init__(self): self.d={}
    def __getitem__(self, key):
        if key not in self.d:
            self.d[key] = ca.SX.sym('s' + str(len(self.d)))
        return self.d[key]

class FakeMPC:
    def __init__(self, model):
        self.settings=SimpleNamespace()
        self.bounds={};self.scaling={};self.cons=[];self.nlp_cons=[];self.nlp_cons_ub=[];self.nlp_cons_lb=[]
        self.opt_x=FakeSymbols();self.opt_p=FakeSymbols()
    def set_objective(self, lterm, mterm): self.lterm=lterm;self.mterm=mterm
    def set_rterm(self, **kw): self.rterm=kw
    def set_tvp_fun(self, fun): self.tvp_fun=fun
    def set_nl_cons(self, name, expr, **kw):self.cons.append((name, expr, kw))
    def prepare_nlp(self): pass
    def create_nlp(self): pass
    def get_tvp_template(self): return None

ns={'np':np, 'ca':ca,'argparse':__import__('argparse'),
    'do_mpc':SimpleNamespace(controller=SimpleNamespace(MPC=FakeMPC)),
    'omega_to_rpm':lambda x:x,'PhasePlant':object,'__name__':'test_module'}
exec(compile(ast.fix_missing_locations(submodule), str(CODE), 'exec'),ns)

class Tests(unittest.TestCase):
    def test_phase_roundoff_feedback(self):
        p=ns['_phase_hold_samples']; dtheta=np.pi/8; dt=.000094
        for speed in [41.8879, 104.72, 207.35, 418.879]:
            theta=0.; origin=0.
            for k in range(500):
                err=theta-(origin+k*dtheta)
                n=p(speed,dt,dtheta,err)
                theta+=speed*dt*n
                self.assertLessEqual(abs(theta-(origin+(k+1)*dtheta)), speed*dt/2+1e-10)
    def test_phase_feedback_is_not_constant_hold(self):
        p=ns['_phase_hold_samples']; a=p(300,.000094,.392699,0)
        b=p(300,.000094,.392699,.017)
        self.assertNotEqual(a,b)
    def test_phase_roundoff_with_variable_speed(self):
        rng=np.random.default_rng(43)
        p=ns['_phase_hold_samples'];dtheta=np.pi/8;dt=.000094
        theta=0.
        for k in range(2000):
            speed=float(rng.uniform(41.8879,418.879))
            phase_error=theta-k*dtheta
            n=p(speed,dt,dtheta,phase_error)
            theta+=speed*dt*n
            self.assertLessEqual(abs(theta-(k+1)*dtheta), speed*dt/2+1e-9)

    def test_reject_bad_timing(self):
        with self.assertRaises(ValueError): ns['_phase_hold_samples'](0,.0001,.4)
    def test_history_estimate_preserved(self):
        dummy=np.arange(12.,dtype=float).reshape(-1,1)
        pp=SimpleNamespace(p=np.pi/2)
        out,policy=ns['_apply_mhe_delay_policy'](dummy,policy='estimated',mhe_ready=True,
            K=3,M=5,theta_now=3.,pp=pp,theta_hist=[0.,3.],wc_hist=[0.,.5],
            qh_pinv=np.full(5,1e4),phi_now=np.ones(3))
        self.assertEqual(policy,'estimated')
        np.testing.assert_array_equal(out,dummy)
    def test_past_tvp_interpolation(self):
        pp=SimpleNamespace(L1=1.0,L2=1.0,ft=.2,N=4,p=np.pi/2,
            phi_at=lambda x,y:np.array([x,y]),
            tooth_grid=lambda th,dth,nq:(np.array([th]), np.array([dth])))
        schedule=ns['PhaseSchedule'](pp,.4,0,2)
        schedule.set_now(0.,1.,.2)
        schedule.set_now(.38,.9,.2)
        schedule.set_now(.81,.8,.2)
        phi,_,_=schedule.at(-1)
        self.assertAlmostEqual(phi[0],np.interp(.41,[0,.38,.81],[1,.9,.8]))
        self.assertAlmostEqual(phi[1],.2)
    def _build(self):
        pp=SimpleNamespace(nS=2, Phi=np.array([[1.,.5,0],[.5,-.4,.3]]),w_limit=.003,
            omega_min=41.89,omega_max=418.879,ap_min=0.,ap_max=12.,
            reward_w_weight=.6,reward_w_scale=.0015,reward_v_weight=.02,
            reward_v_scale=1.,reward_gate_w=.001,reward_gate_v=1.,reward_gate_v_weight=.05)
        model=SimpleNamespace(x={'eta':ca.SX.sym('eta',3),'v':ca.SX.sym('v',3)},
                              u={'omega':ca.SX.sym('om'),'ap':ca.SX.sym('ap')},
                              tvp={'phi':ca.SX.sym('phi',3)})
        sched=SimpleNamespace(dtheta=np.pi/8)
        a=SimpleNamespace(mpc_horizon=8,coll_deg=2,coll_ni=1,max_iter=200,max_wall_time_s=4,
            nl_cons_collocation_check=False, vibration_weight=1.0,prod_weight=8.,terminal_weight=1.,
            hard_w_margin=.8,mpc_omega_floor_rpm=None,mpc_ap_ceiling_mm=None,minimum_cut_depth=0.,
            omega_rate=20000.,ap_rate=20.)
        mpc=ns['build_mpc'](model,pp,sched,26,a,None,float('inf'))
        return pp,model,mpc
    def test_two_term_cost_matches_analytic(self):
        pp,model,mpc=self._build()
        fun=ca.Function('stage',[model.x['eta'],model.x['v'],model.tvp['phi'],model.u['omega'],model.u['ap']],
                        [mpc.lterm,mpc.mterm])
        eta=np.array([.0004,.0002,-.0001]);v=np.array([.01,-.03,.02]);phi=np.array([1,.4,-.2]);om=300.;ap=3.
        stage,terminal=[float(x) for x in fun(eta,v,phi,om,ap)]
        ws=pp.Phi@eta;wd=pp.Phi@v;wc=phi@eta;vc=phi@v
        w2=(sum(ws**2)+wc**2)/3;v2=(sum(wd**2)+vc**2)/3
        vib=.6*w2/.0015**2+.02*v2
        gate=1/(1+w2/.001**2+.05*v2)
        ltime=vib-8*(om/pp.omega_max)*(ap/pp.ap_max)*gate
        self.assertAlmostEqual(stage,(.5*(pp.omega_min+pp.omega_max)/om)*ltime,places=10)
        self.assertAlmostEqual(terminal,vib,places=10)
    def test_no_move_penalty_no_soft_constraint(self):
        _,_,mpc=self._build()
        self.assertEqual(mpc.rterm, {'omega':0.,'ap':0.})
        self.assertEqual({x[0] for x in mpc.cons}, {'w_hard_up','w_hard_dn','wc_hard_up','wc_hard_dn'})
        self.assertTrue(all(x[2]['soft_constraint'] is False for x in mpc.cons))
        self.assertEqual(len(mpc.nlp_cons),16) # 8 horizon stages times (speed,depth)
    def test_full_action_range(self):
        pp,_,mpc=self._build()
        self.assertEqual(mpc.bounds['lower','_u','omega'],pp.omega_min)
        self.assertEqual(mpc.bounds['upper','_u','omega'],pp.omega_max)
        self.assertEqual(mpc.bounds['lower','_u','ap'],pp.ap_min)
        self.assertEqual(mpc.bounds['upper','_u','ap'],pp.ap_max)

if __name__ == '__main__':unittest.main()
