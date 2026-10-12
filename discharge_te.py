"""Self-consistent discharge with the TOTAL-ENERGY electron solver (ekin_cyl_te.CylKineticsTE):
Picard loop  electrons(E,B,phi) -> sigma -> Maxwell(E,B) -> ambipolar ions (balance) -> phi (quasineutrality), at fixed n0.
The kinetic object is rebuilt each iteration because the eps_tot grid depends on phi (0.5 s)."""
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, scipy.constants as co, json, pickle, sys, time
from ekin import load_argon
from ekin_cyl import Discharge
from ekin_cyl_fv import CylKineticsFV
from ekin_cyl_te import CylKineticsTE
e=co.e

class DischargeTE(Discharge):
    def __init__(self,xs,R,ng,omega,B_ext,n0,L=1,n_e=120,umax=45.,m=16,cluster=2.0,wall_absorb=False,order=2,**kw):
        helper=CylKineticsFV(xs,R,ng,omega,L=1,Nh=1,nu=8,umax=10.,m=m,cluster=cluster)     # radial grid + FD matrices for Maxwell / phi'
        super().__init__(helper,B_ext,n0,**kw)
        self.xs,self.R,self.ngas,self.omega=xs,R,ng,omega; self.L,self.n_e,self.umax,self.m,self.cluster,self.wall_absorb,self.order=L,n_e,umax,m,cluster,wall_absorb,order
        self.kin=None
    def electrons(self,E,B,phi):
        self.kin=CylKineticsTE(self.xs,self.R,self.ngas,self.omega,phi-phi.min(),L=self.L,Nh=1,n_e=self.n_e,umax=self.umax,m=self.m,cluster=self.cluster,wall_absorb=self.wall_absorb,order=self.order)
        assert np.allclose(self.kin.X,self.k.X)
        if self.order==2 and getattr(self,'prev_f',None) is not None:        # limiter frozen from the previous Picard iterate: one LU per iteration
            self.kin._theta=self.kin.limiter_theta(self.prev_f); f=self.kin._solve(E,B); self.kin._theta=None
        else: f=self.kin.solve(E,B,limiter_iters=(2 if self.order==2 else 0))
        self.prev_f=f; return f
    def moments(self,f):
        k=self.kin; return k.density(f),k.mean_energy(f)*2/3,k.nu_iz(f)
    def current_theta(self,f,n=1):
        k=self.kin; vth=k.ang.unit_vectors()[0][1]; c=np.einsum('aij,ij,ij->a',k.ang.Y,vth,k.ang.W)
        return -e*np.einsum('a,ik,aik->k',c,k.w*k.v,f[k.Nh+n])
    def run(self,iters=12,relax=0.6,verbose=True,phi_relax=0.5,phi_max=40.,E0=None,B0=None,phi0=None):
        k=self.k; m=k.m; H,Nh=k.H,k.Nh
        E=np.zeros((H,m),complex); B=np.zeros((H,m),complex)
        for n in (-1,1): B[n+Nh]=self.B_ext[n+Nh]; E[n+Nh]=-1j*n*k.omega*k.r*self.B_ext[n+Nh]/2
        if E0 is not None: E=E0.copy(); B=B0.copy()
        phi=np.zeros(m) if phi0 is None else phi0.copy(); self.Te_wall=3.0; hist=[]; bal=np.nan
        for it in range(iters):
            t=time.time(); f=self.electrons(E,B,phi)
            ne,Te,nuiz=self.moments(f); ne=np.maximum(ne,1e-6); Te=np.clip(Te,0.03,30.); nuiz=np.maximum(nuiz,0)
            j1=self.current_theta(f,1)*self.n0
            sigma=j1/np.where(np.abs(E[Nh+1])>1e-12,E[Nh+1],1e-12)
            sigma=np.where(np.abs(E[Nh+1])>1e-3*np.max(np.abs(E[Nh+1])),sigma,sigma[0])
            Enew,Bnew=self.maxwell(sigma)
            self.Te_wall=max(Te[0],0.5)
            ni,lam=self.ambipolar(ne,Te,nuiz); ni=np.maximum(ni,1e-4); ni=ni/(k.wdisk@ni)
            dphi=np.clip(Te*np.log(ni/ne),-2.0,2.0); dphi-=dphi[0]
            phi=np.clip(phi+phi_relax*dphi,-phi_max,phi_max)
            bal=1.0/lam
            dE=float(np.max(np.abs(Enew-E))/max(np.max(np.abs(Enew)),1e-30))
            E=(1-relax)*E+relax*Enew; B=(1-relax)*B+relax*Bnew
            hist.append((it,dE,bal,Te[0],Te[-1],float(np.max(np.abs(dphi)))))
            if verbose: print(f'  it{it} ({time.time()-t:.0f}s): dE={dE:.3f} dphi={np.max(np.abs(dphi)):.2f} Te wall/max/axis {Te[0]:.2f}/{Te.max():.2f}/{Te[-1]:.2f} eV, nuiz wall/axis {nuiz[0]:.2e}/{nuiz[-1]:.2e}, |E|wall {abs(E[Nh+1][0])*2:.0f} V/m, ne axis/wall {ne[-1]/ne[0]:.2f}, ni axis/wall {ni[-1]/ni[0]:.2f}, phi axis {phi[-1]:.2f} V, balance {bal:.3f}, lam_cons {self.kin.lam.real:.1e}',flush=True)
            if it>3 and dE<2e-3 and np.max(np.abs(dphi))<0.05: break
        self.f,self.E,self.B,self.phi,self.ni,self.ne,self.Te,self.nuiz,self.hist,self.balance=f,E,B,phi,ni,ne,Te,nuiz,hist,bal
        return self

if __name__=='__main__':
    xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1
    Bg=float(sys.argv[1]) if len(sys.argv)>1 else 1.3; n0=float(sys.argv[2]) if len(sys.argv)>2 else 1e17; L=int(sys.argv[3]) if len(sys.argv)>3 else 1; m=int(sys.argv[4]) if len(sys.argv)>4 else 16; n_e=int(sys.argv[5]) if len(sys.argv)>5 else 120
    Bext=np.zeros(3,complex); Bext[0]=Bext[2]=Bg*1e-4/2
    d=DischargeTE(xs,R,ng,omega,Bext,n0,L=L,n_e=n_e,m=m)
    # start from the reference (kinetic-energy) self-consistent fields to shorten the Picard path
    ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R']); X=d.k.X
    def interp(key,ang=None):
        a=np.array(ref[key]);
        return np.interp(X,rr,a) if ang is None else np.interp(X,rr,a)*np.exp(1j*np.interp(X,rr,np.array(ref[ang])))
    E1=interp('E_theta_amplitude_V_per_m','E_theta_phase_rad')/2*Bg/1.3; B1=interp('B_z_amplitude_T','B_z_phase_rad')/2*Bg/1.3; phi=interp('phi_V')
    E0=np.zeros((3,m),complex); E0[2]=E1; E0[0]=np.conj(E1); B0=np.zeros((3,m),complex); B0[2]=B1; B0[0]=np.conj(B1)
    t=time.time(); d.run(iters=14,relax=0.6,phi_relax=0.5,E0=E0,B0=B0,phi0=phi); print('total',time.time()-t,'s')
    pickle.dump(dict(X=d.k.X,E=d.E,B=d.B,phi=d.phi,ne=d.ne,ni=d.ni,Te=d.Te,nuiz=d.nuiz,hist=d.hist,balance=d.balance,B_ext=Bg,n0=n0,L=L,m=m,n_e=n_e,
                     F0=np.real(d.f[1,0]),ec=d.kin.ec,ukin=d.kin.ukin,w=d.kin.w),open(f'results/discharge_te_B{Bg}_n{n0:.0e}_L{L}_m{m}{"_o2" if d.order==2 else ""}.pkl','wb'))
