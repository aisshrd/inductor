"""Sanity tests for the total-energy solver ekin_cyl_te.py"""
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, scipy.constants as co, time, sys
from ekin import load_argon, EnergyGridKinetics
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1
m=int(sys.argv[1]) if len(sys.argv)>1 else 12; n_e=int(sys.argv[2]) if len(sys.argv)>2 else 100
kTg=co.k*300/co.e
def run(phi,E0,L=1,absorb=False,umax=50.):
    kin=CylKineticsTE(xs,R,ng,omega,phi,L=L,Nh=1,n_e=n_e,umax=umax,m=m,wall_absorb=absorb)
    E=np.zeros((3,m),complex); E[2]=E0/2; E[0]=E0/2; B=np.zeros((3,m),complex)
    t=time.time(); f=kin.solve(E,B); dt=time.time()-t
    return kin,f,dt
# ---- test 1: zero field, flat potential -> 3/2 kTg everywhere
kin,f,dt=run(np.zeros(m),0.0,umax=0.5)
print(f'T1 zero field ({dt:.1f}s): <u>/(1.5kTg) =',np.round(kin.mean_energy(f)/(1.5*kTg),3))
print('   density uniformity (min/max):',np.round(kin.density(f).min()/kin.density(f).max(),4))
# ---- test 2: uniform RF 200 V/m, flat potential, specular wall -> compare with 0D energy-grid solver
E0=200.
kin,f,dt=run(np.zeros(m),E0)
k0=EnergyGridKinetics(xs,umax=50.,nu=200,L=1,Nh=1,ng=ng); f0=k0.solve(E0,omega); u0=k0.mean_energy(f0)
print(f'T2 uniform RF {E0} V/m ({dt:.1f}s): <u> cyl-TE =',np.round(kin.mean_energy(f),3),' 0D =',round(u0,3))
print('   nu_iz cyl-TE',np.array2string(kin.nu_iz(f),precision=3),' 0D',f'{k0.rate_iz(f0):.3e}')
# ---- test 3: uniform RF + parabolic well 9 V: f0 should be a function of eps_tot only (nonlocal regime)
phi=9.0*(1-kin.X**2)
kin,f,dt=run(phi,E0)
F0=np.real(f[1,0]); n=kin.density(f)
print(f'T3 well 9 V ({dt:.1f}s): Te(r)=',np.round(kin.mean_energy(f)*2/3,2))
print('   n/n(axis)=',np.round(n/n[-1],3),'  Boltzmann exp(-phi/Te_axis)=',np.round(np.exp(-(phi[-1]-phi)/ (kin.mean_energy(f)[-1]*2/3)),3))
# r-dependence of f0 at fixed eps_tot: compare columns normalised at one common value
for i in np.where(kin.allowed.all(axis=1))[0][::8]:
    col=F0[i]; print(f'   eps_tot={kin.ec[i]:6.2f} eV: f0(r)/f0(axis) =',np.round(col/col[-1],3))
import pickle; pickle.dump(dict(X=kin.X,ec=kin.ec,F0=F0,phi=phi,Te=kin.mean_energy(f)*2/3,n=n),open('results/te_test3.pkl','wb'))
