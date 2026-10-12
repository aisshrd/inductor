"""Total-energy solver in the reference fields (E_theta, B_z, phi of the self-consistent point 13.3 Pa, n0=1e17, B_ext=1.3 G)."""
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, json, pickle, sys, time
from ekin import load_argon
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json'))
n_e=int(sys.argv[1]); m=int(sys.argv[2]); L=int(sys.argv[3]); absorb=len(sys.argv)>4 and sys.argv[4]=='absorb'; umax=float(sys.argv[5]) if len(sys.argv)>5 else 45.; order=int(sys.argv[6]) if len(sys.argv)>6 else 2
rr=np.array(ref['r_over_R'])
# radial grid identical to CylKineticsTE (build a throw-away object with phi=0 to get X)
tmp=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=1,n_e=10,umax=5.,m=m); X=tmp.X
def interp(key,ang=None):
    a=np.array(ref[key])
    if ang is None: return np.interp(X,rr,a)
    return np.interp(X,rr,a)*np.exp(1j*np.interp(X,rr,np.array(ref[ang])))
E1=interp('E_theta_amplitude_V_per_m','E_theta_phase_rad')/2; B1=interp('B_z_amplitude_T','B_z_phase_rad')/2; phi=interp('phi_V'); phi=phi-phi.min()
E=np.zeros((3,m),complex); E[2]=E1; E[0]=np.conj(E1); B=np.zeros((3,m),complex); B[2]=B1; B[0]=np.conj(B1)
t=time.time(); kin=CylKineticsTE(xs,R,ng,omega,phi,L=L,Nh=1,n_e=n_e,umax=umax,m=m,cluster=2.0,wall_absorb=absorb,order=order); tb=time.time()-t
f=kin.solve(E,B); ne=kin.density(f); Te=kin.mean_energy(f)*2/3; nuiz=kin.nu_iz(f)
print(f'TE n_e={n_e} m={m} L={L} absorb={absorb} umax={umax} order={order}: build {tb:.1f}s assemble {kin.t_assemble:.1f}s solve {kin.t_solve:.1f}s  unknowns {kin.nTot}  lam={kin.lam.real:.3e} 1/s')
print('  X   ',np.round(kin.X,2)); print('  phi ',np.round(phi,2)); print('  Te  ',np.round(Te,2)); print('  ne/ne(0.5)',np.round(ne/np.interp(0.5,kin.X[::-1],ne[::-1]),2)); print('  nuiz',np.array2string(nuiz,precision=3))
print('  <nu_iz> (density-weighted):',np.sum(kin.wdisk*ne*nuiz)/np.sum(kin.wdisk*ne))
# f0 vs eps_tot at several radii (normalised per column to the density) for the nonlocality check
F0=np.real(f[1,0]); pickle.dump(dict(X=kin.X,Te=Te,ne=ne,nuiz=nuiz,absorb=absorb,ec=kin.ec,F0=F0,phi=phi,ukin=kin.ukin,w=kin.w,lam=kin.lam,
    t_build=tb,t_assemble=kin.t_assemble,t_solve=kin.t_solve,nTot=kin.nTot),open(f'results/te_reffields_ne{n_e}_m{m}_L{L}{"_abs" if absorb else ""}{"_o2" if order==2 else ""}.pkl','wb'))
