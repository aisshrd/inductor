import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, scipy.sparse as sp
from ekin import load_argon
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1; m=6; n_e=60
kin=CylKineticsTE(xs,R,ng,omega,9.0*(1-np.linspace(0.95,0.05,m)**2),L=1,Nh=1,n_e=n_e,umax=40.,m=m,wall_absorb=False)
E=np.zeros((3,m),complex); E[2]=100; E[0]=100; B=np.zeros((3,m),complex)
A=kin.kinetic_matrix(E,B).tocsc(); nTot=A.shape[0]; wv=np.zeros(nTot); base=kin.idx(0,0,0,0); ne=n_e
wv[base:base+ne*m]=(kin.w*kin.wdisk[None,:]).ravel()
r=np.asarray(A.T@wv).ravel(); H,nlm=kin.H,kin.nlm
for n in range(H):
  for a in range(nlm):
    blk=r[(n*nlm+a)*ne*m:(n*nlm+a+1)*ne*m].reshape(ne,m); blk=np.where(kin.allowed,blk,0)
    sc=np.abs(A[(n*nlm+a)*ne*m:(n*nlm+a+1)*ne*m,:]).max()
    print(n-1,kin.ang.lm[a],'max|res|',np.abs(blk).max(),'rel to max|A| in block',np.abs(blk).max()/sc, 'at i,k',np.unravel_index(np.abs(blk).argmax(),blk.shape))
for k in range(m):
    cs=np.asarray(kin.C0[k].T@kin.w[:,k]).ravel()/np.maximum(kin.w[:,k],1e-30)
    bad=np.where(np.abs(cs)>1)[0]; print(k,'phi',kin.phi[k],'bad cols',bad, cs[bad][:5], 'ukin top',kin.ukin[-1,k])
