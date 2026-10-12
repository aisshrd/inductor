"""Cylindrical electron Monte Carlo with real argon cross sections in PRESCRIBED self-consistent fields
(E_theta harmonic, B_z harmonic, static phi) from the kinetic solver. Specular wall. Null-collision method,
synchronous time stepping (dt fixed) with analytic-free Boris-like push in 3D (x,y,z), z ignorable.
Records radial profiles of <eps>, nu_iz, density, and EEDF in three radial bins."""
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, scipy.constants as co, pickle, sys, time, json
from ekin import load_argon, GAMMA
e,me=co.e,co.m_e
xs=load_argon('data/Ar_LXCat.txt')
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json'))
R=0.1; omega=2*np.pi*13.56e6; ng=3.21e21
import os
if os.environ.get('MCTEST'):
    rr=np.linspace(0.05,1,20)*R; Eamp=float(os.environ.get('MCE','80'))*np.ones(20); Eph=np.zeros(20); Bamp=np.zeros(20); Bph=np.zeros(20); phi=float(os.environ['MCTEST'])*(1-(rr/R)**2)
else:
    rr=np.array(ref['r_over_R'])*R; Eamp=np.array(ref['E_theta_amplitude_V_per_m']); Eph=np.array(ref['E_theta_phase_rad']); Bamp=np.array(ref['B_z_amplitude_T']); Bph=np.array(ref.get('B_z_phase_rad',[0]*len(Bamp))); phi=np.array(ref['phi_V'])*(0.0 if os.environ.get('NOPHI') else 1.0)
# extend to axis and wall
rg=np.concatenate([[0],rr]); Eg=np.concatenate([[0],Eamp]); Pg=np.concatenate([[Eph[0]],Eph]); PBg=np.concatenate([[Bph[0]],Bph]); Bg=np.concatenate([[Bamp[0]],Bamp]); phg=np.concatenate([[phi[0]],phi])
# radial field as the exact derivative of the piecewise-linear phi(r) on a fine grid (the earlier np.gradient at the 8 coarse nodes,
# linearly interpolated, integrated to a well 1.7 V (18 %) deeper than phi itself)
rg_f=np.linspace(0,R,2001); ph_f=np.interp(rg_f,rg,phg); Er_f=-np.gradient(ph_f,rg_f)
# cross sections on energy grid
u=np.concatenate([[0],np.logspace(-3,2.3,1500)]); sig=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1])
s_el=sig(xs['elastic'],u); inel=[(th,sig(d,u)*(u>th)) for th,d,_ in xs['inel']]; thi,di=xs['iz']; s_iz=sig(di,u)*(u>thi)
s_tot=s_el+sum(s for _,s in inel)+s_iz; mM=xs['mM']
nu_max=np.max((ng*s_tot*GAMMA*np.sqrt(u))[u<=60])*1.1
cum=[s_el]+[s for _,s in inel]+[s_iz]; ths=[None]+[th for th,_ in inel]+[thi]
rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 1)
N=int(sys.argv[1]) if len(sys.argv)>1 else 2000
dt=0.5/nu_max; T=2*np.pi/omega
t_total=float(sys.argv[3]) if len(sys.argv)>3 else 30e-6; t_burn=0.5*t_total
# init: uniform in disk, 3 eV Maxwellian
r0=R*np.sqrt(rng.random(N)); a0=2*np.pi*rng.random(N); x=np.stack([r0*np.cos(a0),r0*np.sin(a0),np.zeros(N)],1)
v=rng.normal(0,np.sqrt(e*3.0/me),size=(N,3))
nb=10; edges=np.linspace(0,R,nb+1); acc=np.zeros((nb,4)); hed=np.linspace(0,50,101); hist=np.zeros((3,100)); nsamp=0
def fields(x,t):
    r=np.hypot(x[:,0],x[:,1]); rh=np.stack([x[:,0],x[:,1],0*r],1)/np.maximum(r,1e-12)[:,None]; th=np.stack([-rh[:,1],rh[:,0],0*r],1)
    Et=np.interp(r,rg,Eg)*np.cos(omega*t+np.interp(r,rg,Pg)); Bz=(0.0 if os.environ.get('NOB') else 1.0)*np.interp(r,rg,Bg)*np.cos(omega*t+np.interp(r,rg,PBg))
    Er=np.interp(r,rg_f,Er_f)
    E=Er[:,None]*rh+Et[:,None]*th; B=np.zeros_like(E); B[:,2]=Bz
    return E,B,r
def boris(v,E,B,dt):
    qm=-e/me; vm=v+0.5*dt*qm*E; tt=0.5*dt*qm*B; s=2*tt/(1+np.sum(tt*tt,1))[:,None]
    vp=vm+np.cross(vm+np.cross(vm,tt),s); return vp+0.5*dt*qm*E
t=0.0; t0=time.time(); nsteps=int(t_total/dt); report=nsteps//10
for k in range(nsteps):
    E,B,r=fields(x,t); v=boris(v,E,B,dt); x=x+v*dt; t+=dt
    r=np.hypot(x[:,0],x[:,1]); out=r>R
    if out.any():
        rh=x[out,:2]/r[out][:,None]; vr=np.sum(v[out,:2]*rh,1); v[out,:2]-=2*vr[:,None]*rh; x[out,:2]=rh*(2*R-r[out])[:,None]
    # collisions
    # null-collision method with a fixed time step: the number of candidate collisions per particle and step is Poisson(nu_max*dt)
    # (the earlier version used a single candidate with probability 1-exp(-nu_max*dt), which under-counts all collisions by the factor
    #  (1-exp(-x))/x = 0.79 at x=0.5, i.e. an effective gas density 21 % too low)
    ncol=rng.poisson(nu_max*dt,N)
    for rnd in range(int(ncol.max()) if ncol.size else 0):
      uu=0.5*me*np.sum(v*v,1)/e; vel=np.sqrt(np.sum(v*v,1))
      cand=np.where(ncol>rnd)[0]
      if cand.size:
        rr_=rng.random(cand.size)*nu_max; accn=np.zeros(cand.size); done=np.zeros(cand.size,bool)
        for kk,(s,th) in enumerate(zip(cum,ths)):
            accn=accn+ng*np.interp(uu[cand],u,s)*vel[cand]; m=(~done)&(rr_<accn)
            if m.any():
                sel=cand[m]; n=sel.size
                if kk==0: cosx=1-2*rng.random(n); vnew=vel[sel]*np.sqrt(np.maximum(1-2*mM*(1-cosx),0))
                else: vnew=np.sqrt(2*e*np.maximum(uu[sel]-th,0)*(0.5 if th==thi else 1.0)/me)
                cth=1-2*rng.random(n); sth=np.sqrt(1-cth**2); ph=2*np.pi*rng.random(n)
                v[sel,0]=vnew*sth*np.cos(ph); v[sel,1]=vnew*sth*np.sin(ph); v[sel,2]=vnew*cth
            done|=m
    if t>t_burn and k%20==0:
        uu=0.5*me*np.sum(v*v,1)/e; vel=np.sqrt(np.sum(v*v,1)); r=np.hypot(x[:,0],x[:,1]); idx=np.minimum((r/R*nb).astype(int),nb-1)
        nuiz=ng*np.interp(uu,u,s_iz)*vel
        acc[:,0]+=np.bincount(idx,minlength=nb); acc[:,1]+=np.bincount(idx,weights=uu,minlength=nb); acc[:,2]+=np.bincount(idx,weights=nuiz,minlength=nb)
        for b,(lo,hi) in enumerate([(0.9,1.0),(0.4,0.6),(0.0,0.2)]):
            m=(r/R>=lo)&(r/R<hi); hist[b]+=np.histogram(uu[m],bins=hed)[0]
        nsamp+=1
    if k%report==0: print(f'  {100*k/nsteps:.0f}% t={t*1e6:.1f}us  <eps>={np.mean(0.5*me*np.sum(v*v,1)/e):.2f} eV  ({time.time()-t0:.0f}s)',flush=True)
rc=0.5*(edges[1:]+edges[:-1]); dens=acc[:,0]/(np.pi*(edges[1:]**2-edges[:-1]**2)); dens/=dens[-1]
res=dict(r_over_R=rc/R,ne_rel=dens,Te_eV=(acc[:,1]/np.maximum(acc[:,0],1))*2/3,nu_iz=acc[:,2]/np.maximum(acc[:,0],1),eedf=(0.5*(hed[1:]+hed[:-1]),hist/nsamp),N=N,t_total=t_total,time=time.time()-t0)
pickle.dump(res,open(f'results/mc_cyl_real{"_nophi" if os.environ.get("NOPHI") else ""}_N{N}_s{sys.argv[2] if len(sys.argv)>2 else 1}.pkl','wb'))
print('done',time.time()-t0); print('r/R',np.round(rc/R,2)); print('Te',np.round(res['Te_eV'],2)); print('nuiz',np.array2string(res['nu_iz'],precision=2)); print('ne',np.round(dens,2))
