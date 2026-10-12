import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, json, sys
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
rm=m1['r_over_R']; Tm=(m1['Te_eV']+m2['Te_eV'])/2; dT=np.abs(m1['Te_eV']-m2['Te_eV'])/2; num=(m1['nu_iz']+m2['nu_iz'])/2; nm=(m1['ne_rel']+m2['ne_rel'])/2
nm=nm/np.interp(0.5,rm,nm)
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
def ld(p): 
    d=pickle.load(open(p,'rb')); X=d['X'][::-1]; return X,d['Te'][::-1],(d['ne']/np.interp(0.5,X,d['ne'][::-1]))[::-1],d['nuiz'][::-1],d
runs={'Cheb m=8 L=2 (spec)':(rr,np.array(ref['Te_eV']),np.array(ref['ne_rel'])/np.interp(0.5,rr,ref['ne_rel']),np.array(ref['nu_iz_per_s']),None)}
for name,p in [('FV spec L=2','results/fv_reffields_nu44_m16_L2.pkl'),('FV abs L=2','results/fv_reffields_nu44_m16_L2_abs.pkl')]+[(a,b) for a,b in zip(sys.argv[1::2],sys.argv[2::2])]:
    try: runs[name]=ld(p)
    except Exception as ex: print('skip',name,ex)
print('r/R   MC Te      '+'  '.join(f'{k[:14]:>14s}' for k in runs))
for i,r in enumerate(rm):
    print(f'{r:.2f}  {Tm[i]:.2f}+-{dT[i]:.2f}  '+'  '.join(f'{np.interp(r,X,T):14.2f}' for X,T,n,nu,d in runs.values()))
print('density n/n(0.5R):'); 
for i,r in enumerate(rm):
    print(f'{r:.2f}  {nm[i]:.2f}       '+'  '.join(f'{np.interp(r,X,n):14.2f}' for X,T,n,nu,d in runs.values()))
print('nu_iz:')
for i,r in enumerate(rm):
    print(f'{r:.2f}  {num[i]:9.3g}  '+'  '.join(f'{np.interp(r,X,nu):14.3g}' for X,T,n,nu,d in runs.values()))
# integral metrics: <nu_iz> density weighted with MC bins (equal-width rings)
wm=2*rm; Im=np.sum(wm*nm*num)/np.sum(wm*nm)
print('<nu_iz>_n  MC %.4g'%Im,'  '.join(f'{k}: {np.sum(wm*np.interp(rm,X,n)*np.interp(rm,X,nu))/np.sum(wm*np.interp(rm,X,n)):.4g} (ratio {np.sum(wm*np.interp(rm,X,n)*np.interp(rm,X,nu))/np.sum(wm*np.interp(rm,X,n))/Im:.2f})' for k,(X,T,n,nu,d) in runs.items()))
sel=rm>0.6; print('fraction of ionization at r/R>0.6: MC %.2f'%(np.sum((wm*nm*num)[sel])/np.sum(wm*nm*num)),'  '.join(f'{k}: {np.sum((wm*np.interp(rm,X,n)*np.interp(rm,X,nu))[sel])/np.sum(wm*np.interp(rm,X,n)*np.interp(rm,X,nu)):.2f}' for k,(X,T,n,nu,d) in runs.items()))
sel=rm<0.9; print('max|dTe| r/R<0.9:','  '.join(f'{k}: {np.max(np.abs(np.interp(rm,X,T)-Tm)[sel]):.2f}' for k,(X,T,n,nu,d) in runs.items()))
