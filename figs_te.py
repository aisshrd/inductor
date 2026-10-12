import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
rm=m1['r_over_R']; Tm=(m1['Te_eV']+m2['Te_eV'])/2; num=(m1['nu_iz']+m2['nu_iz'])/2; nm=(m1['ne_rel']+m2['ne_rel'])/2
fa=pickle.load(open('results/fv_reffields_nu44_m16_L2_abs.pkl','rb')); fs=pickle.load(open('results/fv_reffields_nu44_m16_L2.pkl','rb'))
te=pickle.load(open(sys.argv[1],'rb')); te2=pickle.load(open(sys.argv[2],'rb')) if len(sys.argv)>2 else None
lab2=sys.argv[3] if len(sys.argv)>3 else 'TE L=2, m=24'
def n05(X,n): return n/np.interp(0.5,X[::-1],n[::-1])
fig,ax=plt.subplots(1,3,figsize=(13.5,3.9))
ax[0].plot(rm,Tm,'ko',label='Monte Carlo (specular wall)'); ax[0].plot(rr,ref['Te_eV'],'-',color='C7',label='II: Chebyshev m=8 (specular)')
ax[0].plot(fs['X'],fs['Te'],'--',color='C1',label='III: FV S_N, kinetic energy (specular)'); ax[0].plot(fa['X'],fa['Te'],':',color='C1',lw=2,label='III: FV S_N, kinetic energy (absorbing)')
ax[0].plot(te['X'],te['Te'],'-',color='C3',lw=2,label='IV: total energy, L=1, m=16 (specular)')
if te2 is not None: ax[0].plot(te2['X'],te2['Te'],'-.',color='C2',lw=2,label='IV: total energy, '+lab2)
ax[0].set(xlabel='r/R',ylabel='T_e, eV',title='reference fields, 13.3 Pa, n0=1e17 m^-3',ylim=(1.8,4.3)); ax[0].legend(fontsize=6.5); ax[0].grid(alpha=.3)
ax[1].plot(rm,nm/np.interp(0.5,rm,nm),'ko'); ax[1].plot(rr,np.array(ref['ne_rel'])/np.interp(0.5,rr,ref['ne_rel']),'-',color='C7')
ax[1].plot(fs['X'],n05(fs['X'],fs['ne']),'--',color='C1'); ax[1].plot(fa['X'],n05(fa['X'],fa['ne']),':',color='C1',lw=2); ax[1].plot(te['X'],n05(te['X'],te['ne']),'-',color='C3',lw=2)
if te2 is not None: ax[1].plot(te2['X'],n05(te2['X'],te2['ne']),'-.',color='C2',lw=2)
ax[1].set(xlabel='r/R',ylabel='n_e/n_e(0.5R)',title='density'); ax[1].grid(alpha=.3)
ok=num>0; ax[2].semilogy(rm[ok],num[ok],'ko'); nr=np.array(ref['nu_iz_per_s']); ok=nr>0; ax[2].semilogy(rr[ok],nr[ok],'-',color='C7')
ax[2].semilogy(fs['X'],np.maximum(fs['nuiz'],1),'--',color='C1'); ax[2].semilogy(fa['X'],np.maximum(fa['nuiz'],1),':',color='C1',lw=2); ax[2].semilogy(te['X'],np.maximum(te['nuiz'],1),'-',color='C3',lw=2)
if te2 is not None: ax[2].semilogy(te2['X'],np.maximum(te2['nuiz'],1),'-.',color='C2',lw=2)
ax[2].set(xlabel='r/R',ylabel='nu_iz, 1/s',title='ionization frequency',ylim=(10,1e6)); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig('figures/figR7_te.png',dpi=130); fig.savefig('figures/figR7_te.png',dpi=130)
# ---- EEPF comparison at three radii + f0(eps_tot) nonlocality
fig,ax=plt.subplots(1,2,figsize=(11,3.9))
uc,h=m1['eedf']; h=(h+m2['eedf'][1])/2; du=0.5; X=te['X']; F0=te['F0']; uk=te['ukin']
for b,lab,sel,c in [(0,'wall 0.9-1','C0',X>0.9),(1,'mid 0.4-0.6','C1',(X>0.4)&(X<0.6)),(2,'axis 0-0.2','C2',X<0.2)]:
    fm=h[b]/np.sum(h[b])/du/np.sqrt(uc); ax[0].semilogy(uc,np.maximum(fm,1e-9),'o',ms=3,color=sel,label='MC '+lab)
    ks=np.where(c)[0]; out=np.zeros_like(uc)
    for k in ks:
        ok=uk[:,k]>0; out+=np.interp(uc,uk[ok,k],F0[ok,k],left=0,right=0)
    out/=len(ks); out/=np.sum(out*np.sqrt(uc)*du); ax[0].semilogy(uc,np.maximum(out,1e-9),'-',color=sel,label='total-energy solver '+lab)
ax[0].set(xlabel='kinetic energy, eV',ylabel='EEPF (normalised)',xlim=(0,30),ylim=(1e-7,1),title='EEPF at three radii'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ec=te['ec']
for k,c in [(len(X)-1,'C2'),(len(X)//2,'C1'),(3,'C0'),(0,'k')]:
    ok=uk[:,k]>0; ax[1].semilogy(ec[ok],np.maximum(F0[ok,k],1e-12),'-',color=c,label=f'r/R={X[k]:.2f}, phi={te["phi"][k]:.1f} V')
ax[1].set(xlabel='total energy eps - e phi(r), eV',ylabel='f0',xlim=(-10,30),ylim=(1e-8,1),title='f0 vs total energy: r-independent where nonlocal'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig('figures/figR8_te_eepf.png',dpi=130); fig.savefig('figures/figR8_te_eepf.png',dpi=130)
print('figures written')
