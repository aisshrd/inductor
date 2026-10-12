import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
P=json.load(open('results/te_sc_o2.json')); d=pickle.load(open(f"results/discharge_te_B{P['Bop']}_n1e+17_L1_m16_o2.pkl",'rb')); X=d['X']
fig,ax=plt.subplots(1,4,figsize=(15,3.6))
ax[0].plot(rr,ref['E_theta_amplitude_V_per_m'],'--',color='C1',label='II: 1.3 G'); ax[0].plot(X,np.abs(d['E'][2])*2,'-',color='C3',label=f"IV: {P['Bop']} G"); ax[0].set(xlabel='r/R',ylabel='|E_theta|, V/m',title='RF field'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(rr,ref['Te_eV'],'--',color='C1'); ax[1].plot(X,d['Te'],'-',color='C3'); ax[1].set(xlabel='r/R',ylabel='T_e, eV',title='electron temperature'); ax[1].grid(alpha=.3)
ax[2].plot(rr,ref['phi_V'],'--',color='C1'); ax[2].plot(X,d['phi']-d['phi'].min(),'-',color='C3'); ax[2].set(xlabel='r/R',ylabel='phi, V',title='ambipolar potential'); ax[2].grid(alpha=.3)
ne=np.array(ref['ne_rel']); ni=np.array(ref['ni_rel'])
ax[3].plot(rr,ne/ne[0],'--',color='C1',label='II n_e'); ax[3].plot(rr,ni/ne[0],':',color='C1',label='II n_i'); ax[3].plot(X,d['ne']/d['ne'][-1],'-',color='C3',label='IV n_e'); ax[3].plot(X,d['ni']/d['ni'][-1],'-.',color='C3',label='IV n_i')
ax[3].set(xlabel='r/R',ylabel='n/n(axis)',title='densities'); ax[3].legend(fontsize=7); ax[3].grid(alpha=.3)
fig.tight_layout(); fig.savefig('figures/figR9_te_sc.png',dpi=130); fig.savefig('figures/figR9_te_sc.png',dpi=130); print('ok')
