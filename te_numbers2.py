"""metrics of all solver variants vs the corrected MC -> results/te_numbers2.json"""
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, json, glob
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
rm=m1['r_over_R']; Tm=(m1['Te_eV']+m2['Te_eV'])/2; dTm=np.abs(m1['Te_eV']-m2['Te_eV'])/2; num=(m1['nu_iz']+m2['nu_iz'])/2; nm=(m1['ne_rel']+m2['ne_rel'])/2; nm=nm/np.interp(0.5,rm,nm)
wm=2*rm; Im=np.sum(wm*nm*num)/np.sum(wm*nm); sel6=rm>0.6
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
def metr(X,T,n,nu):
    n=n/np.interp(0.5,X,n); ni=np.interp(rm,X,n); nui=np.interp(rm,X,nu); Ti=np.interp(rm,X,T)
    I=np.sum(wm*ni*nui)/np.sum(wm*ni); fr=np.sum((wm*ni*nui)[sel6])/np.sum(wm*ni*nui)
    return dict(row=f'{np.interp(0.95,X,T):.2f} / {T.max():.2f} / {np.interp(0.05,X,T):.2f}',n95=float(np.interp(0.95,X,n)),n85=float(np.interp(0.85,X,n)),ratio=float(I/Im),fr=float(fr),
                nu25=float(np.interp(0.25,X,nu)),nu65=float(np.interp(0.65,X,nu)),nu85=float(np.interp(0.85,X,nu)),nu95=float(np.interp(0.95,X,nu)),
                dT=float(np.max(np.abs(Ti-Tm)[rm<0.9])),dTall=float(np.max(np.abs(Ti-Tm))),dn=float(np.max(np.abs(ni-nm))),Tw=float(np.interp(0.95,X,T)),Ta=float(np.interp(0.05,X,T)),Tmax=float(T.max()))
def ld(p):
    d=pickle.load(open(p,'rb')); X=d['X'][::-1]; return metr(X,d['Te'][::-1],d['ne'][::-1],d['nuiz'][::-1])
P={'MC':dict(row=f'{Tm[-1]:.2f} / {Tm.max():.2f} / {Tm[0]:.2f}',n95=float(nm[-1]),n85=float(nm[-2]),ratio=1.0,fr=float(np.sum((wm*nm*num)[sel6])/np.sum(wm*nm*num)),nu25=float(num[2]),nu65=float(num[6]),nu85=float(num[8]),nu95=float(num[9]),Tw=float(Tm[-1]),Ta=float(Tm[0]),Tmax=float(Tm.max()),dTseed=float(dTm.max()),I=float(Im))}
P['II']=metr(rr,np.array(ref['Te_eV']),np.array(ref['ne_rel']),np.array(ref['nu_iz_per_s']))
for k,p in [('III_spec','results/fv_reffields_nu44_m16_L2.pkl'),('III_abs','results/fv_reffields_nu44_m16_L2_abs.pkl'),('IV_o1','results/te_reffields_ne120_m16_L1.pkl'),('IV_o1_abs','results/te_reffields_ne120_m16_L1_abs.pkl'),
            ('IV_o2','results/te_reffields_ne120_m16_L1_o2.pkl'),('IV_o2_L2','results/te_reffields_ne120_m16_L2_o2.pkl'),('IV_o2_m24','results/te_reffields_ne120_m24_L1_o2.pkl'),('IV_o1_m24','results/te_reffields_ne160_m24_L1.pkl')]:
    P[k]=ld(p)
json.dump(P,open('results/te_numbers2.json','w'),indent=1)
for k,v in P.items(): print(k,{a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()})
