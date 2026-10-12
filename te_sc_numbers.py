import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, json, glob, scipy.constants as co
pts=[]
for p in sorted(glob.glob('results/discharge_te_B*_n1e+17_L1_m16_o2.pkl')):
    d=pickle.load(open(p,'rb')); pts.append((d['B_ext'],d['balance'],d))
pts.sort(); S={'scan':[(B,bal) for B,bal,_ in pts]}
S['scan_rows']=[(float(abs(d['E'][2][0])*2),float(d['Te'][0]),float(d['Te'].max()),float(d['Te'][-1]),float(d['phi'].max()-d['phi'].min()),float(d['ne'][-1]/d['ne'][0])) for _,_,d in pts]
Bs=np.array([B for B,_,_ in pts]); bal=np.array([b for _,b,_ in pts]); S['Bstar']=float(np.interp(1.0,bal,Bs))
B,b,d=min(pts,key=lambda t:abs(t[1]-1)); X=d['X']
S.update(Bop=float(B),bal_op=float(b),well=float(d['phi'].max()-d['phi'].min()),compr=float(d['ne'][-1]/d['ne'][0]),Te_wall=float(d['Te'][0]),Te_max=float(d['Te'].max()),Te_axis=float(d['Te'][-1]),
         X_Temax=float(X[np.argmax(d['Te'])]),E_wall=float(abs(d['E'][2][0])*2),IL=float(B*1e-4/co.mu_0),nuiz_max=float(d['nuiz'].max()),X_numax=float(X[np.argmax(d['nuiz'])]),nuiz_axis=float(d['nuiz'][-1]))
json.dump(S,open('results/te_sc_o2.json','w'),indent=1); print(json.dumps(S,indent=1))
