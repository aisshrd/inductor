"""ekin_cyl_te.py -- TOTAL-ENERGY formulation on the finite-volume S_N radial scheme.
Unknown f_{nlm}(eps_tot_i, r_k) on a COMMON grid in total energy eps_tot = eps_kin - e*phi(r) ... we use the convention
   eps_tot = eps_kin + U(r),  U(r) = e*(phi_wall - phi(r)) >= 0   (potential energy of the electron, zero at the wall where phi is max... )
Here phi(r) from the discharge model is >=0 with phi(R)=0 (well: phi larger on axis -> electrons attracted to axis).
Electron potential energy W(r) = -e*phi(r) <= 0 (lowest on axis). Total energy eps_tot = eps_kin + W(r) = eps_kin - e phi(r).
Grid in eps_tot from eps_min = -e*phi_max (axis) to umax. In cell (i,k): eps_kin = eps_tot_i + e*phi_k ; cells with eps_kin<0 are forbidden (f=0).
The electrostatic force term vanishes identically. All energy-dependent operators are evaluated per (i,k) at eps_kin.
Collision operators that shift energy (inelastic) act along eps_kin at fixed r -> in total-energy grid they shift by the same amount (grid uniform in eps_tot? no: quadratic). We therefore build the inelastic gain by interpolation per radial cell (as in ekin.py but with the local eps_kin grid).
To keep the implementation tractable we use a UNIFORM grid in eps_tot (so shifts are exact to interpolation) with n_e cells.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
from ekin import load_argon, GAMMA
from ekin_cyl import AngularBasis, real_Ylm
e,me=co.e,co.m_e

class CylKineticsTE:
    def __init__(self,xs,R,ng,omega,phi,L=2,Nh=1,n_e=120,umax=50.,m=16,cluster=2.0,Tg=300.,wall_absorb=False,stagger=True,order=2):
        """phi: potential at the m radial cell centres (descending r, phi(R)=0, phi>=0 inside). The eps_tot grid depends on phi, so the object is rebuilt when phi changes."""
        self.R,self.ng,self.omega,self.L,self.Nh,self.m=R,ng,omega,L,Nh,m
        self.kTg=co.k*Tg/e; self.wall_absorb=wall_absorb; self.xs=xs; self.stagger=stagger; self.order=order; self._theta=None
        self.ang=AngularBasis(L); self.nlm=self.ang.nlm; self.H=2*Nh+1
        # radial FV grid (same as ekin_cyl_fv)
        s=np.linspace(0,1,m+1); Xe=np.sort(1-(1-s)**cluster)
        X=0.5*(Xe[1:]+Xe[:-1]); o=np.argsort(-X); self.X=X[o]; self.r=R*self.X; self.dX=np.diff(Xe)[o]; self.wdisk=(Xe[1:]**2-Xe[:-1]**2)[o]; self.Xe_desc=Xe[::-1]
        self.phi=np.asarray(phi,float); assert len(self.phi)==m
        # total-energy grid: uniform from -e*phi_max to umax
        self.Wmin=-np.max(self.phi); self.n_e=n_e
        self.eb=np.linspace(self.Wmin,umax,n_e+1); self.ec=0.5*(self.eb[1:]+self.eb[:-1]); self.de=self.eb[1]-self.eb[0]
        # kinetic energy per (i,k)
        self.ukin=self.ec[:,None]+self.phi[None,:]            # (n_e, m)
        self.allowed=self.ukin>0
        self._angular(); self._build_cells()
    def _angular(self):
        ang=self.ang; TH,PH=ang.TH,ang.PH
        Yflip=np.array([real_Ylm(l,mm,np.pi-TH,-PH) for l,mm in ang.lm]); P=np.einsum('aij,bij,ij->ab',ang.Y,Yflip,ang.W); self.Pflip=P
        self.Mcos=ang.mult(np.cos(TH)); self.mu,self.Q=np.linalg.eigh(self.Mcos); self.Refl=self.Q.T@self.Pflip@self.Q
        self.Fth_v,self.Fth_a=ang.force_ops((0,1,0)); self.Fr_v,self.Fr_a=ang.force_ops((1,0,0)); self.Rot_z=ang.rotation_op((0,0,1)); vhat=ang.unit_vectors()[0]; self.Mvth=ang.mult(vhat[1]); self.Cent=-(self.Mvth@self.Rot_z)
    def _build_cells(self):
        """per radial cell k: energy-space operators on the kinetic energies ukin[:,k] (allowed cells only)."""
        xs=self.xs; ng=self.ng; m=self.m; ne=self.n_e
        sig=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1])
        self.v=np.where(self.allowed,GAMMA*np.sqrt(np.maximum(self.ukin,1e-12)),0.0)      # (ne,m)
        self.w=np.where(self.allowed,np.sqrt(np.maximum(self.ukin,0))*self.de,0.0)          # integration weight per cell (sqrt(ukin) d eps)
        self.num=np.where(self.allowed,ng*sig(xs['elastic'],self.ukin)*self.v,0.0)
        thi,di=xs['iz']; self.nuiz=np.where(self.allowed,ng*sig(di,self.ukin)*self.v*(self.ukin>thi),0.0)
        self.inel=[(th,np.where(self.allowed,ng*sig(d,self.ukin)*self.v*(self.ukin>th),0.0)) for th,d,_ in xs['inel']]
        self.nu_loss_l=self.num+sum(nu for _,nu in self.inel)+self.nuiz
        # per cell k: C0_k (ne x ne): elastic SG flux in eps (uniform grid, coefficients at edges) + inelastic shifts (exact: shift by th/de cells with linear weights)
        self.C0=[]; self.Dv=[]; self.D0=[]; self.Dvc=[]
        for k in range(m):
            uk=self.ukin[:,k]; ok=self.allowed[:,k]
            # elastic: W=-(2m/M) gamma N u^2 sigma_m, D = -kTg W on edges (edge kinetic energy)
            ue=self.eb+self.phi[k]; oke=ue>0
            W=np.where(oke,-2*xs['mM']*GAMMA*ng*np.maximum(ue,0)**2*sig(xs['elastic'],np.maximum(ue,0)),0.0); D=-self.kTg*W
            M=sp.lil_matrix((ne,ne))
            for j in range(1,ne):
                if not (ok[j-1] and ok[j]): continue
                h=self.de; Wj,Dj=W[j],D[j]
                if abs(Dj)<1e-300: cL,cR=(Wj,0.0) if Wj>0 else (0.0,Wj)
                else:
                    z=Wj*h/Dj
                    if abs(z)<1e-8: cL,cR=0.5*Wj-Dj/h,0.5*Wj+Dj/h
                    else: ez=np.exp(np.clip(z,-700,700)); cL=Wj*ez/(ez-1); cR=-Wj/(ez-1)
                M[j,j-1]+=cL/h; M[j,j]+=cR/h; M[j-1,j-1]-=cL/h; M[j-1,j]-=cR/h
            Cel=sp.diags(np.where(ok,1/np.sqrt(np.maximum(uk,1e-12)),0.0))@M.tocsr()
            # inelastic: loss + gain from eps_tot + th (uniform grid: shift s=th/de cells, linear weights), particle conserving per column
            Cin=sp.lil_matrix((ne,ne))
            def gain(th,nuk,share=1.0,mult=1.0):
                """parent-based scatter: every parent cell j (allowed, nuk>0) deposits its full loss w_j nuk_j (times mult) into the daughter
                cells around the daughter energy u_d = share*(u_j - th), linear interpolation on the eps_tot grid, clamped to the lowest
                allowed cell (daughters born inside the partially forbidden cell straddling eps_kin=0) -> exact conservation per column."""
                G=sp.lil_matrix((ne,ne)); idxk=np.where(ok)[0]; ilo=idxk[0]
                for j in idxk:
                    if nuk[j,k]<=0: continue
                    ud=share*(uk[j]-th)
                    it=(ud-self.phi[k]-self.eb[0])/self.de-0.5          # fractional daughter index on the eps_tot grid
                    if it<=ilo: i,t=ilo,0.0
                    else: i=int(np.floor(it)); t=it-i
                    if i+1>=ne: i,t=ne-1,0.0
                    tot=self.w[j,k]*nuk[j,k]*mult
                    G[i,j]+=tot*(1-t)/self.w[i,k]
                    if t>0: G[i+1,j]+=tot*t/self.w[i+1,k]
                return G.tocsr()
            for th,nuk in self.inel: Cin=Cin-sp.diags(nuk[:,k]); Cin=Cin+gain(th,nuk)
            Cin=Cin-sp.diags(self.nuiz[:,k]); Cin=Cin+gain(thi,self.nuiz,share=0.5,mult=1.0)
            self.C0.append((Cel+Cin.tocsr()).tocsr())
            # RF force term, STAGGERED pair: f_{l>=1} of cell i is attributed to the upper energy face i+1/2 ->
            #   Dv  = forward difference (df0/dv at the face i+1/2; backward at the last allowed cell),
            #   D0  = conservative (1/v^2) d/dv (v^2 f1) with the face fluxes v_{i+1/2}^2 f1_i and v_{i-1/2}^2 f1_{i-1}.
            # The product D0*Dv is then the compact 3-point energy-diffusion stencil, which removes the odd-even (checkerboard)
            # mode of the centred pair that is undamped where the elastic diffusion is weak (Ramsauer minimum, bottom of the well).
            # With stagger=False the classical centred operators are used (as in ekin.py / ekin_cyl_fv.py).
            vk=self.v[:,k]; Dv=sp.lil_matrix((ne,ne)); D0=sp.lil_matrix((ne,ne)); Dvc=sp.lil_matrix((ne,ne))
            idx=np.where(ok)[0]
            # centred d/dv (used for the couplings between l>=1 harmonics, where no checkerboard mode arises)
            for ii,i in enumerate(idx):
                if 0<ii<len(idx)-1:
                    im,ip=idx[ii-1],idx[ii+1]; h1=vk[i]-vk[im]; h2=vk[ip]-vk[i]
                    Dvc[i,im]=-h2/(h1*(h1+h2)); Dvc[i,i]=(h2-h1)/(h1*h2); Dvc[i,ip]=h1/(h2*(h1+h2))
                elif ii==0 and len(idx)>1: ip=idx[1]; Dvc[i,i]=-1/(vk[ip]-vk[i]); Dvc[i,ip]=1/(vk[ip]-vk[i])
                elif ii==len(idx)-1 and len(idx)>1: im=idx[ii-1]; Dvc[i,im]=-1/(vk[i]-vk[im]); Dvc[i,i]=1/(vk[i]-vk[im])
            self.Dvc.append(Dvc.tocsr())
            if self.stagger:
                for ii,i in enumerate(idx):
                    if ii<len(idx)-1: ip=idx[ii+1]; Dv[i,i]=-1/(vk[ip]-vk[i]); Dv[i,ip]=1/(vk[ip]-vk[i])
                    elif ii>0:        im=idx[ii-1]; Dv[i,im]=-1/(vk[i]-vk[im]); Dv[i,i]=1/(vk[i]-vk[im])
                for ii,i in enumerate(idx):
                    if self.w[i,k]<=0: continue
                    c=2/(GAMMA**3*self.w[i,k])
                    if ii<len(idx)-1: D0[i,i]+=c*GAMMA**2*max(ue[i+1],0.0)                       # upper face flux v^2 f1_i (zero at the top)
                    if ii>0: im=idx[ii-1]; D0[i,im]-=c*GAMMA**2*max(ue[i],0.0)                   # lower face flux v^2 f1_{i-1}
            else:
                for ii,i in enumerate(idx):
                    if 0<ii<len(idx)-1:
                        im,ip=idx[ii-1],idx[ii+1]; h1=vk[i]-vk[im]; h2=vk[ip]-vk[i]
                        Dv[i,im]=-h2/(h1*(h1+h2)); Dv[i,i]=(h2-h1)/(h1*h2); Dv[i,ip]=h1/(h2*(h1+h2))
                    elif ii==0 and len(idx)>1: ip=idx[1]; Dv[i,i]=-1/(vk[ip]-vk[i]); Dv[i,ip]=1/(vk[ip]-vk[i])
                    elif ii==len(idx)-1 and len(idx)>1: im=idx[ii-1]; Dv[i,im]=-1/(vk[i]-vk[im]); Dv[i,i]=1/(vk[i]-vk[im])
                for i in idx:
                    if self.w[i,k]<=0: continue
                    c=2/(GAMMA**3*self.w[i,k])
                    if i+1<ne and ok[i+1]: v2=GAMMA**2*max(ue[i+1],0.0); D0[i,i]+=c*v2*0.5; D0[i,i+1]+=c*v2*0.5
                    if i-1>=0 and ok[i-1]: v2=GAMMA**2*max(ue[i],0.0);   D0[i,i]-=c*v2*0.5; D0[i,i-1]-=c*v2*0.5
            self.Dv.append(Dv.tocsr())
            self.D0.append(D0.tocsr())
        # streaming at fixed eps_tot in CONSERVATIVE finite-volume form with FACE velocities:
        #   v mu df/dr|_eps + (a_r/v)(1-mu^2) df/dmu  =  (1/v)(1/r) d(r v^2 mu f)/dr|_eps  -  mu v f/r  +  (a_r/v) d/dmu[(1-mu^2) f]
        # (a_r = (e/m) dphi/dr ; v = v(eps_tot,r)).  Term 1: upwind flux balance with the face velocity v_f = GAMMA sqrt(eps+phi_face)
        # (telescopes under the weights w_ik wdisk_k = (v_ik de/GAMMA) wdisk_k -> exact discrete particle conservation); term 2 cancels
        # exactly against the l=0 row of the centrifugal operator; term 3 has no l=0 projection (angular part of the electrostatic force,
        # the only piece that survives the change of variables to total energy).  Faces with eps+phi_face<=0 carry no flux; a face between
        # an allowed and a forbidden cell is a turning point: the outgoing flux (mu>0) returns as incoming flux (mu<0) through the
        # reflected ghost (v_r -> -v_r), exactly as at a specular wall.
        Xe=self.Xe_desc; phi_face=np.interp(Xe,self.X[::-1],self.phi[::-1])           # m+1 faces; face k = outer face of cell k
        uf=self.ec[:,None]+phi_face[None,:]
        self.vf=np.where(uf>0,GAMMA*np.sqrt(np.maximum(uf,1e-30)),0.0)                 # (ne, m+1)
        self.g=1/(self.X*self.dX); self.ar=(e/me)*np.gradient(self.phi,self.r)         # a_r = (e/m) dphi/dr [m/s^2]
        self.nTot=self.H*self.nlm*self.n_e*m
    def idx(self,n,a,i,k): return ((n+self.Nh)*self.nlm+a)*self.n_e*self.m+i*self.m+k
    def _stream_arrays(self):
        """Streaming operator coefficients in (i,k) ordering: dict offset -> array (ndir, ne, m) for the cell k+offset, plus the
        ghost factor s_ik.  Face values: first-order upwind (order=1) or Fromm's linear reconstruction from the upwind cell with a
        centred slope (order=2; falls back to first order next to the wall, the axis and forbidden cells)."""
        ne,m=self.n_e,self.m; ok=self.allowed; vinv=np.where(ok,1/np.maximum(self.v,1e-30),0.0); Xe=self.Xe_desc; X=self.X; g=self.g; vf=self.vf
        nd=len(self.mu); C={off:np.zeros((nd,ne,m)) for off in (-2,-1,0,1,2)}
        okm=np.zeros((ne,m+1),bool); okm[:,1:]=ok
        th=self._theta
        def face_coefs(j,sign,kk):
            """coefficients {cell offset relative to j: weight(i)} of the face value at face j (outer face of cell j) for mu>0 (sign=+1,
            upwind cell j) or mu<0 (sign=-1, upwind cell j-1); zero where the upwind cell is forbidden."""
            out={}
            if sign>0:
                base=ok[:,j].astype(float); out[0]=base.copy()
                if self.order==2 and 1<=j<m-1:
                    d=(Xe[j]-X[j])/(X[j-1]-X[j+1])*(ok[:,j-1]&ok[:,j]&ok[:,j+1])*(1.0 if th is None else th[0][kk,:,j])
                    out[-1]=d; out[1]=-d; out[0]=base
            else:
                if j==0: return {}
                base=ok[:,j-1].astype(float); out[-1]=base.copy()
                if self.order==2 and 2<=j<m:
                    d=(Xe[j]-X[j-1])/(X[j-2]-X[j])*(ok[:,j-2]&ok[:,j-1]&ok[:,j])*(1.0 if th is None else th[1][kk,:,j])
                    out[-2]=d; out[0]=-d
            return out
        for kk,mu in enumerate(self.mu):
            sign=1 if mu>0 else -1
            for k in range(m):
                # outer face k: + o_k g_k mu vf_k^2 / v ;  inner face k+1: - i_k g_k mu vf_{k+1}^2 / v
                fo=mu*Xe[k]*g[k]*vf[:,k]**2*vinv[:,k]
                for off,c in face_coefs(k,sign,kk).items():
                    kk2=k+off
                    if 0<=kk2<m: C[off][kk,:,k]+=fo*c
                if k+1<m:
                    fi=-mu*Xe[k+1]*g[k]*vf[:,k+1]**2*vinv[:,k]
                    for off,c in face_coefs(k+1,sign,kk).items():
                        kk2=k+1+off
                        if 0<=kk2<m: C[off+1][kk,:,k]+=fi*c
            C[0][kk]-=mu*self.v/X[None,:]                                   # geometric term (cancelled by the centrifugal operator)
        refl=(~okm[:,:-1])&ok                                               # outer neighbour forbidden (or wall): reflected ghost
        if self.wall_absorb: refl[:,0]=False
        s_ik=np.where(refl, vf[:,:-1]**2*Xe[:-1][None,:]*g[None,:]*vinv, 0.0)
        for off in C: C[off]*=ok[None]
        return C,s_ik
    def kinetic_matrix(self,Eth,Bz):
        H,Nh,nlm,ne,m,ang=self.H,self.Nh,self.nlm,self.n_e,self.m,self.ang
        blocks={}
        def add(n1,a1,n2,a2,M):
            key=(n1,a1,n2,a2); blocks[key]=(blocks[key]+M) if key in blocks else M
        C,s_ik=self._stream_arrays()
        Gfix=self.Q@np.diag(np.where(self.mu<0,self.mu,0.0))@self.Refl@self.Q.T
        kidx=np.arange(ne*m).reshape(ne,m)
        def stream_block(a,b):
            qq=self.Q[a,:]*self.Q[b,:]; rows=[];cols=[];vals=[]
            for off,arr in C.items():
                v=np.einsum('k,kij->ij',qq,arr)/self.R
                if off==0: v=v+Gfix[a,b]*s_ik/self.R
                ks=np.arange(max(0,-off),min(m,m-off)); r=kidx[:,ks]; c=kidx[:,ks+off]
                rows.append(r.ravel()); cols.append(c.ravel()); vals.append(v[:,ks].ravel())
            M=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(ne*m,ne*m)); M.eliminate_zeros(); return M
        # energy-dependent multipliers as (ne*m) diagonals
        Vd=sp.diags(self.v.ravel()); vinv=np.where(self.allowed,1/np.maximum(self.v,1e-30),0.0); Vinv=sp.diags(vinv.ravel())
        Rinv=sp.diags(np.tile(1/self.r,ne)); ARV=sp.diags((self.ar[None,:]*vinv).ravel()); Fes=self.Fr_a-2*self.Mcos
        def cellblock(mats):  # list over k of (ne x ne) -> (ne*m x ne*m) with (i,k) ordering
            rows=[];cols=[];vals=[]
            for k,Mk in enumerate(mats):
                Mk=Mk.tocoo(); rows.append(Mk.row*m+k); cols.append(Mk.col*m+k); vals.append(Mk.data)
            return sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(ne*m,ne*m))
        C0b=cellblock(self.C0); Dvb=cellblock(self.Dv); D0b=cellblock(self.D0); Dvcb=cellblock(self.Dvc)
        Cl=sp.diags(-self.nu_loss_l.ravel())
        for n in range(-Nh,Nh+1):
            for a in range(nlm):
                add(n,a,n,a, 1j*n*self.omega*sp.identity(ne*m))
                if ang.lm[a][0]==0: add(n,a,n,a, -C0b)
                else:               add(n,a,n,a, -Cl)
                for b in range(nlm):
                    Sab=stream_block(a,b)
                    if Sab.nnz: add(n,a,n,b, Sab)
                    if abs(Fes[a,b])>1e-12: add(n,a,n,b, Fes[a,b]*ARV)          # angular electrostatic term
                    cc=self.Cent[a,b]
                    if abs(cc)>1e-12: add(n,a,n,b, cc*(Vd@Rinv))
        for q in range(-Nh,Nh+1):
            Eq=Eth[q+Nh]; Bq=Bz[q+Nh]
            if np.max(np.abs(Eq))<1e-30 and np.max(np.abs(Bq))<1e-30: continue
            Ed=sp.diags(np.tile(-(e/me)*Eq,ne)); Bd=sp.diags(np.tile((e/me)*Bq,ne))
            for n in range(-Nh,Nh+1):
                mm=n-q
                if abs(mm)>Nh: continue
                for a in range(nlm):
                    for b in range(nlm):
                        fv,fa=self.Fth_v[a,b],self.Fth_a[a,b]
                        if a==0 and abs(fv)>1e-12: add(n,a,mm,b, fv*(D0b@Ed))     # l=0 row: conservative form (fa/fv=2 -> (1/v^2) d/dv v^2)
                        else:
                            Dsel=Dvb if ang.lm[b][0]==0 else Dvcb                   # staggered (forward) only for l=1 <- l=0; centred otherwise
                            if abs(fv)>1e-12: add(n,a,mm,b, fv*(Dsel@Ed))
                            if abs(fa)>1e-12: add(n,a,mm,b, fa*(Vinv@Ed))
                        rz=self.Rot_z[a,b]
                        if abs(rz)>1e-12: add(n,a,mm,b, rz*Bd)
        rows=[[blocks.get((n1,a1,n2,a2),None) for n2 in range(-Nh,Nh+1) for a2 in range(nlm)] for n1 in range(-Nh,Nh+1) for a1 in range(nlm)]
        A=sp.bmat(rows,format='csr')
        # forbidden cells: f=0 (identity rows)
        forb=np.where(~self.allowed.ravel())[0]
        if len(forb):
            allrows=np.concatenate([((n+Nh)*nlm+a)*ne*m+forb for n in range(-Nh,Nh+1) for a in range(nlm)])
            keep=np.ones(A.shape[0]); keep[allrows]=0
            A=sp.diags(keep)@A+sp.csr_matrix((np.ones(len(allrows)),(allrows,allrows)),shape=A.shape)
        return A
    def limiter_theta(self,f):
        """minmod slope limiter (frozen-coefficient form) from the direction components g_kk = sum_a Q[a,kk] f_a of the n=0 harmonic:
        theta in [0,1] multiplies the second-order correction of the face value; 0 where the slope changes sign (extrema)."""
        ne,m=self.n_e,self.m; X=self.X; g=np.einsum('ak,aij->kij',self.Q,np.real(f[self.Nh]))     # (ndir, ne, m)
        nd=len(self.mu); thp=np.zeros((nd,ne,m+1)); thm=np.zeros((nd,ne,m+1))
        def mm(a,b): return np.where(a*b>0,np.sign(a)*np.minimum(np.abs(a),np.abs(b)),0.0)
        for j in range(1,m-1):      # faces with full stencils for mu>0 (cells j-1,j,j+1)
            sL=(g[:,:,j-1]-g[:,:,j])/(X[j-1]-X[j]); sR=(g[:,:,j]-g[:,:,j+1])/(X[j]-X[j+1]); sC=(g[:,:,j-1]-g[:,:,j+1])/(X[j-1]-X[j+1])
            thp[:,:,j]=np.clip(np.where(np.abs(sC)>1e-300,mm(sL,sR)/np.where(np.abs(sC)>1e-300,sC,1.0),0.0),0,1)
        for j in range(2,m):        # faces with full stencils for mu<0 (cells j-2,j-1,j)
            sL=(g[:,:,j-2]-g[:,:,j-1])/(X[j-2]-X[j-1]); sR=(g[:,:,j-1]-g[:,:,j])/(X[j-1]-X[j]); sC=(g[:,:,j-2]-g[:,:,j])/(X[j-2]-X[j])
            thm[:,:,j]=np.clip(np.where(np.abs(sC)>1e-300,mm(sL,sR)/np.where(np.abs(sC)>1e-300,sC,1.0),0.0),0,1)
        return thp,thm
    def solve(self,Eth,Bz,passes=2,limiter_iters=3):
        """order=2 with limiter_iters>0: first-order solve, then limited second-order solves with the limiter frozen from the previous
        iterate (TVD-type minmod); limiter_iters=0 -> unlimited Fromm reconstruction."""
        if self.order==2 and limiter_iters>0:
            self._theta=(np.zeros((len(self.mu),self.n_e,self.m+1)),)*2
            f=self._solve(Eth,Bz,passes); hist=[]
            for it in range(limiter_iters):
                self._theta=self.limiter_theta(f); fn=self._solve(Eth,Bz,passes)
                hist.append(float(np.max(np.abs(fn[self.Nh,0]-f[self.Nh,0]))/np.max(np.abs(fn[self.Nh,0])))); f=fn
            self.limiter_hist=hist; self._theta=None; return f
        return self._solve(Eth,Bz,passes)
    def _solve(self,Eth,Bz,passes=2):
        """Bordered (Lagrange-multiplier) solve: [A c; w^T 0][f; lam] = [0; 1].  w = density weights (normalisation sum_k wdisk_k sum_i w_ik f00_ik = 1).
        lam absorbs the residual non-conservation of the discrete operators (top-of-grid inelastic parents, centred Dv) instead of
        dumping it into one replaced equation.  Pass 1: c = w ; pass 2: c = f from pass 1 (residual distributed proportionally to f,
        i.e. lam = effective uniform net loss frequency; |lam| << nu is the conservation check, stored in self.lam)."""
        import time; t0=time.time(); A=self.kinetic_matrix(Eth,Bz).tocsr(); nTot=A.shape[0]; ne,m=self.n_e,self.m; self.t_assemble=time.time()-t0
        wv=np.zeros(nTot); base=self.idx(0,0,0,0)
        wv[base:base+ne*m]=(self.w*self.wdisk[None,:]).ravel()
        c=wv.copy(); t0=time.time()
        for p in range(passes):
            Ab=sp.bmat([[A,sp.csc_matrix(c[:,None])],[sp.csr_matrix(wv[None,:]),None]],format='csc')
            b=np.zeros(nTot+1,complex); b[-1]=1.0
            x=spla.spsolve(Ab,b); f=x[:-1]; self.lam=x[-1]
            c=np.zeros(nTot); c[base:base+ne*m]=np.maximum(np.real(f[base:base+ne*m]),0.0)
        self.t_solve=time.time()-t0
        return f.reshape(self.H,self.nlm,ne,m)
    def density(self,f):     return np.real(np.einsum('ik,ik->k',self.w,f[self.Nh,0]))
    def mean_energy(self,f): return np.real(np.einsum('ik,ik->k',self.w*np.maximum(self.ukin,0),f[self.Nh,0]))/np.maximum(self.density(f),1e-30)
    def nu_iz(self,f):       return np.real(np.einsum('ik,ik->k',self.w*self.nuiz,f[self.Nh,0]))/np.maximum(self.density(f),1e-30)
