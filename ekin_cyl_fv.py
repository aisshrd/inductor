"""ekin_cyl_fv.py -- CylKinetics variant with FINITE-VOLUME radial discretisation (upwind in S_N eigen-directions of cos(theta)),
wall-clustered grid, and specular reflection as an exact reflection map in direction space.  Same interface as CylKinetics."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
from ekin_cyl import CylKinetics, AngularBasis, real_Ylm
from ekin import GAMMA
e,me=co.e,co.m_e

class CylKineticsFV(CylKinetics):
    def __init__(self,xs,R,ng,omega,L=2,Nh=1,nu=120,umax=60.,m=24,Tg=300.,cluster=2.5,wall_absorb=False):
        self.cluster=cluster; self.wall_absorb=wall_absorb
        # build base (energy/angular operators); then override radial grid
        self.R,self.ng,self.omega,self.L,self.Nh,self.m=R,ng,omega,L,Nh,m
        from ekin import EnergyGridKinetics
        self.K=EnergyGridKinetics(xs,umax=umax,nu=nu,L=1,Nh=0,ng=ng,Tg=Tg)
        self.nu=nu; self.uc=self.K.uc; self.w=self.K.w; self.v=GAMMA*np.sqrt(self.uc)
        self.ang=AngularBasis(L); self.nlm=self.ang.nlm; self.H=2*Nh+1
        # radial FV grid: cell edges clustered towards the wall:  X_edge = 1 - (1-s)^cluster ... use s in [0,1]
        s=np.linspace(0,1,m+1); Xe=1-(1-s)**cluster; Xe=np.sort(Xe); self.Xe=Xe
        self.X=0.5*(Xe[1:]+Xe[:-1]); self.r=R*self.X; self.dX=np.diff(Xe)
        self.wdisk=(Xe[1:]**2-Xe[:-1]**2)          # area fractions, sum = 1  (mean over disk = sum w n)
        # order descending X like the Chebyshev version (index 0 = wall) for compatibility with Discharge
        o=np.argsort(-self.X); self.X=self.X[o]; self.r=self.r[o]; self.dX=self.dX[o]; self.wdisk=self.wdisk[o]
        self.Xe_desc=Xe[::-1]                      # edges descending: Xe_desc[k] = outer edge of cell k, Xe_desc[k+1] inner edge
        self.grid={'X':self.X,'Deven':self._fd_even(),'Dodd':self._fd_odd(),'m':m}   # used by Discharge for phi', Maxwell
        self._build_common()
    def _fd_matrix(self,parity):
        """second-order derivative on non-uniform cell centres (descending X); at the wall: one-sided 3-point;
        at the axis: mirror continuation with given parity (+1 even, -1 odd)."""
        X=self.X; m=len(X); D=np.zeros((m,m))
        for k in range(m):
            if k==0:   # one-sided using k, k+1, k+2
                x0,x1,x2=X[0],X[1],X[2]; h1=x0-x1; h2=x0-x2
                D[0,0]=(h1+h2)/(h1*h2); D[0,1]=-h2/(h1*(h2-h1)); D[0,2]=h1/(h2*(h2-h1))
            else:
                xm=X[k-1]; x0=X[k]; xp=X[k+1] if k<m-1 else -X[k]
                h1=xm-x0; h2=x0-xp
                cm=h2/(h1*(h1+h2)); c0=(h1-h2)/(h1*h2); cp=-h1/(h2*(h1+h2))
                D[k,k-1]+=cm; D[k,k]+=c0
                if k<m-1: D[k,k+1]+=cp
                else: D[k,k]+=parity*cp
        return D
    def _fd_even(self): return self._fd_matrix(+1)
    def _fd_odd(self):  return self._fd_matrix(-1)
    def _build_common(self):
        ang,nu,m=self.ang,self.nu,self.m
        TH,PH=ang.TH,ang.PH
        Yflip=np.array([real_Ylm(l,mm,np.pi-TH,-PH) for l,mm in ang.lm])
        P=np.einsum('aij,bij,ij->ab',ang.Y,Yflip,ang.W); self.parity=np.round(np.diag(P)).astype(int); self.Pflip=P
        self.Mcos=ang.mult(np.cos(TH))
        mu,Q=np.linalg.eigh(self.Mcos); self.mu=mu; self.Q=Q           # Mcos = Q diag(mu) Q^T
        self.Fr_v,self.Fr_a=ang.force_ops((1,0,0)); self.Fth_v,self.Fth_a=ang.force_ops((0,1,0))
        self.Rot_z=ang.rotation_op((0,0,1)); vhat=ang.unit_vectors()[0]; self.Mvth=ang.mult(vhat[1]); self.Cent=-(self.Mvth@self.Rot_z)
        self.Dv=self.K._dv_matrix(self.v)
        self.C0=(self.K.Cel0+self.K.Cin0).tocsr(); self.Cl=(-(self.K.Cm+self.K.Closs_l)).tocsr()
        self.nTot=self.H*self.nlm*nu*m
        self._build_streaming()
    def _build_streaming(self):
        """S_N upwind: for eigen-direction k with mu_k>0 (moving outward, towards wall): flux through outer edge uses cell value;
        for mu_k<0 inward.  Cylindrical FV: (1/r) d(r mu v f)/dr -> [r_out F_out - r_in F_in]/(r dr).
        Returns radial operators per eigen-direction: list of (m x m) sparse matrices acting on cell values (descending X)."""
        m=self.m; Xe=self.Xe_desc; X=self.X; dX=self.dX; R=self.R
        self.Sdir=[]
        for mu in self.mu:
            A=np.zeros((m,m))
            for k in range(m):
                ro,ri=Xe[k],Xe[k+1]                     # outer, inner edge (X units)
                # non-conservative advective form mu * df/dr (the geometric v_r/r term is already in the centrifugal operator)
                if mu>0:   # outward: upwind from inner neighbour
                    A[k,k]+=mu/dX[k]
                    if k+1<m: A[k,k+1]-=mu/dX[k]
                    else: A[k,k]-=mu/dX[k]        # axis: even continuation f(-X)=f(X) (inner neighbour of last cell is itself mirrored)
                else:      # inward: upwind from outer neighbour
                    A[k,k]-=mu/dX[k]
                    if k>0: A[k,k-1]+=mu/dX[k]
                    # wall (k=0): incoming flux from the wall -> handled by reflection map (added in kinetic_matrix)
            self.Sdir.append(A/R)                      # d/dr = (1/R) d/dX
        # Reflection at wall (specular): v_r -> -v_r is the parity flip in angle space: f_in = Pflip f_out.
        # In eigen-direction space: incoming (mu<0) components at the wall ghost cell = Pflip applied to outgoing. We implement as:
        # for mu<0 directions, the 'outer neighbour' of cell 0 is the ghost with value (Q^T Pflip Q f)_k  (mixes directions).
        self.Refl=self.Q.T@self.Pflip@self.Q          # in eigen basis: maps full direction vector to reflected one
    def kinetic_matrix(self,Eth,Bz,phi):
        H,Nh,nlm,nu,m,ang=self.H,self.Nh,self.nlm,self.nu,self.m,self.ang
        Im=sp.identity(m); Inu=sp.identity(nu)
        v=self.v; Vinv=sp.diags(1/v); Vd=sp.diags(v)
        blocks={}
        def add(n1,a1,n2,a2,M):
            key=(n1,a1,n2,a2); blocks[key]=(blocks[key]+M) if key in blocks else M
        Er=-(self.grid['Deven']@phi)/self.R
        # streaming in eigen basis: operator on harmonic coefficients = Q [sum_k e_k e_k^T (x) Sdir_k] Q^T
        # build the (nlm*m x nlm*m) radial-angular streaming block once (energy enters as v multiplier)
        Qs=sp.csr_matrix(self.Q)
        S_eig=sp.block_diag([sp.csr_matrix(Sk) for Sk in self.Sdir],format='csr')        # (nlm*m) with direction-major ordering
        # wall reflection for incoming directions: ghost value = (Refl f_dir)_k at cell 0 for mu_k<0: flux term + ro*mu_k/(X0 dX0) * ghost
        Wr=sp.lil_matrix((nlm*m,nlm*m)); ro=self.Xe_desc[0]
        for k,mu in enumerate(self.mu):
            if mu<0:
                coef=mu/self.dX[0]/self.R
                for j in range(nlm):
                    if abs(self.Refl[k,j])>1e-12:
                        Wr[k*m+0,j*m+0]+= coef*self.Refl[k,j]*(0.0 if (self.wall_absorb) else 1.0)
        S_eig=S_eig+Wr.tocsr()
        T=sp.kron(Qs,Im); S_harm=(T@S_eig@T.T).tocsr()                                  # harmonic-major ordering (a,k)
        for n in range(-Nh,Nh+1):
            for a in range(nlm):
                add(n,a,n,a, 1j*n*self.omega*sp.kron(Inu,Im))
                if ang.lm[a][0]==0: add(n,a,n,a, -sp.kron(self.C0,Im))
                else:               add(n,a,n,a, -sp.kron(self.Cl,Im))
                for b in range(nlm):
                    Sab=S_harm[a*m:(a+1)*m, b*m:(b+1)*m]
                    if Sab.nnz: add(n,a,n,b, sp.kron(Vd,Sab))
                    cc=self.Cent[a,b]
                    if abs(cc)>1e-12: add(n,a,n,b, cc*sp.kron(Vd,sp.diags(1/self.r)))
                    fv,fa=self.Fr_v[a,b],self.Fr_a[a,b]
                    if abs(fv)>1e-12: add(n,a,n,b, fv*sp.kron(self.Dv,sp.diags(-(e/me)*Er)))
                    if abs(fa)>1e-12: add(n,a,n,b, fa*sp.kron(Vinv,sp.diags(-(e/me)*Er)))
        for q in range(-Nh,Nh+1):
            Eq=Eth[q+Nh]; Bq=Bz[q+Nh]
            if np.max(np.abs(Eq))<1e-30 and np.max(np.abs(Bq))<1e-30: continue
            for n in range(-Nh,Nh+1):
                mm=n-q
                if abs(mm)>Nh: continue
                for a in range(nlm):
                    for b in range(nlm):
                        fv,fa=self.Fth_v[a,b],self.Fth_a[a,b]
                        if abs(fv)>1e-12: add(n,a,mm,b, fv*sp.kron(self.Dv,sp.diags(-(e/me)*Eq)))
                        if abs(fa)>1e-12: add(n,a,mm,b, fa*sp.kron(Vinv,sp.diags(-(e/me)*Eq)))
                        rz=self.Rot_z[a,b]
                        if abs(rz)>1e-12: add(n,a,mm,b, rz*sp.kron(Inu,sp.diags((e/me)*Bq)))
        rows=[[blocks.get((n1,a1,n2,a2),None) for n2 in range(-Nh,Nh+1) for a2 in range(nlm)] for n1 in range(-Nh,Nh+1) for a1 in range(nlm)]
        A=sp.bmat(rows,format='lil')
        return A,[]
