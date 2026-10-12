"""
ekin_cyl.py -- cylindrical ICP electron kinetics on an energy grid with full spherical-harmonic angular expansion,
time harmonics (harmonic balance), Chebyshev collocation in radius, self-consistent E_theta/B_z (Maxwell),
ambipolar potential from quasineutrality, and self-consistent ion profile (fluid ions with Bohm boundary).

f(r, v, t) = sum_{n,l,m} f_{nlm}(eps, r) e^{i n w t} Y_lm(theta, phi)   (real spherical harmonics, l<=L)
velocity angles: theta from the radial axis r, phi from theta_hat (azimuth) in the (theta_hat, z_hat) plane.

Operators on the angular basis are built numerically once by quadrature on the sphere (exact for the truncated basis):
   streaming   v cos(theta) d/dr  -> matrix S_r (couples l,l+-1)
   force along r :  F_r d/dv_r   -> split into radial-derivative part (d/dv) and angular part
   force along theta_hat: F_th d/dv_theta
   rotation (centrifugal + magnetic) : acts only on angles.
General force operator for a unit vector n:  n . grad_v f = n_v d f/dv + (1/v) n_perp . grad_Omega f
where n_v = n . v_hat ;  we compute matrices  <Y_a | n_v | Y_b>  (multiplication) and  <Y_a | n_perp.grad_Omega | Y_b>  by quadrature.

Energy grid: same conservative FV as ekin.py for the l=0 energy flux (elastic + inelastic), losses for l>=1.
Unknown vector: a[n, lm, i_e, i_r];  fields eps_n(r), b_n(r), phi(r), n_i(r) (ions).
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
from scipy.special import sph_harm_y as _shy
def sph_harm(m,l,ph,th): return _shy(l,m,th,ph)
from ekin import load_argon, GAMMA, EnergyGridKinetics
e,me=co.e,co.m_e

# ----------------------------------------------------------------- real spherical harmonics on a quadrature sphere
def real_Ylm(l,m,th,ph):
    """real spherical harmonics, orthonormal on the sphere"""
    if m==0: return np.real(sph_harm(0,l,ph,th))
    if m>0:  return np.sqrt(2)*np.real(sph_harm(m,l,ph,th))*(-1)**m
    return   np.sqrt(2)*np.imag(sph_harm(-m,l,ph,th))*(-1)**m

class AngularBasis:
    def __init__(self,L,nth=None,nph=None):
        self.L=L; self.lm=[(l,m) for l in range(L+1) for m in range(-l,l+1)]; self.nlm=len(self.lm)
        nth=nth or 2*L+6; nph=nph or 4*L+8
        x,wx=np.polynomial.legendre.leggauss(nth); th=np.arccos(x); ph=2*np.pi*np.arange(nph)/nph
        TH,PH=np.meshgrid(th,ph,indexing='ij'); W=np.outer(wx,np.full(nph,2*np.pi/nph))
        self.TH,self.PH,self.W=TH,PH,W
        self.Y=np.array([real_Ylm(l,m,TH,PH) for l,m in self.lm])         # nlm x nth x nph
        # derivatives on the sphere by finite differences in theta (spectral in phi) -- use analytic via small-step
        h=1e-5
        self.dY_dth=np.array([(real_Ylm(l,m,TH+h,PH)-real_Ylm(l,m,TH-h,PH))/(2*h) for l,m in self.lm])
        self.dY_dph=np.array([(real_Ylm(l,m,TH,PH+h)-real_Ylm(l,m,TH,PH-h))/(2*h) for l,m in self.lm])
        G=np.einsum('aij,bij,ij->ab',self.Y,self.Y,W); assert np.max(np.abs(G-np.eye(self.nlm)))<1e-8
    def mult(self,g):
        """matrix of multiplication by angular function g(th,ph): M_ab = <Y_a| g |Y_b>"""
        return np.einsum('aij,ij,bij,ij->ab',self.Y,g,self.Y,self.W)
    def angular_grad(self,gth,gph):
        """matrix of operator  gth * d/dth + gph/sin(th) * d/dph  acting on Y_b, projected on Y_a"""
        sth=np.sin(self.TH)
        return np.einsum('aij,ij,bij,ij->ab',self.Y,gth,self.dY_dth,self.W)+np.einsum('aij,ij,bij,ij->ab',self.Y,gph/np.maximum(sth,1e-12),self.dY_dph,self.W)
    def unit_vectors(self):
        """components of v_hat, theta_hat_v (polar unit), phi_hat_v in the frame (r, th_az, z)"""
        TH,PH=self.TH,self.PH
        vhat=np.array([np.cos(TH),np.sin(TH)*np.cos(PH),np.sin(TH)*np.sin(PH)])
        ehat=np.array([-np.sin(TH),np.cos(TH)*np.cos(PH),np.cos(TH)*np.sin(PH)])   # d vhat / d theta
        phat=np.array([0*TH,-np.sin(PH),np.cos(PH)])                                 # (1/sin th) d vhat / d phi
        return vhat,ehat,phat
    def force_ops(self,n):
        """for unit force direction n=(n_r,n_th,n_z): returns (A_v, A_ang) with  n.grad_v f = A_v * (df/dv) + (1/v) A_ang * f  (angular matrices)"""
        vhat,ehat,phat=self.unit_vectors()
        nv=sum(n[k]*vhat[k] for k in range(3)); ne=sum(n[k]*ehat[k] for k in range(3)); nph=sum(n[k]*phat[k] for k in range(3))
        return self.mult(nv), self.angular_grad(ne,nph)
    def rotation_op(self,axis):
        """operator (axis x v).grad_v  = angular only:  (axis x vhat) . grad_Omega"""
        vhat,ehat,phat=self.unit_vectors()
        c=np.cross(np.array(axis)[:,None,None]*np.ones_like(vhat),vhat,axis=0)
        ce=sum(c[k]*ehat[k] for k in range(3)); cp=sum(c[k]*phat[k] for k in range(3))
        return self.angular_grad(ce,cp)

# ----------------------------------------------------------------- radial grid (parity-adapted Chebyshev, as before)
def cheb(N):
    x=np.cos(np.pi*np.arange(N+1)/N); c=np.ones(N+1); c[0]=c[-1]=2; c*=(-1)**np.arange(N+1)
    X=np.tile(x,(N+1,1)).T; dX=X-X.T; D=np.outer(c,1/c)/(dX+np.eye(N+1)); D-=np.diag(D.sum(1)); return D,x
def cyl_grid(m):
    D0,x=cheb(2*m-1); Xp=x[:m]; Dpp=D0[:m,:m]; Dpm=D0[:m,2*m-1:m-1:-1]
    return dict(X=Xp,Deven=Dpp+Dpm,Dodd=Dpp-Dpm,m=m)
def disk_weights(Xp):
    from numpy.polynomial.chebyshev import Chebyshev
    m=len(Xp); T=np.array([[Chebyshev.basis(k)(2*xi-1) for k in range(m)] for xi in Xp]); mom=np.zeros(m)
    for k in range(m):
        q=(Chebyshev.basis(k)*Chebyshev([1,1]))*0.25; mom[k]=q.integ()(1)-q.integ()(-1)
    return np.linalg.solve(T.T,mom)

class CylKinetics:
    """units: energy eV, time s, length m, fields SI.  Unknowns dimensional f (EEDF-type, int f sqrt(u) du = n/n0)."""
    def __init__(self,xs,R,ng,omega,L=2,Nh=1,nu=120,umax=60.,m=10,Tg=300.):
        self.R,self.ng,self.omega,self.L,self.Nh,self.m=R,ng,omega,L,Nh,m
        self.K=EnergyGridKinetics(xs,umax=umax,nu=nu,L=1,Nh=0,ng=ng,Tg=Tg)      # reuse energy-grid collision operators
        self.nu=nu; self.uc=self.K.uc; self.w=self.K.w; self.v=GAMMA*np.sqrt(self.uc)
        self.ang=AngularBasis(L); self.nlm=self.ang.nlm; self.H=2*Nh+1
        self.grid=cyl_grid(m); self.X=self.grid['X']; self.r=R*self.X; self.wdisk=2*disk_weights(self.X)
        self._build()
    # ---------- static pieces
    def _build(self):
        ang,nu,m=self.ang,self.nu,self.m
        # parity of each (l,m) under r -> -r (v_r -> -v_r, v_th -> -v_th, v_z -> v_z): theta->pi-theta, phi->-phi
        TH,PH=ang.TH,ang.PH
        Yflip=np.array([real_Ylm(l,mm,np.pi-TH,-PH) for l,mm in ang.lm])
        P=np.einsum('aij,bij,ij->ab',ang.Y,Yflip,ang.W); self.parity=np.round(np.diag(P)).astype(int)   # +1 even, -1 odd
        assert np.max(np.abs(P-np.diag(np.diag(P))))<1e-8
        # streaming: v cos(theta) d/dr  -> angular matrix of cos(theta)
        self.Mcos=ang.mult(np.cos(TH))
        # force operators (unit directions)
        self.Fr_v,self.Fr_a=ang.force_ops((1,0,0)); self.Fth_v,self.Fth_a=ang.force_ops((0,1,0))
        self.Rot_z=ang.rotation_op((0,0,1))           # (z x v).grad_v  : magnetic force  -e v x B -> acceleration = -(e/m) v x B = (e/m) B (z x v)?? careful: v x (B z) = B (v x z) = -B (z x v)
        # centrifugal: (v_th^2/r) d/dv_r - (v_r v_th / r) d/dv_th = (v_th/r) * [v_th d/dv_r - v_r d/dv_th] = (v_th/r) * (z x v).grad_v ... check: (z x v) = (-v_th, v_r, 0) -> (z x v).grad = -v_th d/dv_r + v_r d/dv_th = -(bracket). So centrifugal = -(v_th/r) (z x v).grad_v
        vhat=ang.unit_vectors()[0]; self.Mvth=ang.mult(vhat[1])    # multiplication by v_th/v
        self.Cent=-(self.Mvth@self.Rot_z)                         # times v/r
        # energy-derivative matrix d/dv on the grid (centred), and v
        self.Dv=self.K._dv_matrix(self.v)
        # collision operators: l=0 full (elastic FV + inelastic), l>=1 loss only
        self.C0=(self.K.Cel0+self.K.Cin0).tocsr(); self.Cl=(-(self.K.Cm+self.K.Closs_l)).tocsr()
        # index helpers
        self.nTot=self.H*self.nlm*nu*m
        # radial derivative per parity
        self.Dr_even=self.grid['Deven']/self.R; self.Dr_odd=self.grid['Dodd']/self.R
    def idx(self,n,a,i,k): return ((n+self.Nh)*self.nlm+a)*self.nu*self.m+i*self.m+k
    # ---------- assemble kinetic matrix for given fields (harmonics of E_theta, B_z at nodes; phi at nodes)
    def kinetic_matrix(self,Eth,Bz,phi):
        """Eth,Bz: arrays (H, m) complex harmonics (E(t)=sum E_n e^{i n w t}); phi: (m,) potential [V] (static). Returns (A, bc_rows)"""
        H,Nh,nlm,nu,m,ang=self.H,self.Nh,self.nlm,self.nu,self.m,self.ang
        Im=sp.identity(m); Inu=sp.identity(nu); Ilm=sp.identity(nlm)
        v=self.v; Vinv=sp.diags(1/v); Vd=sp.diags(v)
        blocks={}
        def add(n1,a1,n2,a2,M):
            key=(n1,a1,n2,a2); blocks[key]=(blocks[key]+M) if key in blocks else M
        # ---- time derivative, collisions, streaming, centrifugal, electrostatic force: diagonal in n
        Er=-(self.grid['Deven']@phi)/self.R       # static radial field E_r = -dphi/dr
        for n in range(-Nh,Nh+1):
            for a in range(nlm):
                add(n,a,n,a, sp.kron(sp.kron(sp.csr_matrix(np.eye(1)*1j*n*self.omega),Inu),Im).tocsr() if False else 1j*n*self.omega*sp.kron(Inu,Im))
                # collisions
                if self.ang.lm[a][0]==0: add(n,a,n,a, -sp.kron(self.C0,Im))
                else:                    add(n,a,n,a, -sp.kron(self.Cl,Im))
                for b in range(nlm):
                    c=self.Mcos[a,b]
                    if abs(c)>1e-12:
                        Dr=self.Dr_even if self.parity[b]>0 else self.Dr_odd
                        add(n,a,n,b, c*sp.kron(Vd,sp.csr_matrix(Dr)))
                    cc=self.Cent[a,b]
                    if abs(cc)>1e-12: add(n,a,n,b, cc*sp.kron(Vd,sp.diags(1/self.r)))
                    # electrostatic radial force: acceleration a_r = -(e/m) E_r ;  a_r (Fr_v d/dv + Fr_a / v)
                    fv,fa=self.Fr_v[a,b],self.Fr_a[a,b]
                    if abs(fv)>1e-12: add(n,a,n,b, fv*sp.kron(self.Dv,sp.diags(-(e/me)*Er)))
                    if abs(fa)>1e-12: add(n,a,n,b, fa*sp.kron(Vinv,sp.diags(-(e/me)*Er)))
        # ---- RF fields: convolution in n
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
                        # magnetic: acceleration = -(e/m) v x B z = (e/m) B (z x v)  -> operator (e/m)B * Rot_z (angular, no 1/v: (z x v).grad_v is dimensionless*... (z x v).grad_v f : scale v * (1/v) -> pure angular)
                        rz=self.Rot_z[a,b]
                        if abs(rz)>1e-12: add(n,a,mm,b, rz*sp.kron(Inu,sp.diags((e/me)*Bq)))
        rows=[]
        for n1 in range(-Nh,Nh+1):
            for a1 in range(nlm):
                row=[]
                for n2 in range(-Nh,Nh+1):
                    for a2 in range(nlm): row.append(blocks.get((n1,a1,n2,a2),None))
                rows.append(row)
        A=sp.bmat(rows,format='lil')
        # ---- boundary at wall (k=0): specular reflection: odd-parity components vanish for electrons that cannot escape; escaping ones (u > e*phi_w) removed... (kept specular for all here; loss handled via loss term below)
        bc=[]
        for n in range(-Nh,Nh+1):
            for a in range(nlm):
                if self.parity[a]<0:
                    for i in range(nu):
                        r=self.idx(n,a,i,0); A[r,:]=0; A[r,r]=1; bc.append(r)
        return A,bc
    # ---------- moments
    def density(self,f):      # f: (H,nlm,nu,m) -> n(r)/n0 from l=0,n=0: int f sqrt(u) du * sqrt(4pi)?? Y00 = 1/sqrt(4pi): n = int f00 Y00 d^3v... we define f00 so that density = sum_i w_i f00_i
        return np.real(np.einsum('i,ik->k',self.w,f[self.Nh,0]))
    def mean_energy(self,f):  return np.real(np.einsum('i,ik->k',self.w*self.uc,f[self.Nh,0]))/np.maximum(self.density(f),1e-30)
    def nu_iz(self,f):        return np.real(np.einsum('i,ik->k',self.w*self.K.nu_iz_c,f[self.Nh,0]))/np.maximum(self.density(f),1e-30)
    def current_theta(self,f,n=1):
        """j_theta harmonic n per unit n0: -e int v_theta f d^3v ; v_theta = v * (component of vhat along th_hat) -> uses Y_1,? component: <v_th/v> projected"""
        # <Y_a | vhat_th> for a: integrate vhat_th * Y_a over sphere = coefficient c_a ; then j = -e sum_a c_a int v f_a sqrt(u) du
        vth=self.ang.unit_vectors()[0][1]; c=np.einsum('aij,ij,ij->a',self.ang.Y,vth,self.ang.W)
        return -e*np.einsum('a,i,aik->k',c,self.w*self.v,f[self.Nh+n])

# ----------------------------------------------------------------- self-consistent discharge
class Discharge:
    """Kinetic electrons (CylKinetics) + Maxwell (E_theta,B_z) + ions (fluid, mobility-limited with Bohm edge) + quasineutral potential.
    Global ionization balance fixes the absolute density n0 for a given coil current amplitude (B_ext)."""
    def __init__(self,kin,B_ext,n0_guess,Ti_eV=0.03,sigma_i=1e-18):
        self.k=kin; self.B_ext=B_ext; self.n0=n0_guess; self.M=39.95*co.atomic_mass
        self.Ti=Ti_eV; self.ng=kin.ng; self.lam_i=1/(self.ng*sigma_i); self.mu_i=co.e*self.lam_i/(self.M*np.sqrt(8*co.e*Ti_eV/(np.pi*self.M)))  # crude mobility
    # ----- Maxwell: given j_theta harmonics (A/m^2 per unit n0 * n0), solve for E_theta,B_z harmonics n=+-1 (and higher if present)
    def maxwell(self,sigma,jextra=None):
        """Solve Maxwell for harmonic n=+-1 with a LOCAL effective conductivity sigma(r) (complex, S/m) taken from the kinetic
        current (sigma = j_kin / E_prev), so that the coupled E-B problem is solved implicitly (stable for delta << R):
           (1/r)(r E)' = -i w B ,  B' = -mu0 sigma E  (+ -mu0 jextra, nonlocal remainder) ,  B(R)=Bext.
        Written as 2m x 2m linear system on nodes with parity-adapted derivatives (E odd, B even)."""
        k=self.k; H,Nh,m=k.H,k.Nh,k.m; R=k.R; w=k.omega; mu0=co.mu_0
        E=np.zeros((H,m),complex); B=np.zeros((H,m),complex)
        Dodd=k.grid['Dodd']/R; Deven=k.grid['Deven']/R; r=k.r
        for n in (-1,1):
            A=np.zeros((2*m,2*m),complex); rhs=np.zeros(2*m,complex)
            # rows 0..m-1: r E' + E + i n w r B = 0
            A[:m,:m]=np.diag(r)@Dodd+np.eye(m); A[:m,m:]=1j*n*w*np.diag(r)
            # rows m..2m-1: B' + mu0 sigma E = -mu0 jextra ; wall row: B(R)=Bext
            A[m:,m:]=Deven; A[m:,:m]=mu0*np.diag(sigma)
            if jextra is not None: rhs[m:]=-mu0*jextra[n+Nh]
            A[m,:]=0; A[m,m]=1; rhs[m]=self.B_ext[n+Nh]
            x=np.linalg.solve(A,rhs); E[n+Nh]=x[:m]; B[n+Nh]=x[m:]
        return E,B
    def ions(self,ne,nuiz,phi,nr=200):
        """Steady ion continuity on a uniform r-grid (finite volumes, Scharfetter-Gummel for drift-diffusion):
        (1/r) d/dr [ r Gamma ] = S,  Gamma = -D_i n' + mu_i n E_r,  E_r=-dphi/dr ; wall: Gamma(R)= n(R) u_B (Bohm). Returns n_i at Chebyshev nodes."""
        k=self.k; R=k.R; Xc=k.X[::-1]; 
        rr=np.linspace(0,R,nr+1); rc=0.5*(rr[1:]+rr[:-1]); dr=rr[1]-rr[0]
        phi_u=np.interp(rc,Xc*R,phi[::-1]); S=np.interp(rc,Xc*R,(nuiz*ne)[::-1])
        Er_b=-np.gradient(np.interp(rr,Xc*R,phi[::-1]),rr)              # E_r at boundaries
        D=self.mu_i*self.Ti; mu=self.mu_i; uB=np.sqrt(co.e*self.Te_wall/self.M)
        A=np.zeros((nr,nr)); rhs=S*rc*dr
        for j in range(1,nr):                      # interior boundaries j between cells j-1, j
            W=mu*Er_b[j]; z=W*dr/D; ez=np.exp(np.clip(z,-50,50))
            if abs(z)<1e-8: cL,cR=0.5*W+D/dr,0.5*W-D/dr
            else: cL=W*ez/(ez-1); cR=-W/(ez-1)     # Gamma_j = cL n_{j-1} + cR n_j
            fac=rr[j]
            A[j-1,j-1]+=fac*cL; A[j-1,j]+=fac*cR   # outflow from cell j-1
            A[j,j-1]-=fac*cL; A[j,j]-=fac*cR       # inflow to cell j
        # wall boundary: Gamma_wall = uB * n_last (Bohm)
        A[nr-1,nr-1]+=rr[nr]*uB
        n=np.linalg.solve(A,rhs)
        n=np.maximum(n,0); return np.interp(k.X*R,rc,n)
    # ----- electrons at given fields: solve kinetic matrix with normalisation sum_k wdisk_k n_e(k) = 1
    def electrons(self,E,B,phi):
        k=self.k; A,bc=k.kinetic_matrix(E,B,phi); nTot=A.shape[0]
        nrow=k.idx(0,0,k.nu//3,k.m//2); A[nrow,:]=0
        for kk in range(k.m):
            for i in range(k.nu): A[nrow,k.idx(0,0,i,kk)]=k.w[i]*k.wdisk[kk]
        b=np.zeros(nTot,complex); b[nrow]=1.0
        x=spla.spsolve(A.tocsc(),b); return x.reshape(k.H,k.nlm,k.nu,k.m)
    def ambipolar(self,ne_shape,Te,nuiz,nr=200):
        """Ambipolar ion transport: Gamma = -D_a n', D_a = mu_i (Te + Ti) (Te = local kinetic temperature of electrons),
        (1/r)(r Gamma)' = nuiz * n, Bohm at wall. Solves the eigen-like problem for the profile n(r) given nuiz(r) and Te(r) from kinetics,
        with the ionization frequency scaled by a factor lam so that a solution exists: returns (n_profile, lam) where lam=1 means
        exact balance (self-sustained). phi follows from the electron response (updated in run)."""
        k=self.k; R=k.R; Xc=k.X[::-1]
        rr=np.linspace(0,R,nr+1); rc=0.5*(rr[1:]+rr[:-1]); dr=rr[1]-rr[0]
        Te_u=np.interp(rc,Xc*R,Te[::-1]); nu_u=np.interp(rc,Xc*R,nuiz[::-1])
        Da_b=self.mu_i*(np.interp(rr,Xc*R,Te[::-1])+self.Ti)
        uB=np.sqrt(co.e*max(Te[0],0.3)/self.M)
        # generalized eigenproblem: L n = lam * diag(nu) n , L = -(1/(r dr)) [ r Da n' ] + Bohm BC
        L=np.zeros((nr,nr))
        for j in range(1,nr):
            c=rr[j]*Da_b[j]/dr
            L[j-1,j-1]+=c; L[j-1,j]-=c; L[j,j-1]-=c; L[j,j]+=c
        L[nr-1,nr-1]+=rr[nr]*uB
        L=L/(rc*dr)[:,None]
        from scipy.linalg import eig
        wv,V=eig(L,np.diag(np.maximum(nu_u,1e-30)))
        wv=np.real(wv); ok=np.isfinite(wv)&(wv>0); i=np.argmin(np.where(ok,wv,np.inf)); lam=wv[i]; n=np.real(V[:,i]); n=n*np.sign(n[0])
        n=np.maximum(n,0); n=n/np.max(n)
        return np.interp(k.X*R,rc,n), lam
    def run(self,iters=12,relax=0.6,verbose=True,phi_relax=0.5,phi_max=40.):
        """Picard at FIXED n0: fields <-> electrons <-> ambipolar ions/potential. balance = 1/lam (ionization / required)."""
        k=self.k; m=k.m; H,Nh=k.H,k.Nh
        E=np.zeros((H,m),complex); B=np.zeros((H,m),complex)
        for n in (-1,1): B[n+Nh]=self.B_ext[n+Nh]; E[n+Nh]=-1j*n*k.omega*k.r*self.B_ext[n+Nh]/2
        phi=np.zeros(m); self.Te_wall=3.0; hist=[]; bal=np.nan
        for it in range(iters):
            f=self.electrons(E,B,phi)
            ne=np.maximum(k.density(f),1e-6); Te=np.clip(k.mean_energy(f)*2/3,0.03,30.); nuiz=np.maximum(k.nu_iz(f),0)
            j1=k.current_theta(f,1)*self.n0
            sigma=j1/np.where(np.abs(E[Nh+1])>1e-12,E[Nh+1],1e-12)
            sigma=np.where(np.abs(E[Nh+1])>1e-3*np.max(np.abs(E[Nh+1])),sigma,sigma[0])
            Enew,Bnew=self.maxwell(sigma)
            self.Te_wall=max(Te[0],0.5)
            ni,lam=self.ambipolar(ne,Te,nuiz); ni=np.maximum(ni,1e-4); ni=ni/(k.wdisk@ni)
            dphi=np.clip(Te*np.log(ni/ne),-2.0,2.0); dphi-=dphi[0]
            phi=np.clip(phi+phi_relax*dphi,-phi_max,phi_max)
            bal=1.0/lam
            dE=float(np.max(np.abs(Enew-E))/max(np.max(np.abs(Enew)),1e-30))
            E=(1-relax)*E+relax*Enew; B=(1-relax)*B+relax*Bnew
            hist.append((it,dE,bal,Te[0],Te[-1],float(np.max(np.abs(dphi)))))
            if verbose: print(f'  it{it}: dE={dE:.3f} dphi={np.max(np.abs(dphi)):.2f} Te wall/axis {Te[0]:.2f}/{Te[-1]:.2f} eV, nuiz wall/axis {nuiz[0]:.2e}/{nuiz[-1]:.2e}, |E|wall {abs(E[Nh+1][0])*2:.0f} V/m, ne axis/wall {ne[-1]/ne[0]:.2f}, ni axis/wall {ni[-1]/ni[0]:.2f}, phi axis {phi[-1]:.2f} V, balance {bal:.3f}',flush=True)
            if it>3 and dE<2e-3 and np.max(np.abs(dphi))<0.05: break
        self.f,self.E,self.B,self.phi,self.ni,self.ne,self.Te,self.nuiz,self.hist,self.balance=f,E,B,phi,ni,ne,Te,nuiz,hist,bal
        return self
