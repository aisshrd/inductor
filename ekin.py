"""
ekin.py -- electron kinetics on an ENERGY GRID x LEGENDRE harmonics x TIME harmonics,
with real argon cross sections (LXCat format, IST-Lisbon/Phelps/Khakoo set).

Unknown: f_{n,l}(eps_i): time harmonic n (-N..N), Legendre index l (0..L), energy node i.
3D velocity space (sigma-type expansion f(v,theta,t) = sum_l f_l(v,t) P_l(cos theta)), theta measured from the field axis.
Normalisation: int f_0 sqrt(eps) d eps = 1  (f_l are EEDF-type: density of states included separately).

Equation (uniform field E(t) = E0 cos(omega t) along z, in eV units for energy):
  d f_l/dt  - (e E / m) [ l/(2l-1) ( d/dv + (l+1)/v ) f_{l-1} ... ]   (standard Allis expansion, see Hagelaar&Pitchford 2005 for l<=1,
                                                                        Loffhagen&Winkler for multi-term)
We implement the multi-term form in energy variable u = eps (eV), v = sqrt(2 e u/m):
  field operator  F_l[f] = -(eE/m) * { l/(2l-1) * v^{l-1} d/dv [ v^{-(l-1)} f_{l-1} ]   + (l+1)/(2l+3) * v^{-(l+2)} d/dv [ v^{l+2} f_{l+1} ] }
  Elastic (Lorentz) for l>=1: -nu_m(u) f_l ;  for l=0: energy-diffusion form with 2m/M (continuous)
  Inelastic: -nu_k(u) f_l + (for l=0) (u+u_k)/u * nu_k(u+u_k) f_0(u+u_k) * sqrt-weights (particle-conserving, BOLSIG+ form)
  Ionization: loss at u, two electrons share (u - u_iz) equally (BOLSIG+ "equal sharing")  -- growth ignored in this 0D
Discretisation: conservative finite volumes in u for the l=0 energy flux; upwind-free centred for l>=1 (they are damped by nu_m).
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
from bolos import parser

e=co.e; me=co.m_e; kB=co.k
GAMMA=np.sqrt(2*e/me)       # v = GAMMA sqrt(u[eV])

def load_argon(path):
    with open(path) as fp: procs=parser.parse(fp)
    out=dict(elastic=None,inel=[],iz=None,mM=None)
    for p in procs:
        d=np.array(p['data'])
        if p['kind']=='ELASTIC':
            out['elastic']=d; out['mM']=p.get('mass_ratio',1.371e-5)
        elif p['kind']=='EXCITATION':
            out['inel'].append((p['threshold'],d,p['product']))
        elif p['kind']=='IONIZATION':
            out['iz']=(p['threshold'],d)
    return out

class EnergyGridKinetics:
    def __init__(self, xs, umax=80., nu=400, L=1, Nh=0, ng=3.2e21, Tg=300., quadratic=True):
        self.xs=xs; self.L=L; self.Nh=Nh; self.H=2*Nh+1; self.ng=ng; self.kTg=kB*Tg/e
        # cell edges / centres (quadratic grid like BOLOS)
        s=np.linspace(0,1,nu+1); self.ub=umax*(s**2 if quadratic else s); self.uc=0.5*(self.ub[1:]+self.ub[:-1]); self.du=np.diff(self.ub)
        self.nu=nu; u=self.uc
        # cross sections at centres and boundaries
        sig=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1] if d[0,0]>0 else d[0,1],right=d[-1,1])
        self.sig_m_c=sig(xs['elastic'],u); self.sig_m_b=sig(xs['elastic'],self.ub)
        self.nu_m_c=ng*self.sig_m_c*GAMMA*np.sqrt(u)          # momentum transfer frequency at centres
        self.inel=[(th,ng*sig(d,u)*GAMMA*np.sqrt(u)*(u>th),name) for th,d,name in xs['inel']]
        thi,di=xs['iz']; self.iz_th=thi; self.nu_iz_c=ng*sig(di,u)*GAMMA*np.sqrt(u)*(u>thi)
        self.nu_tot_inel=sum(nu for _,nu,_ in self.inel)+self.nu_iz_c
        self.mM=xs['mM']
        self.w=np.sqrt(u)*self.du                               # integration weight for density
        self._build_static()

    # ---------- static operators
    def _build_static(self):
        nu,u,ub,du=self.nu,self.uc,self.ub,self.du; L=self.L
        I=sp.identity(nu,format='csr')
        # elastic energy loss for l=0 (conservative flux): d/du [ 2m/M * u^{3/2} nu_m(u)/sqrt(u)... ] standard BOLSIG+ form in u:
        #   J_el = -2(m/M) u^2 sigma_m N GAMMA ( f + kTg df/du )  (flux of f0 across u-boundary), with f normalised with sqrt(u)
        # Using Hagelaar-Pitchford: d/du[ W f + D df/du ] with W = -3 a (m/M) ... we use their eqs. (18)-(21) without the field part:
        a=2*self.mM; W=-a*ub**2*self.sig_m_b*self.ng*GAMMA; D=a*self.kTg*ub**2*self.sig_m_b*self.ng*GAMMA
        self.Cel0=self._flux_operator(W,D)   # HP2005 eq.(13): sqrt(u) df0/dt = d/du[W f0 + D df0/du]
        # inelastic: loss at u, gain from u+u_k (particle conserving with sqrt(u) measure): (sqrt(u+uk)/sqrt(u)) nu_k(u+uk) f0(u+uk)
        Cin=sp.lil_matrix((nu,nu))
        for th,nuk,_ in self.inel:
            Cin-= sp.diags(nuk)
            Cin+= self._shift_gain(th,nuk)
        # ionization: loss; gain: equal sharing -> both electrons at (u - u_iz)/2 ; particle gain ignored for growth (source counted separately)
        Cin-= sp.diags(self.nu_iz_c)
        Cin+= 2*self._shift_gain_scaled(self.iz_th,self.nu_iz_c,share=0.5)
        self.Cin0=Cin.tocsr()
        self.Closs_l=sp.diags(self.nu_tot_inel)                 # for l>=1: only loss
        self.Cm=sp.diags(self.nu_m_c)                           # momentum transfer for l>=1
        # field operators between l and l+-1 (energy form); per unit (eE/m)/... we build in terms of u with v=GAMMA sqrt(u)
        # Using velocity-space Allis form, converted: d/dv = 2 sqrt(u)/GAMMA d/du * ... implemented by centred differences on uc.
        self.Fdown=[None]*(L+1); self.Fup=[None]*(L+1)         # Fdown[l]: acts on f_{l-1}; Fup[l]: acts on f_{l+1}
        v=GAMMA*np.sqrt(u)
        Dv=self._dv_matrix(v)                                   # d/dv centred, nu x nu
        for l in range(L+1):
            if l>=1:
                A=sp.diags(v**(l-1))@Dv@sp.diags(v**(-(l-1)))
                self.Fdown[l]=(l/(2*l-1))*A
            if l<L:
                B=sp.diags(v**(-(l+2)))@Dv@sp.diags(v**(l+2))
                self.Fup[l]=((l+1)/(2*l+3))*B
        self.Dv=Dv

    def _dv_matrix(self,v):
        n=len(v); M=sp.lil_matrix((n,n))
        for i in range(n):
            if 0<i<n-1:
                h1=v[i]-v[i-1]; h2=v[i+1]-v[i]
                M[i,i-1]=-h2/(h1*(h1+h2)); M[i,i]=(h2-h1)/(h1*h2); M[i,i+1]=h1/(h2*(h1+h2))
            elif i==0:
                M[i,i]=-1/(v[1]-v[0]); M[i,i+1]=1/(v[1]-v[0])
            else:
                M[i,i-1]=-1/(v[i]-v[i-1]); M[i,i]=1/(v[i]-v[i-1])
        return M.tocsr()

    def _flux_operator(self,W,D):
        """Scharfetter-Gummel (exponential) scheme for d/du[ W F + D dF/du ], F = sqrt(u) f on cell centres.
        Returns the operator acting on f (BOLSIG+ eq. for F; converted with F = sqrt(u) f)."""
        nu,ub,uc,du=self.nu,self.ub,self.uc,self.du
        M=sp.lil_matrix((nu,nu))
        for j in range(1,nu):                       # boundary j between cells j-1 and j
            h=uc[j]-uc[j-1]; Wj,Dj=W[j],D[j]
            if abs(Dj)<1e-300:
                # pure drift (Tg=0): upwind; W<0 means flux towards lower u
                cL,cR=(Wj,0.0) if Wj>0 else (0.0,Wj)
            else:
                z=Wj*h/Dj
                if abs(z)<1e-8: cL,cR=0.5*Wj-Dj/h,0.5*Wj+Dj/h
                else:
                    ez=np.exp(z)
                    cL=Wj*ez/(ez-1); cR=-Wj/(ez-1)   # flux = cL F_{j-1} + cR F_j ;  W F + D dF/du with F exponential between nodes
            # flux_j enters cell j (+) and leaves cell j-1 (-): dF/dt = -(flux_{j+1}-flux_j)/du  -> here operator = -div
            M[j,j-1]+=cL/du[j]; M[j,j]+=cR/du[j]
            M[j-1,j-1]-=cL/du[j-1]; M[j-1,j]-=cR/du[j-1]
        Op=M.tocsr()
        return sp.diags(1/np.sqrt(uc))@Op

    def _shift_gain(self,th,nuk):
        """gain term G f0 at u from u'=u+th: (sqrt(u')/sqrt(u)) nuk(u') f0(u') with linear interpolation; conserves sum w_i f_i."""
        nu,uc=self.nu,self.uc; M=sp.lil_matrix((nu,nu))
        for i in range(nu):
            up=uc[i]+th
            if up>uc[-1]: continue
            j=np.searchsorted(uc,up)-1; j=max(0,min(j,nu-2)); t=(up-uc[j])/(uc[j+1]-uc[j])
            nk=np.interp(up,uc,nuk)
            M[i,j]+=(1-t)*nk*np.sqrt(up)/np.sqrt(uc[i]); M[i,j+1]+=t*nk*np.sqrt(up)/np.sqrt(uc[i])
        # enforce exact particle conservation per column (scale columns so that sum_i w_i G_ij = w_j nuk_j)
        M=M.tocsc(); col=np.asarray(M.T@self.w).ravel(); target=self.w*nuk
        s=np.where(col>0,target/np.maximum(col,1e-300),0.0); return (M@sp.diags(s)).tocsr()

    def _shift_gain_scaled(self,th,nuk,share=0.5):
        """ionization: secondary+primary at u = share*(u'-th): gain at u from u' = u/share + th"""
        nu,uc=self.nu,self.uc; M=sp.lil_matrix((nu,nu))
        for i in range(nu):
            up=uc[i]/share+th
            if up>uc[-1]: continue
            j=np.searchsorted(uc,up)-1; j=max(0,min(j,nu-2)); t=(up-uc[j])/(uc[j+1]-uc[j])
            nk=np.interp(up,uc,nuk)
            M[i,j]+=(1-t)*nk; M[i,j+1]+=t*nk
        M=M.tocsc(); col=np.asarray(M.T@self.w).ravel(); target=self.w*nuk*0.5   # each of the two gain terms carries half -> times 2 outside = total 1 (particle gain ignored => treat as 1 out, 1 in)
        s=np.where(col>0,target/np.maximum(col,1e-300),0.0); return (M@sp.diags(s)).tocsr()

    # ---------- assemble for field amplitude E0 (V/m), angular frequency omega; DC if omega=0 (then Nh=0)
    def assemble(self,E0,omega=0.0):
        L,Nh,H,nu=self.L,self.Nh,self.H,self.nu
        a=e*E0/me                                               # acceleration amplitude
        nb=(L+1)*nu
        # block index: (n, l, i) -> ((n+Nh)*(L+1)+l)*nu+i
        blocks={}
        def add(n1,l1,n2,l2,M):
            key=(n1,l1,n2,l2); blocks[key]=blocks[key]+M if key in blocks else M
        I=sp.identity(nu)
        for n in range(-Nh,Nh+1):
            for l in range(L+1):
                add(n,l,n,l, 1j*n*omega*I)
                if l==0: add(n,l,n,l, -(self.Cel0+self.Cin0))
                else:    add(n,l,n,l, self.Cm+self.Closs_l)
                # field: E(t)=E0 cos(wt) -> harmonics +-1 (DC: n=0 only, factor 1)
                for q,cq in ([(0,1.0)] if omega==0 else [(1,0.5),(-1,0.5)]):
                    m=n-q
                    if abs(m)>Nh: continue
                    if l>=1: add(n,l,m,l-1, -a*cq*self.Fdown[l])
                    if l<L:  add(n,l,m,l+1, -a*cq*self.Fup[l])
        rows=[]
        for n1 in range(-Nh,Nh+1):
            for l1 in range(L+1):
                row=[]
                for n2 in range(-Nh,Nh+1):
                    for l2 in range(L+1):
                        row.append(blocks.get((n1,l1,n2,l2),None))
                rows.append(row)
        return sp.bmat(rows,format='lil')

    def solve(self,E0,omega=0.0):
        A=self.assemble(E0,omega); L,Nh,nu=self.L,self.Nh,self.nu
        nrow=((0+Nh)*(L+1)+0)*nu + nu//3     # replace one l=0,n=0 equation by normalisation
        A[nrow,:]=0
        for i in range(nu): A[nrow,((Nh)*(L+1))*nu+i]=self.w[i]
        b=np.zeros(A.shape[0],dtype=complex); b[nrow]=1.0
        x=spla.spsolve(A.tocsc(),b)
        f=x.reshape(2*Nh+1,L+1,nu)
        return f

    # ---------- moments
    def mean_energy(self,f):      return float(np.real(np.sum(self.w*self.uc*f[self.Nh,0])))
    def rate_iz(self,f):          return float(np.real(np.sum(self.w*self.nu_iz_c*f[self.Nh,0])))
    def rate_exc_total(self,f):   return float(np.real(np.sum(self.w*(self.nu_tot_inel-self.nu_iz_c)*f[self.Nh,0])))
    def drift(self,f,harm=0):
        """drift velocity (DC) or complex amplitude of oscillating drift for harmonic 'harm' (m/s): <v_z> = (1/3) int v f1 sqrt(u) du"""
        v=GAMMA*np.sqrt(self.uc); return complex(np.sum(self.w*v*f[self.Nh+harm,1]))/3
    def eedf(self,f):             return np.real(f[self.Nh,0])

# ---------------------------------------------------------------- electron-electron collisions (Fokker-Planck in energy, isotropic part)
def ee_operator(K, ne, lnL=10.0):
    """[EXPERIMENTAL - NOT VALIDATED: Maxwellian not exactly stationary on the discrete grid, absolute constant unverified; do not use in production]
    Hagelaar & Pitchford 2005, eqs (26)-(27), for normalised F0 (int F0 sqrt(u) du = 1), energies in eV:
         sqrt(u) dF0/dt = d/du [ a ( 3 A1 F0 + 2 (A2 + u^{3/2} A3) dF0/du ) ]
         A1 = int_0^u sqrt(u') F0 du' ,  A2 = int_0^u u'^{3/2} F0 du' ,  A3 = int_u^inf F0 du'
         a = (e^2/(4 pi eps0))^2 * (2 pi / m^2) * ne * lnL  ->  in (eV, s): a = 2 pi ne lnL (e/(4 pi eps0))^2 / (e^{1/2} ... )
       We fix the constant by the exact result that a Maxwellian is stationary (fixes the W/D ratio automatically, since both
       carry the same 'a') and by the NRL e-e relaxation rate nu_ee = 2.91e-6 ne[cm-3] lnL Te^-3/2 [1/s] for the magnitude.
       The analytic constant in HP2005 units is  a = gamma * (e/(4 pi eps0))^2 * 2 pi * ne * lnL / e   with gamma = sqrt(2e/m) [m/s/V^1/2],
       which we use directly (checked against NRL within the expected O(1) definitional factor)."""
    import numpy as np, scipy.constants as co
    u=K.uc; w=K.w
    gamma=GAMMA
    a=gamma*(co.e/(4*np.pi*co.epsilon_0))**2*2*np.pi*ne*lnL/co.e         # eV^{3/2}... units: (m/s V^-1/2)(V m)^2 m^-3 / V = m^0 V^{1/2} s^-1 -> eV^{1/2}/s ; times (3 A1 ~ 1)*F0 ~ eV^-3/2 -> d/du of eV^-1/s ok
    def build(f0):
        f0=np.maximum(np.real(f0),0)
        # cumulative integrals evaluated exactly at cell EDGES (consistent with cell weights): A(ub_j) = sum_{i<j} w_i f_i
        c1=np.concatenate([[0],np.cumsum(w*f0)]); c2=np.concatenate([[0],np.cumsum(w*u*f0)]); c3=np.concatenate([[0],np.cumsum(K.du*f0)])
        A1=c1; A2=c2; A3=c3[-1]-c3
        We=-a*3*A1; De=a*2*(A2+K.ub**1.5*A3)
        return K._flux_operator(We,De)      # same conservative Scharfetter-Gummel form as the elastic operator: (1/sqrt u) d/du[W f + D f']
    return build
