import numpy as np
from scipy.linalg import expm, logm
# basis (D1D1, D1D2, D2D1, D2D2); sigma reconstructed from the spectrum and off-diagonal entry quoted in the old Sec. IV-B
a, s = (0.298+0.087)/2, (0.298-0.087)/2
sig = np.diag([0.308, a, a, 0.308]).astype(float); sig[1,2]=sig[2,1]=s
sig/=np.trace(sig)
H = np.diag([3.0,0.6,1.8,3.4]); lam=0.5
print("sigma offdiag", s, "eig", np.linalg.eigvalsh(sig))
G = logm(sig)-H/lam
rs = expm(G); rs/=np.trace(rs)
w,V = np.linalg.eigh(sig)
# pinching: group eigenvalues
def pinch(X):
    out=np.zeros_like(X)
    done=[]
    for i,wi in enumerate(w):
        if any(abs(wi-wj)<1e-9 for wj in done): continue
        done.append(wi)
        idx=[k for k in range(4) if abs(w[k]-wi)<1e-9]
        P=V[:,idx]@V[:,idx].T
        out+=P@X@P
    return out
PH=pinch(H)
rp = expm(logm(sig)-PH/lam); rp/=np.trace(rp)
vs=-lam*np.log(np.trace(expm(G))); vp=-lam*np.log(np.trace(expm(logm(sig)-PH/lam)))
print("Delta", vp-vs, "cost star", np.trace(H@rs).real, "cost pinched", np.trace(H@rp).real)
det = np.zeros((4,4)); det[1,1]=1   # deterministic product policy R1->D1, R2->D2
print("||[det, sigma]||_F =", np.linalg.norm(det@sig-sig@det))
d = np.real(np.diag(rp)); P=d.reshape(2,2)
print("diag of rho_sigma (joint dist):", d.round(4))
print("product of its marginals      :", np.outer(P.sum(1),P.sum(0)).ravel().round(4))
