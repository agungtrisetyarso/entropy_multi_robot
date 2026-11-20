# Independent re-derivation of headline numbers, not using fg.py
import numpy as np, itertools
from scipy.optimize import minimize
from scipy.special import logsumexp
# ---- running example
c1=np.array([0.2,1.0]); c2=np.array([0.8,0.4]); W=2*np.eye(2)
C=c1[:,None]+c2[None,:]+W
for lam in [0.5,1.0,2.0,4.0]:
    pi=0.25*np.exp(-C/lam); Z=pi.sum(); pi/=Z; Js=-lam*np.log(Z)
    klJ=(pi*np.log(pi/0.25)).sum(); ecJ=(pi*C).sum()
    def unpack(th):
        a,b=1/(1+np.exp(-th[0])),1/(1+np.exp(-th[1])); return np.array([a,1-a]),np.array([b,1-b])
    def J(th):
        q1,q2=unpack(th); q=np.outer(q1,q2); return (q*C).sum()+lam*(q*np.log(q/0.25)).sum()
    best=min((minimize(J,x0,method='Nelder-Mead',options=dict(xatol=1e-12,fatol=1e-14,maxiter=20000)) for x0 in [(2,-2),(-2,2),(0,0),(3,3),(-3,-3)]),key=lambda r:r.fun)
    gap=best.fun-Js
    # matched: min E[C] over product with KL<=klJ
    cons=dict(type='ineq',fun=lambda th: klJ-(lambda q:(q*np.log(q/0.25)).sum())(np.outer(*unpack(th))))
    bm=min((minimize(lambda th:(np.outer(*unpack(th))*C).sum(),x0,constraints=[cons],method='SLSQP',options=dict(ftol=1e-14,maxiter=500)) for x0 in [(2,-2),(1,-1),(0.5,-0.5),(3,-1),(1,-3)]),key=lambda r:r.fun if r.success else 9)
    q1,q2=unpack(bm.x)
    print("lam %.1f gap %.3f law %.3f | joint EC %.3f conf %.3f | matched EC %.3f conf %.3f prem %.3f" % (lam,gap,1.0/(2*lam),ecJ,pi[0,0]+pi[1,1],bm.fun,q1@q2,bm.fun-ecJ))
    if lam==1.0: print("   pi*",pi.ravel().round(3),"q1",unpack(best.x)[0].round(3),"q2",unpack(best.x)[1].round(3),"matched q1",q1.round(3),"q2",q2.round(3))
# ---- variance law on a random 4-robot, 3-task instance, large lambda
rng=np.random.default_rng(0); n,m=4,3
c=rng.uniform(0,1,(n,m)); E=[(0,1),(1,2),(2,3),(0,3),(0,2)]
Wt={e:rng.uniform(.5,2)*np.eye(m)+rng.normal(0,1,(m,m)) for e in E}
X=np.array(list(itertools.product(range(m),repeat=n)))
Cx=sum(c[i,X[:,i]] for i in range(n))+sum(Wt[e][X[:,e[0]],X[:,e[1]]] for e in E)
def v(Wm):
    Wc=Wm-Wm.mean(1,keepdims=True)-Wm.mean(0,keepdims=True)+Wm.mean(); return (Wc**2).mean()
V=sum(v(Wt[e]) for e in E)
for lam in [16.,64.,256.]:
    Js=-lam*(logsumexp(-Cx/lam)-np.log(len(Cx)))
    q=np.full((n,m),1/m)
    for it in range(20000):
        qo=q.copy()
        for i in range(n):
            f=c[i].copy()
            for (a,b) in E:
                if a==i: f+=Wt[(a,b)]@q[b]
                if b==i: f+=Wt[(a,b)].T@q[a]
            z=-f/lam; q[i]=np.exp(z-logsumexp(z))
        if abs(q-qo).max()<1e-15: break
    logq=sum(np.log(q[i,X[:,i]]) for i in range(n)); qx=np.exp(logq)
    Jp=qx@Cx+lam*(qx@(logq+n*np.log(m)))
    print("lam %5.0f  gap*2lam/V = %.4f" % (lam,(Jp-Js)*2*lam/V))
