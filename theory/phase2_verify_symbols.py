"""Exact shear, metric, energy and entropy identities used in Phase 2."""
import json
import sympy as s


def main():
    h, a, b, c, d1, d2 = s.symbols("h a b c d1 d2", real=True)
    I, Z = s.eye(2), s.zeros(2)
    V=s.diag(d1,d2)
    J=Z.row_join(I).col_join((-I).row_join(Z))
    drift=I.row_join(h*V/2).col_join(Z.row_join(I))
    kick=I.row_join(Z).col_join((-h*s.Matrix([[a,b],[b,c]])).row_join(I))
    jac=drift*kick*drift
    q=s.Matrix(s.symbols("x:2"))
    p=s.Matrix(s.symbols("p:2"))
    g=s.Matrix(s.symbols("g:2"))
    mid=q+h*V*p/2
    endp=p-h*g
    endq=mid+h*V*endp/2
    rmid=endq-h*V*endp/2
    rendp=-endp-h*g
    rendq=rmid+h*V*rendp/2
    x,y=q
    e,sigma=s.symbols("e sigma")
    ax,ay,w1,w2=s.symbols("ax ay w1 w2")
    A=(w1*(x-ax)**2+w2*(y-ay)**2)/2
    u=(a*x*x+2*b*x*y+c*y*y)/2
    z=q.col_join(p).col_join(s.Matrix([e,sigma]))
    def grad(expr):
        return s.Matrix([expr]).jacobian(z).T
    ga=grad(A)[:2,:]
    gu=grad(u)[:2,:]
    tangent=s.eye(4).col_join(s.zeros(1,4)).col_join(ga.T.row_join(s.zeros(1,2)))
    L=tangent*J*tangent.T
    B=I.col_join(Z).col_join(-gu.T).col_join(s.zeros(1,2))
    M=B*B.T
    E=u+(p.T*V*p)[0]/2+e
    S=sigma-A
    flux=M*grad(S)
    jacobi=s.Matrix([sum(L[i,l]*s.diff(L[j,k],z[l])+L[j,l]*s.diff(L[k,i],z[l])
                          +L[k,l]*s.diff(L[i,j],z[l]) for l in range(6))
                     for i in range(6) for j in range(i+1,6) for k in range(j+1,6)])
    D,KL,KH=s.symbols("D KL KH",positive=True)
    tests={"mass_DKD_symplectic":jac.T*J*jac-J,
           "same_reverse_midpoint":rmid-mid,"reflected_inverse":(rendq-q).col_join(-rendp-p),
           "poisson_antisymmetric":L+L.T,"poisson_Jacobi":jacobi,
           "entropy_Casimir":L*grad(S),"metric_energy_degeneracy":M*grad(E),
           "mechanical_projection":(L*grad(E))[:4,:]-(V*p).col_join(-gu),
           "anchor_velocity":flux[:2,:]+ga,
           "entropy_production":grad(S).T*flux-ga.T*ga,
           "matched_remove_low":s.Matrix([KL+KH-(KL*(1-D/KL)+KH)-D]),
           "matched_remove_high":s.Matrix([KL+KH-(KL+KH*(1-D/KH))-D])}
    checks={k:all(s.expand(item)==0 for item in v) for k,v in tests.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps({"sympy":s.__version__,"exact_identities":checks},indent=2))


if __name__=="__main__":
    main()
