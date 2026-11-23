"""Independent exact-arithmetic controls for the Phase 3 specification."""
from fractions import Fraction
from itertools import product
import json
import sympy as s


def main():
    h,a,b,c=s.symbols("h a b c")
    I,Z=s.eye(2),s.zeros(2)
    J=Z.row_join(I).col_join((-I).row_join(Z))
    D=I.row_join(h*I/2).col_join(Z.row_join(I))
    G=I.row_join(Z).col_join((-h*s.Matrix([[a,b],[b,c]])).row_join(I))
    jac=D*G*D
    q=s.Matrix(s.symbols("q:2"));p=s.Matrix(s.symbols("p:2"));g=s.Matrix(s.symbols("g:2"))
    mid=q+h*p/2
    pn=p-h*g
    end=mid+h*pn/2
    back_mid=end-h*pn/2
    back_p=-pn-h*g
    inverse=(back_mid+h*back_p/2-q).col_join(-back_p-p)
    checks={"DKD_symplectic":all(s.expand(v)==0 for v in jac.T*J*jac-J),
            "DKD_reflected_inverse":all(s.expand(v)==0 for v in inverse)}
    def integer_step(q,p,index):
        mid=q+round(Fraction(p,4))
        p=p-round(Fraction(mid*mid+index,8))
        return mid+round(Fraction(p,4)),p
    count=0
    for q,p in product(range(-12,13),repeat=2):
        a,b=q,p
        for i in (2,5,3):
            a,b=integer_step(a,b,i)
        b=-b
        for i in (3,5,2):
            a,b=integer_step(a,b,i)
        assert (a,-b)==(q,p)
        count+=1
    checks["integer_rational_ties_even"]=True
    for depth in (1,2,4,8):
        codes=[c for c in product((0,1),repeat=depth) if sum(c)<=1]
        edges={c[:i] for c in codes for i in range(1,depth+1)}
        assert len(codes)==depth+1 and len(edges)==depth+depth*(depth+1)//2
        # A 7/8 majority for bit 1 and bit 0 cannot occur at the same address.
        assert not any(v>=7 and 8-v>=7 for v in range(9))
    checks["single_submission_bound_and_trie_sizes"]=True
    assert all(checks.values())
    print(json.dumps({"sympy":s.__version__,"checks":checks,"exhaustive_lattice_cases":count},indent=2))


if __name__=="__main__":
    main()
