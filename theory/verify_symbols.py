"""Exact arithmetic witnesses. Run with python -B theory/verify_symbols.py."""
import json

import sympy as s


def main():
    h, a, b, c, d, e, f, r = s.symbols("h a b c d e f r", real=True)
    q = s.Matrix(s.symbols("q0:2"))
    p = s.Matrix(s.symbols("p0:2"))
    g0 = s.Matrix(s.symbols("u0:2"))
    g1 = s.Matrix(s.symbols("v0:2"))
    half = p - h * g0 / 2
    q1, p1 = q + h * half, half - h * g1 / 2
    halfback = -p1 - h * g1 / 2
    qback, pback = q1 + h * halfback, -(halfback - h * g0 / 2)
    eye, zero = s.eye(2), s.zeros(2)
    omega = zero.row_join(eye).col_join((-eye).row_join(zero))
    def kick(k):
        return eye.row_join(zero).col_join((-h * k / 2).row_join(eye))
    drift = eye.row_join(h * eye).col_join(zero.row_join(eye))
    jac = kick(s.Matrix([[a, b], [b, c]])) * drift * kick(s.Matrix([[d, e], [e, f]]))
    damp = s.diag(1, 1, r, r)
    proj = s.diag(1, 0)
    # A unit-temperature heat reservoir gives the strict metriplectic degeneracies.
    gp = proj * p
    kinetic_block = proj.row_join(-gp).col_join((-gp.T).row_join(p.T * gp))
    factor = proj.col_join(-p.T * proj)
    grad_e = p.col_join(s.ones(1, 1))
    grad_s = s.Matrix([0, 0, 1])
    x, y = q
    u = ((x*x-1)**2 + (y*y-1)**2) / 4 + 3 * (x-y)**2 / 50
    t = p.dot(p) / 2
    z = q.col_join(p)
    def bracket(v, w):
        return (s.Matrix([v]).jacobian(z) * omega * s.Matrix([w]).jacobian(z).T)[0]
    shadow2 = bracket(t, bracket(t, u)) / 12 - bracket(u, bracket(u, t)) / 24
    grad_u = s.Matrix([u]).jacobian(q).T
    tests = {
        "inverse_positions_and_momenta": (qback - q).col_join(pback - p),
        "canonical_form": jac.T * omega * jac - omega,
        "isotropic_half_brake_form": damp.T * omega * damp - r * omega,
        "bath_PSD_factorization": kinetic_block - factor * factor.T,
        "bath_energy_degeneracy": kinetic_block * grad_e,
        "bath_entropy_production": grad_s.T * kinetic_block * grad_s - p.T * proj * p,
        "high_subspace_bracket_null": (eye - proj) * proj,
        "high_direction_glide_counterexample": (eye - proj) * (q + h*p - q) - s.Matrix([0, h*p[1]]),
        "double_well_gradient": grad_u - s.Matrix([x*(x*x-1) + 3*(x-y)/25, y*(y*y-1) + 3*(y-x)/25]),
        "BCH_shadow_coefficient": s.Matrix([shadow2 - (p.T*s.hessian(u,q)*p)[0]/12 + grad_u.dot(grad_u)/24]),
    }
    checks = {name: all(s.expand(v) == 0 for v in entries) for name, entries in tests.items()}
    print(json.dumps({"sympy": s.__version__, "exact_identities": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
