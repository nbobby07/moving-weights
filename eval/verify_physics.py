"""Small independent black-box probes for the code actually passed to the suite."""
import torch

from .common import PHASE, l2, run, unwind


def phase_probe(factory, name, q, p, f):
    h, gamma = 0.17, 1.3
    opt = factory(name, h=h, fisher=f.clone())
    reported = opt.brake_diag(q)
    if reported.shape != q.shape or reported.device != q.device or reported.dtype != q.dtype:
        raise ValueError("brake_diag(q) must match q")
    mask = torch.ones_like(q) if name != "fisher" else (f <= f.median()).to(q)
    gate = 0.0 if name == "glide" else gamma
    d = torch.linalg.matrix_exp(torch.diag(mask) * (-h * gate / 2))
    k = q.new_tensor([[1.0, 0.3, 0.0], [0.3, -0.8, 0.2], [0.0, 0.2, 0.6]])

    def gradient(i):
        return lambda x: k @ x + (i * 5 % 13 - 3) * 0.09

    points = []
    def recording_grad(x):
        points.append(x.clone())
        return gradient(1)(x)

    state = opt.init(q, p)
    qstart, pstart = q.clone(), p.clone()
    first, out = opt.step(q, state, recording_grad, gamma=gate)
    half = d @ pstart - 0.5 * h * gradient(1)(qstart)
    expected = qstart + h * half
    pend = d @ (half - 0.5 * h * gradient(1)(expected))
    step_error = max(l2(first, expected), l2(out[0], pend))
    points_ok = len(points) == 2 and l2(points[0], qstart) == 0 and l2(points[1], expected) < 1e-13
    pure = torch.equal(q, qstart) and torch.equal(p, pstart) and torch.equal(state[0], pstart)

    q1, state = q.clone(), opt.init(q, p)
    for i in range(9):
        q1, state = opt.step(q1, state, gradient(i), gamma=0.0)
    fresh = factory(name, h=h, fisher=f.clone())
    back, (pback,) = unwind(fresh, q1, state, 9, gradient)
    wrong, _ = unwind(fresh, q1, state, 9, lambda i: gradient(8 - i))
    roundtrip = max(l2(q, back), l2(p, pback))
    # Jacobian by centered differences also works for adapters that detach outputs.
    z = torch.cat((q, p))
    def mapping(x):
        u, v = x.chunk(2)
        u, (v,) = opt.step(u, (v,), gradient(2), gamma=0.0)
        return torch.cat((u, v))
    eye = torch.eye(6, device=q.device, dtype=q.dtype)
    jac = torch.stack([(mapping(z + e * 1e-5) - mapping(z - e * 1e-5)) / 2e-5 for e in eye], 1)
    omega = torch.cat((eye[3:], -eye[:3]))
    sympl = l2(jac.T @ omega @ jac, omega)
    # Recover the actual full brake map from zero-force momentum, one basis vector at a time.
    kicks = [opt.step(q, (e,), torch.zeros_like, gamma=gate)[1][0] for e in eye[:3, :3]]
    actual = torch.stack(kicks, 1)
    map_error = l2(actual, d @ d)
    eig_min = float(reported.min())
    wrong_order = l2(wrong, q)
    mask_error = l2(reported, mask)
    checks = (len(out) == 1 and pure and points_ok and step_error < 1e-13 and
              mask_error == 0 and eig_min >= 0 and map_error < 1e-13 and
              roundtrip < 1e-12 and wrong_order > 1e-4 and sympl < 1e-8)
    return {"method": name, "fisher": f.tolist(), "reported_brake": reported.tolist(),
            "min_brake_eigenvalue": eig_min, "mask_l2": mask_error, "brake_map_l2": map_error,
            "split_step_l2": step_error, "gradient_sites_ok": points_ok, "pure": pure,
            "roundtrip_l2": roundtrip, "wrong_batch_order_q_l2": wrong_order,
            "symplectic_form_residual": sympl, "passed": checks}


def baseline_probe(factory, name, q):
    h = 0.017
    opt = factory(name, h=h, fisher=torch.zeros_like(q))
    x = torch.nn.Parameter(q.clone())
    native = (torch.optim.AdamW([x], lr=h, weight_decay=0.01, foreach=False, fused=False)
              if name == "adamw" else torch.optim.SGD([x], lr=h, momentum=0.9, foreach=False))
    state = opt.init(q, torch.zeros_like(q))
    calls = 0
    def grad(v):
        nonlocal calls
        calls += 1
        return v.sin() + i / 7
    errors = []
    for i in range(7):
        x.grad = x.detach().sin() + i / 7
        native.step()
        q, state = opt.step(q, state, grad, gamma=0.0)
        errors.append(l2(q, x.detach()))
    return {"method": name, "native_torch_q_l2": errors, "gradient_calls": calls,
            "passed": max(errors) < 1e-13 and calls == 7}


def evaluate(factory, device):
    q = torch.tensor([0.23, -0.71, 0.35], device=device, dtype=torch.float64)
    p = q.new_tensor([0.51, 0.8, -0.44])
    rows = [phase_probe(factory, name, q, p, q.new_tensor([0.1, 0.4, 4.0])) for name in PHASE]
    rows += [phase_probe(factory, "fisher", q, p, q.new_tensor(f))
             for f in ([0.0, 0.0, 0.0], [3.0, 1.0, 1.0])]
    rows += [baseline_probe(factory, name, q) for name in ("sgdm", "adamw")]
    return {"probes": rows, "checks_passed": all(row["passed"] for row in rows)}


if __name__ == "__main__":
    run("physics", evaluate)
