"""Independent optimizer contracts and mechanical probes."""

import math
import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mttt.baselines import EWC
from mttt.optimizers import make_optimizer
from mttt.phase import unwind


PHASE_METHODS = ("glide", "strang", "fisher")
DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
DTYPES = (torch.float64, torch.float32)
HORIZONS = (1, 10, 50, 100, 250, 500)


def setUpModule():
    torch.manual_seed(1729)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def l2(actual, expected):
    return torch.linalg.vector_norm(actual.double() - expected.double()).item()


class OptimizerTests(unittest.TestCase):
    def test_split_force_sites_and_immutability(self):
        for device in DEVICES:
            for dtype in DTYPES:
                for name in PHASE_METHODS:
                    with self.subTest(device=device, dtype=dtype, method=name):
                        q = torch.tensor([0.23, -0.71, 0.35], device=device, dtype=dtype)
                        q.requires_grad_()
                        p = q.new_tensor([0.51, 0.8, -0.44], requires_grad=True)
                        fisher = q.new_tensor([0.12, 0.9, 0.12])
                        opt = make_optimizer(name, h=0.17, fisher=fisher)
                        q_start, p_start = q.detach().clone(), p.detach().clone()
                        state = opt.init(q, p)
                        self.assertIsNot(state[0], p)
                        self.assertFalse(state[0].requires_grad)

                        calls = []

                        def grad(value):
                            calls.append(value.detach().clone())
                            with torch.enable_grad():
                                leaf = value.detach().requires_grad_()
                                return torch.autograd.grad(leaf.sin().sum(), leaf)[0]

                        mask = q.new_tensor([1, 0, 1]) if name == "fisher" else torch.ones_like(q)
                        gamma = 1.3
                        gate = 0.0 if name == "glide" else gamma
                        brake = torch.exp(-0.085 * gate * mask)
                        mid = p_start * brake - 0.085 * q_start.cos()
                        expected_q = q_start + 0.17 * mid
                        expected_p = (mid - 0.085 * expected_q.cos()) * brake
                        fisher.fill_(0)

                        end_q, (end_p,) = opt.step(q, state, grad, gamma=gamma)
                        tolerance = 1e-14 if dtype == torch.float64 else 2e-7
                        self.assertEqual(len(calls), 2)
                        torch.testing.assert_close(calls[0], q_start, rtol=0, atol=0)
                        torch.testing.assert_close(calls[1], expected_q, rtol=0, atol=tolerance)
                        for output, expected in ((end_q, expected_q), (end_p, expected_p)):
                            torch.testing.assert_close(output, expected, rtol=0, atol=tolerance)
                            self.assertFalse(output.requires_grad)
                        torch.testing.assert_close(q, q_start, rtol=0, atol=0)
                        torch.testing.assert_close(p, p_start, rtol=0, atol=0)
                        torch.testing.assert_close(state[0], p_start, rtol=0, atol=0)

    def test_ordered_checkpoint_free_rollback(self):
        for device in DEVICES:
            for dtype in DTYPES:
                for name in PHASE_METHODS:
                    with self.subTest(device=device, dtype=dtype, method=name):
                        q0 = torch.tensor([0.73, -0.91], device=device, dtype=dtype)
                        p0 = q0.new_tensor([0.42, -0.25])
                        fisher = q0.new_tensor([1, 4])
                        spring = q0.new_tensor([1.4, 0.75])
                        opt = make_optimizer(name, h=0.075, fisher=fisher)

                        def gradient(index):
                            tilt = ((19 * index) % 37 - 18) / 50
                            return lambda q: spring * q + tilt

                        q, state = q0, opt.init(q0, p0)
                        for step in range(1, HORIZONS[-1] + 1):
                            q, state = opt.step(q, state, gradient(step - 1), gamma=0.0)
                            if step not in HORIZONS:
                                continue
                            fresh = make_optimizer(name, h=0.075, fisher=fisher)
                            replay_order = []

                            def record(index):
                                replay_order.append(index)
                                return gradient(index)

                            back_q, (back_p,) = unwind(fresh, q, state, step, record)
                            self.assertEqual(replay_order, list(range(step - 1, -1, -1)))
                            tolerance = 1e-11 if dtype == torch.float64 else 2e-5
                            self.assertLess(l2(back_q, q0), tolerance, f"position at step {step}")
                            self.assertLess(l2(back_p, p0), tolerance, f"momentum at step {step}")

                        wrong_q, _ = unwind(fresh, q, state, step, lambda i: gradient(step - 1 - i))
                        self.assertGreater(l2(wrong_q, q0), 1e-4)

    def test_brake_geometry(self):
        for device in DEVICES:
            with self.subTest(device=device):
                q = torch.zeros(4, device=device, dtype=torch.float64)
                p = q.new_tensor([0.4, -0.3, 0.2, -0.5])
                h, gamma = 0.17, 1.3
                fisher = q.new_tensor([6, 2, 2, 1])
                opt = make_optimizer("fisher", h=h, fisher=fisher)
                diag = opt.brake_diag(q)
                torch.testing.assert_close(diag, q.new_tensor([0, 1, 1, 1]), rtol=0, atol=0)
                self.assertGreaterEqual(torch.linalg.eigvalsh(torch.diag(diag)).min().item(), 0)
                torch.testing.assert_close((1 - diag) * fisher * diag, q, rtol=0, atol=0)
                moved, (damped,) = opt.step(q, (p,), torch.zeros_like, gamma=gamma)
                torch.testing.assert_close((1 - diag) * damped, (1 - diag) * p, rtol=0, atol=0)
                torch.testing.assert_close((1 - diag) * moved, h * (1 - diag) * p, rtol=0, atol=0)
                self.assertLess(damped.square().sum().item(), p.square().sum().item())

                distinct = make_optimizer("fisher", h=h, fisher=q.new_tensor([1, 2, 3, 4]))
                torch.testing.assert_close(distinct.brake_diag(q), q.new_tensor([1, 1, 0, 0]))
                zero = make_optimizer("fisher", h=h, fisher=q)
                torch.testing.assert_close(zero.brake_diag(q), torch.ones_like(q), rtol=0, atol=0)
                diag.zero_()
                torch.testing.assert_close(opt.brake_diag(q), q.new_tensor([0, 1, 1, 1]))

                opt = make_optimizer("strang", h=h, fisher=fisher)
                eye = torch.eye(8, device=device, dtype=q.dtype)
                canonical_form = torch.cat((eye[4:], -eye[:4]))
                stiffness = q.new_tensor([0.4, 1.3, 0.7, 1.1])
                columns = []
                for basis in eye:
                    u, v = basis.chunk(2)
                    new_u, (new_v,) = opt.step(u, (v,), lambda x: stiffness * x, gamma=gamma)
                    columns.append(torch.cat((new_u, new_v)))
                jac = torch.stack(columns, dim=1)
                torch.testing.assert_close(
                    jac.T @ canonical_form @ jac,
                    math.exp(-h * gamma) * canonical_form,
                    rtol=0,
                    atol=1e-14,
                )

    def test_quadratic_stability_boundary(self):
        for device in DEVICES:
            for scaled_step in (0.5, 1.0, 1.8, 1.95, 1.99, 2.0, 2.01, 2.05, 2.2):
                with self.subTest(device=device, h_omega=scaled_step):
                    omega = 2.0
                    h = scaled_step / omega
                    q0 = torch.tensor([0.83], device=device, dtype=torch.float64)
                    p0 = q0.new_tensor([0.31])
                    opt = make_optimizer("glide", h=h, fisher=torch.zeros_like(q0))
                    q, state = q0, opt.init(q0, p0)
                    stiffness = omega**2 * (1 - scaled_step**2 / 4)
                    invariant = p0.square() + stiffness * q0.square()
                    for _ in range(48):
                        q, state = opt.step(q, state, lambda x: omega**2 * x, gamma=0.0)
                        if scaled_step < 2:
                            current = state[0].square() + stiffness * q.square()
                            torch.testing.assert_close(current, invariant, rtol=0, atol=1e-12)
                    if scaled_step == 2:
                        torch.testing.assert_close(q, q0 - 48 * h * p0, rtol=0, atol=1e-11)
                    elif scaled_step > 2:
                        self.assertTrue(torch.isfinite(q).all().item())
                        self.assertGreater(q.abs().item(), 50)

    def test_fresh_baselines_match_native(self):
        for device in DEVICES:
            for dtype in DTYPES:
                for name in ("sgdm", "adamw"):
                    with self.subTest(device=device, dtype=dtype, method=name):
                        h = 0.017
                        q = torch.tensor([0.2, -0.7, 0.4], device=device, dtype=dtype)
                        fisher = torch.zeros_like(q)
                        native_q = torch.nn.Parameter(q.clone())
                        if name == "sgdm":
                            native = torch.optim.SGD([native_q], lr=h, momentum=0.9, foreach=False)
                        else:
                            native = torch.optim.AdamW([native_q], lr=h, foreach=False, fused=False)
                        opt = make_optimizer(name, h=h, fisher=fisher)
                        state = opt.init(q, torch.ones_like(q))
                        for step in range(8):
                            calls = []

                            def grad(x):
                                calls.append(x.detach().clone())
                                return x.cos() + step / 7

                            native_q.grad = native_q.detach().cos() + step / 7
                            native.step()
                            opt = make_optimizer(name, h=h, fisher=fisher)
                            old_q = q.clone()
                            q, state = opt.step(q, state, grad, gamma=0.0)
                            self.assertEqual(len(calls), 1)
                            torch.testing.assert_close(calls[0], old_q, rtol=0, atol=0)
                            tolerance = 1e-14 if dtype == torch.float64 else 2e-7
                            torch.testing.assert_close(q, native_q.detach(), rtol=0, atol=tolerance)
                            self.assertFalse(q.requires_grad)

    def test_ewc_penalty_and_frozen_coefficients(self):
        for device in DEVICES:
            for dtype in DTYPES:
                with self.subTest(device=device, dtype=dtype):
                    q = torch.tensor([0.8, -0.4, 0.3], device=device, dtype=dtype)
                    anchor = q.new_tensor([0.5, -0.7, 0.3])
                    fisher = q.new_tensor([2, 0, 4])
                    h, strength = 0.025, 0.7
                    opt = EWC(h, fisher=fisher, anchor=anchor, strength=strength)
                    expected_v = strength * fisher * (q - anchor)
                    state = opt.init(q, torch.ones_like(q))
                    fisher.fill_(0)
                    anchor.fill_(0)
                    actual, (velocity,) = opt.step(q, state, torch.zeros_like, gamma=0.0)
                    torch.testing.assert_close(velocity, expected_v)
                    torch.testing.assert_close(actual, q - h * expected_v)
                    with self.assertRaises(ValueError):
                        EWC(h, fisher=fisher, anchor=anchor, strength=-1)
