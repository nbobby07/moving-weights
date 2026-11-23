"""Independent checks of the Phase 2 adapter's public mechanics."""

import math
import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mttt.phase2 import make_optimizer


DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
SCALE = 2**44


def setUpModule():
    torch.manual_seed(11173)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def optimizer(name, *, h=0.12, alpha=0.0, strength=0.0, fisher=None, anchor=None):
    return make_optimizer(name, h=h, alpha=alpha, strength=strength, fisher=fisher, anchor=anchor)


def close(actual, expected, atol=1e-13):
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


def force_for(index):
    shift = ((13 * index) % 29 - 14) * 0.021
    return lambda x: x.sin() + shift


class Phase2Tests(unittest.TestCase):
    def test_synchronized_dkd_matches_fused_staggered_drifts(self):
        for device in DEVICES:
            with self.subTest(device=device):
                q0 = torch.tensor([0.31, -0.81, 0.72], dtype=torch.float64, device=device)
                p0 = q0.new_tensor([0.41, -0.09, 0.07])
                fisher = q0.new_tensor([0, 2, 3])
                h, alpha = 0.13, 2.0
                half_drift = (h / 2) / (1 + alpha * fisher / fisher.mean())
                q, p, x = q0, p0, q0 + half_drift * p0
                opt = optimizer("mass", h=h, alpha=alpha, fisher=fisher)
                state = opt.init(q0, p0)
                for i in range(32):
                    force = force_for(i)
                    points = []

                    def gradient(value):
                        points.append(value.clone())
                        return force(value)

                    q, state = opt.step(q, state, gradient, gate=0.0)
                    p = p - h * force(x)
                    self.assertEqual(len(points), 1)
                    close(points[0], x)
                    close(q, x + half_drift * p)
                    close(state[0], p)
                    x = x + (2 * half_drift) * p

    def test_fsal_velocity_verlet_requires_a_matching_force(self):
        q0 = torch.tensor([0.7, -0.2], dtype=torch.float64)
        p0 = q0.new_tensor([0.41, 0.09])
        h = 0.24
        opt = optimizer("glide", h=h)
        force = force_for(3)

        def kdk(q, p, grad, cached=None):
            old_force = grad(q) if cached is None else cached
            mid_p = p - (h / 2) * old_force
            q = q + h * mid_p
            end_force = grad(q)
            return q, mid_p - (h / 2) * end_force, end_force

        def stagger(q, p):
            p = p - (h / 2) * force(q)
            return q + (h / 2) * p, p

        q, p, cached = kdk(q0, p0, force)
        synced_q, (synced_p,) = opt.step(q0, (p0,), force, gate=0.0)
        self.assertGreater(torch.linalg.vector_norm(synced_q - q).item(), 1e-5)
        self.assertGreater(torch.linalg.vector_norm(synced_p - p).item(), 1e-5)
        staggered_q, staggered_p = stagger(q0, p0)
        actual_q, (actual_p,) = opt.step(staggered_q, (staggered_p,), force, gate=0.0)
        expected_q, expected_p = stagger(q, p)
        close(actual_q, expected_q)
        close(actual_p, expected_p)

        fixed_q, fixed_p, _ = kdk(q, p, force, cached)
        fresh_q, fresh_p, _ = kdk(q, p, force)
        close(fixed_q, fresh_q, atol=0)
        close(fixed_p, fresh_p, atol=0)
        changed = force_for(4)
        stale_q, stale_p, _ = kdk(q, p, changed, cached)
        fresh_q, fresh_p, _ = kdk(q, p, changed)
        self.assertGreater(torch.linalg.vector_norm(stale_q - fresh_q).item(), 1e-5)
        self.assertGreater(torch.linalg.vector_norm(stale_p - fresh_p).item(), 1e-5)

        constant = lambda x: torch.full_like(x, 0.3)
        q, p, _ = kdk(q0, p0, constant)
        synced_q, (synced_p,) = opt.step(q0, (p0,), constant, gate=0.0)
        close(synced_q, q)
        close(synced_p, p)

    def test_mass_jacobian_volume_and_fresh_replay(self):
        for device in DEVICES:
            with self.subTest(device=device):
                q0 = torch.tensor([0.65, -0.91], dtype=torch.float64, device=device)
                p0 = q0.new_tensor([0.38, 0.29])
                fisher = q0.new_tensor([0.7, 3.0])
                h, alpha = 0.12, 4.0
                opt = optimizer("mass", h=h, alpha=alpha, fisher=fisher)
                spring = q0.new_tensor([[1.1, 0.2], [0.2, 0.8]])
                eye = torch.eye(2, dtype=q0.dtype, device=device)
                zero = torch.zeros_like(eye)
                half_drift = torch.diag((h / 2) / (1 + alpha * fisher / fisher.mean()))
                drift = torch.cat((torch.cat((eye, half_drift), 1), torch.cat((zero, eye), 1)))
                kick = torch.cat((torch.cat((eye, zero), 1), torch.cat((-h * spring, eye), 1)))
                canonical = torch.cat((torch.cat((zero, eye), 1), torch.cat((-eye, zero), 1)))
                axes = torch.eye(4, dtype=q0.dtype, device=device)
                columns = []
                for axis in axes:
                    q, p = axis.chunk(2)
                    q, (p,) = opt.step(q, (p,), lambda x: spring @ x, gate=0.0)
                    columns.append(torch.cat((q, p)))
                jacobian = torch.stack(columns, dim=1)
                close(jacobian, drift @ kick @ drift)
                close(jacobian.T @ canonical @ jacobian, canonical)
                self.assertAlmostEqual(torch.linalg.det(jacobian).item(), 1.0, places=13)

                q, state = q0, opt.init(q0, p0)
                for i in range(500):
                    q, state = opt.step(q, state, force_for(i), gate=0.0)
                fresh = optimizer("mass", h=h, alpha=alpha, fisher=fisher)
                back_q, back_state = q, (-state[0],)
                wrong_q, wrong_state = back_q, back_state
                for i in range(499, -1, -1):
                    back_q, back_state = fresh.step(back_q, back_state, force_for(i), gate=0.0)
                    wrong_q, wrong_state = fresh.step(wrong_q, wrong_state, force_for(499 - i), gate=0.0)
                self.assertLess(torch.linalg.vector_norm(back_q - q0).item(), 1e-11)
                self.assertLess(torch.linalg.vector_norm(-back_state[0] - p0).item(), 1e-11)
                self.assertGreater(torch.linalg.vector_norm(wrong_q - q0).item(), 1e-4)

    def test_generic_reservoir_balance_and_nonzero_gate(self):
        for device in DEVICES:
            with self.subTest(device=device):
                q = torch.tensor([0.6, -0.7, 0.9], dtype=torch.float64, device=device)
                p = q.new_tensor([0.2, 0.35, -0.8])
                anchor = q.new_tensor([0.1, 0.2, 0.3])
                fisher = q.new_tensor([1.0, 0.0, 4.0])
                weights = fisher / fisher.mean()
                h, alpha, strength, gate = 0.13, 1.7, 0.47, 0.35
                generic = optimizer("generic", h=h, alpha=alpha, strength=strength, fisher=fisher, anchor=anchor)
                mass = optimizer("mass", h=h, alpha=alpha, fisher=fisher)

                def prior(x):
                    return (weights * (x - anchor).square()).sum() / 2

                def potential(x):
                    return (0.2 * x.square() + x.sin()).sum()

                def grad(x):
                    return 0.4 * x + x.cos()

                duration = h * gate / 2
                start = generic.anchor_flow(q, duration)
                close(start, anchor + torch.exp(-duration * strength * weights) * (q - anchor))
                close(generic.anchor_flow(start, duration), generic.anchor_flow(q, 2 * duration))
                self.assertEqual(start[1].item(), q[1].item())
                mid, (new_p,) = mass.step(start, (p,), grad, gate=0.0)
                end = generic.anchor_flow(mid, duration)
                actual, (actual_p,) = generic.step(q, (p,), grad, gate=gate)
                close(actual, end)
                close(actual_p, new_p)
                mass_q, (mass_p,) = mass.step(q, (p,), grad, gate=0.0)
                off_q, (off_p,) = generic.step(q, (p,), grad, gate=0.0)
                close(off_q, mass_q, atol=0)
                close(off_p, mass_p, atol=0)

                first_flux = potential(q) - potential(start)
                second_flux = potential(mid) - potential(end)
                close(potential(start) + first_flux, potential(q))
                close(potential(end) + second_flux, potential(mid))
                sigma_gain = prior(mid) - prior(start)
                entropy_gain = sigma_gain - prior(end) + prior(q)
                close(entropy_gain, prior(q) - prior(start) + prior(mid) - prior(end))
                self.assertGreater(entropy_gain.item(), 0)
                self.assertGreater(prior(q).item(), prior(start).item())
                self.assertGreater(prior(mid).item(), prior(end).item())

    def test_energy_arm_capacity_and_empty_projector(self):
        conformance = (("iso", [True, True, True, True]), ("low", [True, True, False, True]),
                       ("high", [False, False, True, False]))
        for device in DEVICES:
            with self.subTest(device=device):
                fisher = torch.tensor([2.0, 2.0, 9.0, 1.0], dtype=torch.float64, device=device)
                momenta = (fisher.new_tensor([1.0, -0.2, 0.9, -0.5]),
                           fisher.new_tensor([0.1, 0.3, 0.8, -0.1]),
                           fisher.new_tensor([0.9, -0.4, 0.7, 0.3]))
                masks = [torch.tensor(mask, device=device) for _, mask in conformance]
                iso, low, high = momenta
                _, low_mask, high_mask = masks
                iso_kinetic = iso.square().sum() / 2
                # Exact saturation uses the coordinator's full-width masked reductions.
                cap_low = torch.where(low_mask, low.square(), 0).sum() / 2
                cap_high = torch.where(high_mask, high.square(), 0).sum() / 2
                requested = -math.expm1(-2 * 0.12 * 0.9) * iso_kinetic
                common = torch.minimum(requested, torch.minimum(cap_low, cap_high))
                self.assertGreater(requested.item(), common.item())
                self.assertEqual(common.item(), cap_low.item())
                for (name, _), p, mask in zip(conformance, momenta, masks):
                    opt = optimizer(name, fisher=fisher)
                    q = torch.zeros_like(p)
                    _, (pre_brake,) = opt.step(q, opt.init(q, p), torch.zeros_like, gate=0.0)
                    close(pre_brake, p, atol=0)
                    out = opt.brake(pre_brake, common)
                    close((p.square() - out.square()).sum() / 2, common)
                    close(out[~mask], p[~mask], atol=0)
                    close(opt.brake(p, common * 0), p, atol=0)
                    with self.assertRaises(ValueError):
                        opt.brake(p, p.square().sum())
                    if name == "low":
                        close(out[mask], torch.zeros_like(out[mask]), atol=0)
                        indexed_cap = p[mask].square().sum() / 2
                        alternative = torch.minimum(indexed_cap, common)
                        alternate_out = opt.brake(p, alternative)
                        removed = (p.square().sum() - alternate_out.square().sum()) / 2
                        close(removed, alternative)
                empty = optimizer("high", fisher=torch.zeros_like(fisher))
                close(empty.brake(iso, common * 0), iso, atol=0)
                with self.assertRaises(ValueError):
                    empty.brake(iso, common)

    def test_lattice_rounding_and_bitwise_replay(self):
        for device in DEVICES:
            with self.subTest(device=device):
                ties = torch.tensor([1, 3, -1, -3], dtype=torch.int64, device=device)
                zero = torch.zeros_like(ties)
                expected = ties.new_tensor([0, 4, 0, -4])
                opt = optimizer("lattice", h=1.0)
                q, (p,) = opt.step(zero, opt.init(zero, ties), torch.zeros_like, gate=0.0)
                self.assertTrue(torch.equal(q, expected))
                self.assertTrue(torch.equal(p, ties))
                force_ties = ties.float() / (2 * SCALE)
                q, (p,) = opt.step(zero, (zero,), lambda _: force_ties, gate=0.0)
                self.assertTrue(torch.equal(q, ties.new_tensor([0, -1, 0, 1])))
                self.assertTrue(torch.equal(p, ties.new_tensor([0, -2, 0, 2])))
                with self.assertRaises(ValueError):
                    opt.step(zero, (zero,), lambda x: torch.full_like(x, 128.0), gate=0.0)
                with self.assertRaises(ValueError):
                    opt.step(torch.full_like(zero, 2**52 - 1), (torch.full_like(zero, 2),),
                             torch.zeros_like, gate=0.0)

                q0 = ties.new_tensor([3 * SCALE // 4, -SCALE // 2, SCALE // 4])
                p0 = ties.new_tensor([SCALE // 16, SCALE // 8, -SCALE // 4])
                opt = optimizer("lattice")
                q, state = q0, opt.init(q0, p0)
                calls = []

                def gradient(i):
                    def grad(x):
                        calls.append(i)
                        self.assertEqual(x.dtype, torch.float32)
                        self.assertEqual(x.device.type, device)
                        return force_for(i)(x)
                    return grad

                for i in range(500):
                    q, state = opt.step(q, state, gradient(i), gate=0.0)
                self.assertEqual(q.dtype, torch.int64)
                self.assertFalse(torch.equal(q, q0))
                self.assertFalse(torch.equal(state[0], p0))
                fresh = optimizer("lattice")
                state = (-state[0],)
                for i in range(499, -1, -1):
                    q, state = fresh.step(q, state, gradient(i), gate=0.0)
                self.assertTrue(torch.equal(q, q0))
                self.assertTrue(torch.equal(-state[0], p0))
                self.assertEqual(calls, list(range(500)) + list(range(499, -1, -1)))

                with self.assertRaises((ValueError, OverflowError)):
                    opt.step(q0, (p0,), lambda x: torch.full_like(x, float("nan")), gate=0.0)
                with self.assertRaises((ValueError, OverflowError)):
                    opt.step(q0, (p0,), lambda x: torch.full_like(x, 2048.0), gate=0.0)
                with self.assertRaises((ValueError, OverflowError)):
                    opt.step(torch.full_like(q0, 2**52), (p0,), torch.zeros_like, gate=0.0)
                with self.assertRaises(TypeError):
                    opt.init(q0.float(), p0)
                with self.assertRaises(TypeError):
                    opt.step(q0, (p0,), lambda x: torch.zeros_like(x).double(), gate=0.0)

    def test_baseline_trajectories_match_native_with_fresh_instances(self):
        for device in DEVICES:
            for dtype in (torch.float64, torch.float32):
                for name in ("sgdm", "adamw", "ewc"):
                    with self.subTest(device=device, dtype=dtype, method=name):
                        q = torch.tensor([0.2, -0.7, 0.4], device=device, dtype=dtype)
                        fisher, anchor = q.new_tensor([0.0, 2.5, 4.0]), q.clone()
                        h, strength = 0.012, 0.4
                        prior = {"fisher": fisher, "anchor": anchor, "strength": strength} if name == "ewc" else {}
                        native_q = torch.nn.Parameter(q.clone())
                        if name == "adamw":
                            native = torch.optim.AdamW([native_q], lr=h, foreach=False, fused=False)
                        else:
                            native = torch.optim.SGD([native_q], lr=h, momentum=0.9, foreach=False)
                        state = optimizer(name, h=h, **prior).init(q, torch.ones_like(q))
                        for i in range(8):
                            force = force_for(i)
                            calls = []

                            def grad(x):
                                calls.append(x)
                                return force(x)

                            x = native_q.detach()
                            native_q.grad = force(x)
                            if name == "ewc":
                                native_q.grad += strength * (fisher / fisher.mean()) * (x - anchor)
                            native.step()
                            q, state = optimizer(name, h=h, **prior).step(q, state, grad, gate=0.0)
                            self.assertEqual(len(calls), 1)
                            self.assertFalse(q.requires_grad)
                            close(q, native_q.detach(), atol=1e-13 if dtype == torch.float64 else 2e-7)
