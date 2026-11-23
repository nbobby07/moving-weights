"""Small synthetic tests of the Phase 3 engine adapter."""

import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from mttt.phase3 import make_engine


DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
FORMATS = (("int32", 2**31 - 1), ("int64", 2**50 - 1))


def setUpModule():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def close(actual, expected, atol=2e-7):
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


class Phase3Tests(unittest.TestCase):
    def test_native_force_sites_fresh_engine_and_detachment(self):
        for device in DEVICES:
            with self.subTest(device=device):
                q = torch.tensor([0.5, -0.75, 0.125], dtype=torch.float32, device=device).requires_grad_()
                p = q.new_tensor([-0.25, 0.375, 0.0625], requires_grad=True)
                q0, p0 = q.detach().clone(), p.detach().clone()
                h = 1 / 8
                engine = make_engine("native", h=h, shifts=[0, 0])
                state = engine.init(q, p)
                self.assertEqual(len(state), 2)
                self.assertFalse(any(value.requires_grad for value in state))
                old = state
                for batch, shift in enumerate((0.125, -0.375, 0.5)):
                    calls = []

                    def grad(mid):
                        calls.append(mid.detach().clone())
                        with torch.enable_grad():
                            leaf = mid.detach().requires_grad_()
                            potential = (leaf.sin() + shift * leaf).sum()
                            return torch.autograd.grad(potential, leaf)[0]

                    mid = q0 + (h / 2) * p0
                    p0 = p0 - h * (mid.cos() + shift)
                    q0 = mid + (h / 2) * p0
                    engine = make_engine("native", h=h, shifts=[0, 0])
                    if batch == 0:
                        state = engine.step((q, p), grad)
                    else:
                        state = engine.step(state, grad)
                    self.assertEqual(len(calls), 1)
                    close(calls[0], mid)
                    close(engine.weights(state), q0)
                    close(state[1], p0)
                    for value in state:
                        self.assertFalse(value.requires_grad)
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertEqual(value.device.type, device)
                close(q, q.new_tensor([0.5, -0.75, 0.125]), atol=0)
                close(p, p.new_tensor([-0.25, 0.375, 0.0625]), atol=0)
                close(old[0], q, atol=0)
                close(old[1], p, atol=0)

                reverse = make_engine("native", h=h, shifts=[0, 0])
                state = (state[0], -state[1])
                for shift in (0.5, -0.375, 0.125):
                    state = reverse.step(state, lambda x: x.cos() + shift)
                close(state[0], q)
                close(-state[1], p)

    def test_lattice_rounds_ties_even_and_revisits_identical_midpoints(self):
        for device in DEVICES:
            for method, _ in FORMATS:
                with self.subTest(device=device, method=method):
                    start_q = torch.tensor([4, -5, 0, 2], dtype=torch.float32, device=device) / 16
                    start_p = start_q.new_tensor([2, 6, -2, -6]) / 32
                    h, shifts = 1.0, [4, 5]
                    engine = make_engine(method, h=h, shifts=shifts)
                    state = engine.init(start_q, start_p)
                    code_q, code_p = state
                    self.assertEqual(code_q.dtype, getattr(torch, method))
                    self.assertEqual(code_p.dtype, getattr(torch, method))
                    close(engine.weights(state), start_q, atol=0)
                    self.assertIsNone(engine.diagnostics)
                    seen = []

                    def force(mid):
                        self.assertEqual(mid.dtype, torch.float32)
                        self.assertEqual(mid.device.type, device)
                        seen.append(mid.clone())
                        return mid.new_tensor([1, 3, -1, -3]) / 64

                    end_q, end_p = engine.step(state, force)
                    close(end_q, code_q.new_tensor([4, -2, 0, -1]), atol=0)
                    close(end_p, code_p.new_tensor([2, 4, -2, -4]), atol=0)
                    close(seen[0], start_q.new_tensor([4, -3, 0, 0]) / 16, atol=0)
                    reverse = make_engine(method, h=h, shifts=shifts)
                    q, p = reverse.step((end_q, -end_p), force)
                    self.assertEqual(len(seen), 2)
                    self.assertTrue(torch.equal(seen[0].view(torch.int32), seen[1].view(torch.int32)))
                    self.assertTrue(torch.equal(q, code_q))
                    self.assertTrue(torch.equal(-p, code_p))
                    close(code_q, end_q.new_tensor([4, -5, 0, 2]), atol=0)
                    close(code_p, end_p.new_tensor([2, 6, -2, -6]), atol=0)

    def test_lattice_checks_increment_sum_and_lossless_encoding(self):
        for device in DEVICES:
            for method, limit in FORMATS:
                with self.subTest(device=device, method=method):
                    engine = make_engine(method, h=1.0, shifts=[0, 0])
                    edge = torch.tensor([limit], dtype=getattr(torch, method), device=device)
                    engine.where = ["synthetic_sum_guard"]
                    with self.assertRaisesRegex(ArithmeticError, "synthetic_sum_guard") as sum_error:
                        engine.step((edge, torch.full_like(edge, 4)), torch.zeros_like)
                    self.assertIn("drift1", str(sum_error.exception))
                    self.assertIn("index=0", str(sum_error.exception))
                    self.assertIn(f"attempted={limit + 2}", str(sum_error.exception))

                    engine.where = ["synthetic_increment_guard"]
                    with self.assertRaisesRegex(ArithmeticError, "synthetic_increment_guard") as increment_error:
                        engine.step(
                            (torch.zeros_like(edge), torch.full_like(edge, -2)),
                            lambda mid: torch.full_like(mid, -(limit + 1)),
                        )
                    self.assertIn("kick", str(increment_error.exception))
                    self.assertIn("index=0", str(increment_error.exception))
                    self.assertIn(f"increment={limit + 1}", str(increment_error.exception))

                    engine.where = ["synthetic_nonfinite_guard"]
                    zero = torch.zeros_like(edge)
                    with self.assertRaisesRegex(ArithmeticError, "synthetic_nonfinite_guard"):
                        engine.step((zero, zero), lambda mid: torch.full_like(mid, float("nan")))

                    f32 = torch.tensor([0.25], dtype=torch.float32, device=device)
                    with self.assertRaises(ArithmeticError):
                        engine.init(f32, torch.zeros_like(f32))

    def test_lattice_traverses_blocks_with_global_rounding_diagnostics(self):
        device = DEVICES[-1]
        boundary = 2**21
        coordinates = [0, boundary - 1, boundary, boundary + 3]
        for method, _ in FORMATS:
            with self.subTest(device=device, method=method):
                engine = make_engine(method, h=0.5, shifts=[8, 8])
                q = torch.zeros(boundary + 5, dtype=torch.float32, device=device)
                p = torch.zeros_like(q)
                p[coordinates] = p.new_tensor([3, 4, 1, -6]) / 256
                state = engine.init(q, p)
                engine.where = ["synthetic_blocks"]
                engine.diagnostics = {}
                calls = []

                def gradient(mid):
                    calls.append(mid[coordinates].clone())
                    out = torch.zeros_like(mid)
                    out[coordinates] = out.new_tensor([4, -4, 1, 0]) / 256
                    return out

                end = engine.step(state, gradient)
                close(end[0][coordinates], state[0].new_tensor([1, 3, 0, -4]), atol=0)
                close(end[1][coordinates], state[1].new_tensor([1, 6, 1, -6]), atol=0)
                self.assertEqual(int(end[0].count_nonzero()), 3)
                self.assertEqual(int(end[1].count_nonzero()), 4)
                self.assertEqual(engine.diagnostics["lost_components"], 4)
                self.assertEqual(engine.diagnostics["nonzero_components"], 11)
                self.assertEqual(engine.diagnostics["first_sub_lsb"],
                                 [["synthetic_blocks"], "drift1", boundary, 0.25])
                self.assertNotIn("first_whole_shear_lost", engine.diagnostics)

                fresh = make_engine(method, h=0.5, shifts=[8, 8])
                back_q, back_p = fresh.step((end[0], -end[1]), gradient)
                self.assertEqual(len(calls), 2)
                self.assertTrue(torch.equal(calls[0].view(torch.int32), calls[1].view(torch.int32)))
                self.assertTrue(torch.equal(back_q, state[0]))
                self.assertTrue(torch.equal(-back_p, state[1]))

                event = engine.diagnostics["first_sub_lsb"]
                small = torch.zeros(2, dtype=getattr(torch, method), device=device)
                engine.where = ["synthetic_whole_shear"]
                engine.step((small, small), lambda mid: torch.full_like(mid, 1 / 1024))
                self.assertEqual(engine.diagnostics["first_sub_lsb"], event)
                self.assertEqual(engine.diagnostics["first_whole_shear_lost"],
                                 [["synthetic_whole_shear"], "kick", 2])
                self.assertEqual(engine.diagnostics["lost_components"], 6)
                self.assertEqual(engine.diagnostics["nonzero_components"], 13)

    def test_baseline_fresh_states_match_native_pytorch(self):
        for device in DEVICES:
            for method in ("sgd", "adam"):
                with self.subTest(device=device, method=method):
                    q = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32, device=device)
                    h = 1 / 64
                    engine = make_engine(method, h=h, shifts=[0, 0])
                    state = engine.init(q, torch.ones_like(q))
                    native_q = torch.nn.Parameter(q.clone())
                    if method == "sgd":
                        native = torch.optim.SGD([native_q], lr=h, momentum=0.9, foreach=False)
                    else:
                        native = torch.optim.AdamW([native_q], lr=h, foreach=False, fused=False)
                        self.assertEqual(state[3], 0)
                    for i in range(1, 9):
                        recorded = []

                        def force(x):
                            recorded.append(x.detach().clone())
                            return x.cos() + (i - 3) / 16

                        old_q = state[0].clone()
                        native_q.grad = native_q.detach().cos() + (i - 3) / 16
                        native.step()
                        engine = make_engine(method, h=h, shifts=[0, 0])
                        state = engine.step(state, force)
                        self.assertEqual(len(recorded), 1)
                        close(recorded[0], old_q, atol=0)
                        close(engine.weights(state), native_q.detach())
                        self.assertFalse(state[0].requires_grad)
                        buffers = native.state[native_q]
                        if method == "sgd":
                            close(state[1], buffers["momentum_buffer"], atol=1e-6)
                        else:
                            close(state[1], buffers["exp_avg"])
                            close(state[2], buffers["exp_avg_sq"])
                            self.assertIs(type(state[3]), int)
                            self.assertEqual(state[3], i)
