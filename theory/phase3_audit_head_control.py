"""Post-hoc depth-eight SGD control using the existing Transformer output head."""
import argparse
import hashlib
import json
from pathlib import Path

from eval_v3.run import cell, configure
from eval_v3.search import search
from eval_v3.task import ROOT, TEST, Transformer, check, tree_hash
from mttt.phase3 import make_engine
import torch
import torch.nn.functional as F


class HeadControl(Transformer):
    def __init__(self, device, verify=False):
        super().__init__(device)
        self.full = Transformer(device) if verify else None
        self.backbone = []
        self.challenge_checks = 0
        self.initial_backbone_hash = None

    def backbone_hash(self):
        digest = hashlib.sha256()
        for w in self.backbone:
            digest.update(w.cpu().numpy().tobytes())
        return digest.hexdigest()

    def initial(self, seed):
        self.backbone = []
        weights = super().parameters(super().initial(seed))
        self.backbone = weights[:-1]
        if self.full is not None:
            self.initial_backbone_hash = self.backbone_hash()
        return weights[-1].reshape(-1).clone()

    def parameters(self, head, grad=False):
        return [*self.backbone, head.detach().view(self.shapes[-1]).requires_grad_(grad)]

    def grad(self, head, batch):
        weights = self.parameters(head, grad=True)
        loss = F.cross_entropy(self.logits(weights, batch), batch[2])
        return torch.autograd.grad(loss, weights[-1])[0].flatten()

    def logits(self, weights, batch):
        out = super().logits(weights, batch)
        if self.full is not None and len(batch[0]) == 8 * 8:
            # Separate untimed traversal verifies the ordinary full weight-vector interface.
            packed = torch.cat([w.reshape(-1) for w in weights])
            direct = self.full.logits(self.full.parameters(packed), batch)
            check(torch.equal(out.view(torch.int32), direct.view(torch.int32)), "challenge logits disagree")
            self.challenge_checks += 1
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    submission = ROOT.parent / "artifacts/sol/p3/sol-20261122-cuda-lock"
    test = json.loads((submission/"test.json").read_bytes())
    lock = (submission/"validation.json").read_bytes()
    check(hashlib.sha256(lock).hexdigest() == test["validation_sha256"] ==
          "152c6bbe335f33bd2b41388bdb67ef221fbd13f7e47621a96d112c92317cc0e9", "validation digest mismatch")
    check(tree_hash(ROOT) == test["eval_sha256"] == "4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d", "eval mismatch")
    check(tree_hash(ROOT.parent/"src/mttt", True) == test["adapter_tree_sha256"], "adapter mismatch")
    cfg = json.loads(lock)["chosen"]
    device = configure("cuda")
    model = HeadControl(device)
    trial = cell(make_engine, model, "sgd_gpu", 8, cfg, TEST)
    check("failure" not in trial, "control failed")
    # These full-vector comparisons occur after all resource measurements.
    auditor = HeadControl(device, verify=True)
    for seed in TEST:
        search(make_engine, auditor, seed, 8, "sgd_gpu", cfg["rates"]["sgd"], [0,0])
        check(auditor.backbone_hash() == auditor.initial_backbone_hash, "frozen backbone changed")
    check(auditor.challenge_checks == 3*(1+9), "missing leaf or initial-model comparison")
    out = {"scope":"Post-hoc readout-only sensitivity on the previously scored public seeds",
           "submission_lock":test["validation_sha256"], "eval_sha256":test["eval_sha256"],
           "adapter_tree_sha256":test["adapter_tree_sha256"], "hardware":torch.cuda.get_device_name(device),
           "seeds":TEST, "h":cfg["rates"]["sgd"], "full_parameters":model.n, "trainable_parameters":model.sizes[-1],
           "full_vector_logit_checks":auditor.challenge_checks, "frozen_backbones_verified":len(TEST), "cell":trial}
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(out,f,allow_nan=False,separators=(",",":"))


if __name__=="__main__":
    main()
