"""Run physics, validation-only selection, or locked held-out evaluation."""
import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from . import physics, recovery, retention
from .protocol import PIN1, ROOT, TEST, VAL, check, prepare, tree_hash


def choose(raw):
    return {"C":retention.choose(raw["C"]), "B":recovery.choose(raw["B"])}


def locked(args, roots):
    check(args.lock and args.lock_sha,"test requires both a validation lock and its independently retained SHA-256")
    blob = args.lock.read_bytes()
    check(hashlib.sha256(blob).hexdigest()==args.lock_sha,"validation receipt digest mismatch")
    lock = json.loads(blob)
    for k,v in roots.items():
        check(lock[k]==v,k+" changed since validation")
    check(lock["stage"]=="validate" and lock["device"]=="cpu" and lock["seeds"]==list(VAL),
          "test requires the canonical CPU validation receipt")
    chosen=choose(lock["raw"])
    check(chosen==lock["chosen"],"selection disagrees with validation records")
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("physics","validate","test"))
    parser.add_argument("--device",choices=("cpu","cuda","auto"),default="auto")
    parser.add_argument("--factory",default="eval_v2.reference:make_optimizer")
    parser.add_argument("--expected-root",default=None)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--lock-sha")
    args=parser.parse_args()
    output=args.output.resolve()
    forbidden=(ROOT,ROOT.parent/"eval")
    check(not any(output.is_relative_to(p) for p in forbidden),"output points inside a frozen tree")
    check(not output.exists(),"choose a new output path")
    pin=args.expected_root or (ROOT.parent/"theory/phase2_eval.sha256").read_text().strip()
    def integrity():
        check(tree_hash(ROOT)==pin,"Phase 2 evaluator pin mismatch")
        check(tree_hash(ROOT.parent/"eval")==PIN1,"shared Phase 1 code pin mismatch")
    integrity()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(11173)
    device=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    module,fn=args.factory.split(":")
    package=importlib.import_module(module.split(".")[0])
    factory=getattr(importlib.import_module(module),fn)
    folder=Path(package.__file__).resolve().parent
    roots={"eval_v2_sha256":pin,"eval_v1_sha256":PIN1,"factory":args.factory,
           "adapter_python_tree_sha256":tree_hash(folder,python_only=True)}
    report={**roots,"stage":args.stage,"device":device.type,"python":sys.version.split()[0],
            "torch":str(torch.__version__),"cuda_build":torch.version.cuda,"deterministic":True,
            "hardware":torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor()}
    if args.stage=="physics":
        report["probes"]=physics.verify(factory,device)
    elif args.stage=="validate":
        preps=[prepare(s,device) for s in VAL]
        raw={"C":retention.validate(factory,preps),"B":recovery.validate(factory,preps,VAL,device)}
        report.update(seeds=VAL,raw=raw,chosen=choose(raw))
    else:
        cfg=locked(args,roots)
        report.update(validation_sha256=args.lock_sha,chosen=cfg,seeds=TEST)
        report["probes"]=physics.verify(factory,device)
        bs,cs,hs=[],[],[]
        for i,s in enumerate(TEST):
            prep=prepare(s,device)
            cs.append(retention.trial(factory,cfg["C"],prep,i))
            bs.append(recovery.trial(factory,cfg["B"],prep,s,device))
            if i<3:
                hs.append(recovery.horizon(factory,prep))
            if i==0:
                report["timing"]=retention.timing(factory,cfg["C"],prep)
        report.update(B=bs,C=cs,horizon_seeds=TEST[:3],horizons=hs,
                      decisions={"B":recovery.verdict(bs),"C":retention.verdict(cs),
                                 "horizons":recovery.horizon_verdict(hs)})
    integrity()
    check(tree_hash(folder,python_only=True)==roots["adapter_python_tree_sha256"],"adapter changed during run")
    payload=json.dumps(report,allow_nan=False,separators=(",",":")).encode()+b"\n"
    with output.open("xb") as dst:
        dst.write(payload)
    print(args.stage,device.type,output,hashlib.sha256(payload).hexdigest(),flush=True)


if __name__=="__main__":
    main()
