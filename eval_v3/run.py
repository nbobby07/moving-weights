"""Immutable calibration receipt, separate held-out workers and reference audit."""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean
import subprocess
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch

from .metrics import coverage,fit_monitor,smooth,summarize
from .physics import physics,repeat_probe,tangent_probe,tape
from .search import Memory,mode,search,stamp
from .task import DEPTHS,METHODS,ROOT,TEST,VAL,Problem,Transformer,check,tree_hash


GRID={"native":(1/32,1/16),"sgd":(1/1024,1/256),"adam":(1/16384,1/4096)}


def write(path,data):
    blob=json.dumps(data,separators=(",",":"),allow_nan=False).encode()+b"\n"
    with path.open("xb") as dst:
        dst.write(blob)
    return hashlib.sha256(blob).hexdigest()


def rank(candidate):
    rows=candidate["rows"]
    smoothness=min(smooth(rows))
    loss=mean(l["signals"][1] for r in rows for l in r["leaves"])
    return smoothness>=0.90,mean(coverage(r)["assigned_accuracy"] for r in rows),-loss


def choose(raw):
    winners={m:max(trials,key=rank) for m,trials in raw["grids"].items()}
    amplitude=[max(r[k] for r in raw["ranges"]) for k in ("q_max","p_max")]
    shifts={}
    for name,limit,cap in (("int32",2**31-1,30),("int64",2**50-1,44)):
        shifts[name]=[min(cap,math.floor(math.log2(limit/(4*max(a,2**-20))))) for a in amplitude]
    return {"rates":{m:t["h"] for m,t in winners.items()},"shifts":shifts,
            "monitor":fit_monitor(winners["native"]["rows"])}


def validate(factory,model):
    grids={}
    for m,rates in GRID.items():
        name=m if m=="native" else m+"_gpu"
        grids[m]=[{"h":h,"rows":[search(factory,model,s,4,name,h,[0,0]) for s in VAL]} for h in rates]
    best=max(grids["native"],key=rank)["h"]
    ranges=[]
    for s in VAL:
        for h in (best,2*best):
            q=model.initial(s)
            st=(q,torch.sign(q)*2**-14)
            eng=factory("native",h=h,shifts=[0,0])
            prob=Problem(s,8,model.device)
            peaks=torch.stack([x.abs().amax() for x in st])
            for i in range(64):
                st=eng.step(st,lambda x:model.grad(x,tape(prob,i)))
                peaks=torch.maximum(peaks,torch.stack([x.abs().amax() for x in st]))
            ranges.append({"seed":s,"h":h,"q_max":peaks[0].item(),"p_max":peaks[1].item()})
    raw={"grids":grids,"ranges":ranges}
    return {"seeds":VAL,"repeat":repeat_probe(model,VAL[0]),"tangent_probe":tangent_probe(factory,model,VAL[0]),
            "raw":raw,"chosen":choose(raw)}


def configure(device):
    torch.set_num_threads(1)
    torch.manual_seed(33187)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch.device(device)


def cell(factory,model,method,depth,cfg,seeds):
    # Warm only kernels, using a validation seed. Audit checkpoints are created after all measured searches.
    warm=model.initial(VAL[0])
    model.grad(warm,Problem(VAL[0],1,model.device).bank[0])
    del warm
    rows=[]; audits=[]
    name=mode(method); h=cfg["rates"].get(name,cfg["rates"]["native"])
    shifts=cfg["shifts"].get(name,[0,0])
    try:
        for s in seeds:
            gc.collect()
            if model.device.type=="cuda":
                torch.cuda.empty_cache()
            with Memory(model.device) as mem:
                t=stamp(model.device)
                row=search(factory,model,s,depth,method,h,shifts)
                row["setup_plus_search_s"]=stamp(model.device)-t
            row["memory"]=mem.result()
            rows.append(row)
        if name in ("int32","int64","native") and method!="single":
            for s in seeds:
                a=search(factory,model,s,depth,method,h,shifts,audit=True)
                audits.append({k:a[k] for k in ("seed","parent_errors","bit_equal","rounding","forward","reverse")})
    except (ArithmeticError,torch.OutOfMemoryError) as exc:
        return {"failure":type(exc).__name__+": "+str(exc),"seed":s,"rows":rows,"audits":audits}
    return {"rows":rows,"audits":audits}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("smoke","validate","test","cell","physics"))
    parser.add_argument("--factory",default="eval_v3.reference:make_engine")
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--expected-root")
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--lock-sha")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--depth",type=int,choices=DEPTHS)
    args=parser.parse_args()
    path=args.output.resolve()
    check(not path.exists() and path.parent.is_dir(),"choose a new artifact path in an existing directory")
    check(not any(path.is_relative_to(ROOT.parent/t) for t in ("eval","eval_v2","eval_v3")),"frozen output path")
    pin=args.expected_root or (ROOT.parent/"theory/phase3_eval.sha256").read_text().strip()
    check(tree_hash(ROOT)==pin,"evaluator pin mismatch")
    module,fn=args.factory.split(":")
    factory=getattr(importlib.import_module(module),fn)
    pkg=Path(importlib.import_module(module.split(".")[0]).__file__).resolve().parent
    roots={"eval_sha256":pin,"factory":args.factory,"adapter_tree_sha256":tree_hash(pkg,True)}
    # The orchestrator itself must not initialize a CUDA context while measuring child processes.
    report={**roots,"stage":args.stage,"device":args.device,"torch":str(torch.__version__),
            "python":sys.version.split()[0],"platform":platform.platform(),"cuda_build":torch.version.cuda}
    cfg=None
    if args.stage in ("test","cell","physics"):
        check(args.lock and args.lock_sha,"expected a pre-recorded validation lock and digest")
        blob=args.lock.read_bytes()
        check(hashlib.sha256(blob).hexdigest()==args.lock_sha,"validation lock digest mismatch")
        receipt=json.loads(blob)
        check(all(receipt[k]==v for k,v in roots.items()) and receipt["stage"]=="validate" and
              receipt["device"]=="cuda" and receipt["seeds"]==list(VAL),"wrong validation provenance")
        cfg=choose(receipt["raw"])
        check(cfg==receipt["chosen"],"selection does not reproduce from validation data")
        report["validation_sha256"]=args.lock_sha
    if args.stage=="test":
        check(args.device=="cuda","full systems audit needs CUDA")
        parent=path.with_name(path.stem+"_parts")
        parent.mkdir(exist_ok=True)
        cells={};paths={}
        for key in ("physics",*(f"{m}_{d}" for d in DEPTHS for m in METHODS)):
            dest=parent/(key+".json")
            stage="physics" if key=="physics" else "cell"
            extra=[]
            if stage=="cell":
                method,depth=key.rsplit("_",1)
                extra=["--method",method,"--depth",depth]
            if not dest.exists():
                subprocess.run([sys.executable,"-B","-m","eval_v3.run",stage,"--factory",args.factory,
                                "--expected-root",pin,"--device",args.device,"--lock",str(args.lock.resolve()),
                                "--lock-sha",args.lock_sha,"--output",str(dest),*extra],check=True)
            payload=dest.read_bytes(); sub=json.loads(payload)
            check(all(sub[k]==v for k,v in roots.items()) and sub["validation_sha256"]==args.lock_sha and
                  sub["stage"]==stage and sub["device"]==args.device and sub["seeds"]==list(TEST)
                  and sub["torch"]==report["torch"] and sub["python"]==report["python"],"stale child artifact")
            if stage=="physics":
                phys=sub["physics"];hardware=sub["hardware"]
            else:
                check(sub["method"]==method and sub["depth"]==int(depth) and sub["hardware"]==hardware,"wrong cell")
                cells[key]=sub["cell"]
            paths[key]={"path":str(dest.relative_to(path.parent).as_posix()),"sha256":hashlib.sha256(payload).hexdigest()}
        report.update(seeds=TEST,hardware=hardware,chosen=cfg,parts=paths,decisions=summarize(cells,cfg["monitor"],phys))
    else:
        device=configure(args.device)
        model=Transformer(device,smoke=args.stage=="smoke")
        report.update(hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),parameters=model.n)
        if args.stage=="validate":
            check(device.type=="cuda","canonical calibration uses CUDA")
            report.update(validate(factory,model))
        elif args.stage=="smoke":
            cfg={"rates":{"native":1/16,"adam":1/4096,"sgd":1/256},"shifts":{"int32":[27,30],"int64":[40,42]}}
            report["physics"]=physics(factory,model,cfg,VAL[:1],True)
            report["cells"]={m:cell(factory,model,m,2,cfg,VAL[:1]) for m in METHODS}
            full=Transformer(device)
            report["full_size_cpu_probe"]={"parameters":full.n,"repeat":repeat_probe(full,VAL[0]),
                                            "tangent":tangent_probe(factory,full,VAL[0])}
            check(report["physics"]["long_integer_pass"] and all("failure" not in c for c in report["cells"].values()),"smoke failed")
        elif args.stage=="physics":
            report.update(seeds=TEST,physics=physics(factory,model,cfg,TEST))
        else:
            check(args.method and args.depth,"cell needs method and depth")
            report.update(seeds=TEST,method=args.method,depth=args.depth,
                          cell=cell(factory,model,args.method,args.depth,cfg,TEST))
    check(tree_hash(ROOT)==pin and tree_hash(pkg,True)==roots["adapter_tree_sha256"],"source changed during run")
    print(args.stage,path,write(path,report),flush=True)


if __name__=="__main__":
    main()
