"""Descriptive follow-up to the frozen grid. Find the first failed integer horizon."""
import argparse
import hashlib
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from eval_v2.protocol import PIN1, ROOT, TEST, ForceCounter, accuracy, check, l2, make, prepare, tree_hash
from eval_v2.reference import make_optimizer


def first_failure(seed, device, grid, stop):
    q, _, tests, streams, *_ = prepare(seed,device)
    q0 = q.float()
    p0 = (0.02*torch.cos(torch.arange(q0.numel(),device=device,dtype=torch.float64))).float()
    test = (tests[0][0].float(),tests[0][1])
    initial_correct = round(300*accuracy(q0,test))
    tape = [(x.float(),y) for task in streams for x,y in task]
    cfg = {"h":0.12}
    opt = make(make_optimizer,"glide",cfg,q0)
    q,state = q0.clone(),opt.init(q0,p0)
    counter = ForceCounter()
    trace = []
    for t in range(1,stop+1):
        q,state = counter.step(opt,q,state,tape[(t-1)%len(tape)])
        fresh = make(make_optimizer,"glide",cfg,q)
        back,rev = q.clone(),(-state[0].clone(),)
        for j in range(t-1,-1,-1):
            back,rev = counter.step(fresh,back,rev,tape[j%len(tape)])
        qe,pe = l2(back,q0),l2(-rev[0],p0)
        delta = round(300*accuracy(back,test))-initial_correct
        if t in grid:
            row=grid[t]
            check((qe,pe,delta)==(row["q_l2"],row["p_l2"],row["change_in_correct"]),"frozen grid mismatch")
        trace.append([t,qe,pe,delta])
        if max(qe,pe)>=1e-5 or abs(delta)>1:
            return {"seed":seed,"first_miss":t,"last_contiguous_pass":t-1,"trace":trace}
    return {"seed":seed,"first_miss":None,"last_contiguous_pass":stop,"trace":trace}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    args=parser.parse_args()
    folder=ROOT.parent/"theory/evidence/phase2"
    source=folder/("test_"+args.device+".json")
    blob=source.read_bytes()
    report=json.loads(blob)
    pin="19aef02de8768e14dec3f1557130b838f217e0684c6acd8f926e6a87f20270f0"
    check(tree_hash(ROOT)==report["eval_v2_sha256"]==pin and tree_hash(ROOT.parent/"eval")==PIN1,"pin mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(11173)
    limit=report["decisions"]["horizons"]["glide"]["first_failed_grid_horizon"]
    check(limit is not None,"frozen grid was right censored")
    rows=[]
    for seed,raw in zip(TEST[:3],report["horizons"]):
        grid={r["T"]:r for r in raw if r["method"]=="glide"}
        rows.append(first_failure(seed,torch.device(args.device),grid,limit))
    result={"scope":"Post-grid descriptive localization; original threshold, tape, and seeds",
            "device":args.device,"source_test_sha256":hashlib.sha256(blob).hexdigest(),
            "trace_columns":["T","q_l2","p_l2","change_in_correct"],"rows":rows,
            "first_any_seed_miss":min(r["first_miss"] for r in rows if r["first_miss"] is not None)}
    with (folder/("dense_horizon_"+args.device+".json")).open("x",encoding="utf-8") as out:
        json.dump(result,out,allow_nan=False,separators=(",",":"))
    print(args.device,result["first_any_seed_miss"],[r["first_miss"] for r in rows])


if __name__=="__main__":
    main()
