"""Reduce the locked Phase 3 reference cells independently to a compact audit JSON."""
from bisect import bisect_left, bisect_right
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean,median

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"theory/evidence/phase3"
PIN="4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d"
LOCK="4f89ac460880140573f7746766eb4da15dd9b3a0d15fb998e15ea124a002d37a"
N=10_517_504


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def auc(values,labels):
    neg=sorted(v for v,y in zip(values,labels) if not y)
    pos=[v for v,y in zip(values,labels) if y]
    if not neg or not pos:
        return None
    return sum(bisect_left(neg,v)+bisect_right(neg,v) for v in pos)/(2*len(neg)*len(pos))


def part(report,key):
    desc=report["parts"][key]
    path=FOLDER/desc["path"]
    assert digest(path)==desc["sha256"]
    sub=read(path)
    assert sub["eval_sha256"]==PIN and sub["validation_sha256"]==LOCK
    assert sub["parameters"]==N and sub["device"]=="cuda" and sub["seeds"]==report["seeds"]
    return sub


def leaf_features(edges):
    def aggregate(es):
        initial=[e[0] for e in es];final=[e[-1] for e in es]
        jumps=[s[0]-p[0] for e in es for p,s in zip(e,e[1:])]
        return [mean(x[0] for x in initial),mean(x[0] for x in final),final[-1][0],
                max(x[0] for x in initial),max(jumps),mean(jumps),mean(x[1] for x in final),
                mean(x[2] for x in initial),final[-1][2]]
    return aggregate(edges[:1])+aggregate(edges[:max(1,len(edges)//2)])+aggregate(edges)


def monitors(rows,model,scheme):
    # Expand membership only after features have been constructed from the clean probes.
    signals=[];labels=[];ranks=[]
    for r in rows:
        features=[]
        for leaf in r["leaves"]:
            x=leaf_features(leaf["edge_curves"])
            # Use the recorded accumulation order for the ridge score and tied-rank comparisons.
            assert np.allclose(x[-9:],leaf["signals"],atol=1e-12,rtol=0)
            def recorded(es):
                f=leaf_features(es)[-9:]
                f[0]=sum(e[0][0] for e in es)/len(es)
                f[1]=sum(e[-1][0] for e in es)/len(es)
                jumps=[b[0]-a[0] for e in es for a,b in zip(e,e[1:])]
                f[5]=sum(jumps)/len(jumps)
                f[6]=sum(e[-1][1] for e in es)/len(es)
                f[7]=sum(e[0][2] for e in es)/len(es)
                return f
            es=leaf["edge_curves"]
            x=recorded(es[:1])+recorded(es[:max(1,len(es)//2)])+recorded(es)
            features.append(x)
        for s in (range(r["D"]+1) if scheme=="exact_prior" else r["private_draws"]):
            for i,leaf in enumerate(r["leaves"]):
                verdict=leaf["terminal_pass"][s]
                signals.append(features[i]);labels.append(verdict);ranks.append(i)
                if scheme=="conditional_on_rank" and verdict:
                    break
    mat=np.array(signals); ridge=((mat-model["mu"])/model["sd"])@model["w"]
    values=[]
    for column in np.column_stack((mat,ridge)).T:
        if scheme!="conditional_on_rank":
            values.append(auc(column,labels))
        else:
            scores=[];weights=[]
            for rank in set(ranks):
                idx=[i for i,n in enumerate(ranks) if n==rank]
                lab=[labels[i] for i in idx]
                value=auc(column[idx],lab)
                if value is not None:
                    scores.append(value); weights.append(sum(lab)*(len(lab)-sum(lab)))
            values.append(sum(a*b for a,b in zip(scores,weights))/sum(weights) if weights else None)
    return values


def main():
    assert digest(FOLDER/"validate.json")==LOCK
    report=read(FOLDER/"test.json")
    validation=read(FOLDER/"validate.json")
    assert report["eval_sha256"]==PIN and report["validation_sha256"]==LOCK
    assert set(report["seeds"]).isdisjoint(validation["seeds"]) and report["chosen"]==validation["chosen"]
    phys=part(report,"physics")["physics"]
    assert phys["repeat"]["equal"] and phys["repeat"]["max_difference"]==0
    audit={"eval_sha256":PIN,"validation_sha256":LOCK,"test_sha256":digest(FOLDER/"test.json"),
           "hardware":report["hardware"],"parameters":N,"source_tree":report["adapter_tree_sha256"],
           "decisions":{k:report["decisions"][k] for k in ("P3_A","P3_B","P3_C")},"cells":{},"numerics":{}}
    b_pass=True
    for depth in (1,2,4,8):
        edges=depth+depth*(depth+1)//2
        codes=[tuple([0]*depth)]+[tuple(int(i==j) for i in range(depth)) for j in reversed(range(depth))]
        decision=report["decisions"]["depths"][str(depth)]
        for method in ("native","int32","int64","sgd_gpu","adam_gpu","sgd_cpu","adam_cpu","single"):
            key=f"{method}_{depth}"; cell=part(report,key)["cell"]
            if "failure" in cell:
                audit["cells"][key]={"failure":cell["failure"]};b_pass=False;continue
            rows=cell["rows"]
            assert [r["seed"] for r in rows]==report["seeds"]
            exact=[]; sampled=[];accuracy=[];stop=[]
            for r in rows:
                reversible=method in ("native","int32","int64")
                mult=2 if method!="int64" else 4
                state=(12*N+8 if method.startswith("adam") else mult*4*N)
                assert r["n"]==N and r["active_state_bytes"]==state
                assert r["forward"]==(2*depth if method=="single" else 2*edges)
                assert r["reverse"]==(2*(edges-depth) if reversible else 0)
                assert r["peak_snapshot_bytes"]==(state*depth if "_" in method else 0)
                assert r["peak_tape_payload"]==(16*depth if reversible else 0)
                base=torch.randint(2,(depth,),generator=torch.Generator().manual_seed(r["seed"]*101+1)).tolist()
                solved={}; hits=0
                for i,l in enumerate(r["leaves"]):
                    assert tuple(l["path"])==codes[i]
                    for j,c in enumerate(codes):
                        right=[v if bit^b else 8-v for v,bit,b in zip(l["challenge_ones"],base,c)]
                        accept=min(right)>=7
                        assert accept==l["terminal_pass"][j]
                        if accept and j not in solved: solved[j]=l["elapsed"]
                        if i==j:
                            assert sum(right)==l["assigned_correct"]
                            hits+=sum(right)
                exact.append(len(solved)/(depth+1));sampled.append(mean(s in solved for s in r["private_draws"]))
                accuracy.append(hits/(8*depth*len(r["leaves"])))
                stop.append(mean(solved.get(s,r["elapsed"]) for s in r["private_draws"]))
            c={"exact_prior":mean(exact),"sampled_solve":mean(sampled),"assigned_accuracy":mean(accuracy),
               "peak_gpu_allocated":max(r["memory"]["cuda_peak_allocated"] for r in rows),
               "peak_gpu_reserved":max(r["memory"]["cuda_peak_reserved"] for r in rows),
               "peak_scoped_host_rss":max(r["memory"]["rss_peak_sampled"] for r in rows),
               "os_lifetime_peak_host_rss":max(r["memory"]["os_process_lifetime_peak_rss"] or 0 for r in rows),
               "peak_host_checkpoint_bytes":rows[0]["peak_snapshot_bytes"] if method.endswith("_cpu") else 0,
               "peak_checkpoint_bytes":rows[0]["peak_snapshot_bytes"],"active_bytes":rows[0]["active_state_bytes"],
               "tape_bytes":rows[0]["peak_tape_payload"],"external_bank_bytes":rows[0]["external_bank_bytes"],
               "transfer_bytes":rows[0]["host_transfer_tensor_bytes"],"forward":rows[0]["forward"],"reverse":rows[0]["reverse"],
               "median_exhaust_s":median(r["elapsed"] for r in rows),"mean_stop_s":mean(stop)}
            archived=decision.get("methods",{}).get(method)
            if archived:
                for k,l in (("exact_prior","exact_prior"),("assigned_accuracy","assigned_accuracy"),("peak_gpu_allocated","peak_gpu"),("mean_stop_s","mean_time_to_stop")):
                    assert math.isclose(c[k],archived[l],rel_tol=1e-12,abs_tol=1e-12)
            if cell["audits"]:
                checks=[(errs,bit) for a in cell["audits"] for errs,bit in zip(a["parent_errors"],a["bit_equal"])]
                assert len(checks)==len(rows)*edges
                c.update(parent_edges=len(checks),bit_exact_parents=sum(all(b) for _,b in checks),
                         max_parent_q_l2=max(e[1] for e,_ in checks),max_parent_p_l2=max(e[2] for e,_ in checks))
                if method.startswith("int"):
                    diag=[a["rounding"] for a in cell["audits"]]
                    c["sub_lsb_fraction"]=sum(v["lost_components"] for v in diag)/max(1,sum(v["nonzero_components"] for v in diag))
                    c["first_sub_lsb"]=diag[0].get("first_sub_lsb")
                if archived:
                    mon=decision["monitors"][method]
                    for name in ("sampled","exact_prior","conditional_on_rank"):
                        obs=monitors(rows,report["chosen"]["monitor"],name)
                        assert all((u is None and v is None) or math.isclose(u,v,abs_tol=2e-10) for u,v in zip(obs,mon[name]["auc"]))
                    c["worst_detector_auc"]=max(mon["sampled"]["first_leaf_max"],*(mon[n]["max_orientation_free"] for n in ("sampled","exact_prior","conditional_on_rank")))
                    c["smooth_losing_winning"]=mon["smooth_losing_winning"]
                    c["history_only_auc"]=mon["conditional_on_rank"]["visit_rank_alone_auc"]
                    good=[0,0];total=[0,0]
                    for r in rows:
                        for leaf in r["leaves"]:
                            wins=sum(leaf["terminal_pass"][s] for s in r["private_draws"])
                            for edge in leaf["edge_curves"]:
                                clean=edge[0][0]>edge[-1][0] and max(b[0]-a[0] for a,b in zip(edge,edge[1:]))<=1e-4
                                for outcome,weight in enumerate((len(r["private_draws"])-wins,wins)):
                                    total[outcome]+=weight
                                    good[outcome]+=weight*clean
                    assert [g/t for g,t in zip(good,total)]==c["smooth_losing_winning"]
                    b_pass &= min(c["smooth_losing_winning"])>=0.9 and c["worst_detector_auc"]<=0.60+1e-12
            audit["cells"][key]=c
        if "methods" in decision:
            single=audit["cells"][f"single_{depth}"]
            for name in ("native","int32","int64"):
                c=audit["cells"][f"{name}_{depth}"]
                b_pass &= c["sampled_solve"]>single["sampled_solve"] and c["exact_prior"]-max(single["exact_prior"],1/(1+depth))>=0.20-1e-12
    assert b_pass==report["decisions"]["P3_B"]["pass"]
    for r in phys["native_runs"]:
        assert all(math.isclose(sum(r["local_log_stretch"][:x["T"]]),x["log_gain"],rel_tol=1e-12,abs_tol=1e-12) for x in r["samples"])
        use=[x for x in r["samples"] if 0<x["relative_error"]<0.1]
        if len(use)>=4:
            t=np.array([x["T"] for x in use]); y=np.log([x["relative_error"] for x in use])
            for xs,k in ((t,"r2_exponential"),(np.log(t),"r2_power")):
                design=np.column_stack([np.ones(len(xs)),xs])
                beta=np.linalg.lstsq(design,y,rcond=None)[0]
                r2=1-np.sum((y-design@beta)**2)/np.sum((y-y.mean())**2)
                assert math.isclose(r2,r["fit"][k],abs_tol=1e-10)
        assert r["fit"]["first_grid_miss"]==next((x["T"] for x in r["samples"] if x["relative_error"]>=1e-5),None)
    phys_speed={k:median(x/y for x,y in zip(ts,phys["speed"]["seconds_12_forward"]["sgd"])) for k,ts in phys["speed"]["seconds_12_forward"].items()}
    assert phys_speed==phys["speed"]["ratio"]
    audit["numerics"]={"repeat":phys["repeat"],"tangent_probe":phys["tangent_probe"],"native_fits":[
        {"seed":r["seed"],"h":r["h"],"fit":r["fit"],"T500":r["samples"][-1]} for r in phys["native_runs"]],
        "long_lattices":phys["long_integer"],"forward_seconds_12":phys["speed"]["seconds_12_forward"],"PCIe":phys["copy"]}
    with (FOLDER/"reference_audit.json").open("x",encoding="utf-8") as f:
        json.dump(audit,f,indent=2,allow_nan=False)


if __name__=="__main__":
    main()
