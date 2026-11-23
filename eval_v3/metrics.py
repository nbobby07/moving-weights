"""Terminal coverage, pre-verifier discrimination and resource comparisons."""
import math
from statistics import mean,median

import numpy as np
from .search import features
from .task import DEPTHS,METHODS


def online(leaf):
    edges=leaf["edge_curves"]
    return features(edges[:1])+features(edges[:max(1,len(edges)//2)])+leaf["signals"]


def draws(records,exact=False,surviving=False):
    xs,ys,groups=[],[],[]
    for r in records:
        for secret in (range(r["D"]+1) if exact else r["private_draws"]):
            for rank,leaf in enumerate(r["leaves"]):
                xs.append(online(leaf)); ys.append(int(leaf["terminal_pass"][secret]));groups.append(rank)
                if surviving and ys[-1]:
                    break
    return np.asarray(xs),np.asarray(ys),np.asarray(groups)


def fit_monitor(records):
    x,y,_=draws(records)
    mu=x.mean(axis=0); sd=x.std(axis=0)
    sd[sd<1e-8]=1.0
    z=(x-mu)/sd
    w=np.linalg.solve(z.T @ z/len(z)+np.eye(z.shape[1]),z.T @ (y-y.mean())/len(y))
    return {"mu":mu.tolist(),"sd":sd.tolist(),"w":w.tolist()}


def auc(a,y):
    positive=int(y.sum()); negative=len(y)-positive
    if not positive or not negative:
        return None
    # Average tied ranks.
    _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
    ranks=(np.cumsum(counts)-(counts-1)/2)[inverse]
    return float((ranks[y==1].sum()-positive*(positive+1)/2)/(positive*negative))


def detectors(rows,monitor,exact=False,surviving=False):
    x,y,group=draws(rows,exact,surviving)
    score=(x-monitor["mu"])/monitor["sd"]@monitor["w"]
    scores=np.column_stack((x,score))
    values=[]
    for col in scores.T:
        if not surviving:
            values.append(auc(col,y))
            continue
        total=0; weighted=0
        for g in np.unique(group):
            take=group==g
            a=auc(col[take],y[take])
            if a is not None:
                pairs=int(y[take].sum())*int((1-y[take]).sum())
                total+=pairs;weighted+=pairs*a
        values.append(weighted/total if total else None)
    first=group==0
    first_auc=[auc(v[first],y[first]) for v in scores.T]
    return {"auc":values,"max_orientation_free":max((max(a,1-a) for a in values if a is not None),default=None),
            "first_leaf_max":max((max(a,1-a) for a in first_auc if a is not None),default=None),
            "visit_rank_alone_auc":auc(group,y),"positive":int(y.sum()),"negative":int(len(y)-y.sum())}


def coverage(r):
    leaves=r["leaves"]
    hidden=r["private_draws"]
    covered=[next((i for i,l in enumerate(leaves) if l["terminal_pass"][s]),None) for s in range(r["D"]+1)]
    return {"exact_prior":mean(i is not None for i in covered),
            "sampled":mean(covered[s] is not None for s in hidden),
            "assigned_accuracy":sum(l["assigned_correct"] for l in leaves)/(8*r["D"]*len(leaves)),
            "mean_time_to_stop":mean(leaves[covered[s]]["elapsed"] if covered[s] is not None else r["elapsed"] for s in hidden)}


def smooth(rows):
    tally=[[0,0],[0,0]]
    for r in rows:
        for leaf in r["leaves"]:
            for s in r["private_draws"]:
                y=int(leaf["terminal_pass"][s])
                for edge in leaf["edge_curves"]:
                    losses=[v[0] for v in edge]
                    tally[y][0]+=int(losses[-1]<losses[0] and all(b<=a+1e-4 for a,b in zip(losses,losses[1:])))
                    tally[y][1]+=1
    return [a/b if b else 0.0 for a,b in tally]


def summarize(cells,monitor,physics):
    out={"depths":{}}
    fidelities={m:True for m in ("int32","int64")}
    exact={m:True for m in fidelities}
    all_b=True
    for depth in DEPTHS:
        by={m:cells[f"{m}_{depth}"] for m in METHODS}
        if any("failure" in c for c in by.values()):
            out["depths"][depth]={"failures":{m:c["failure"] for m,c in by.items() if "failure" in c}}
            all_b=False
            for m in exact:
                if "failure" in by[m] or "failure" in by["native"]:
                    exact[m]=fidelities[m]=False
            continue
        metrics={}
        for m,c in by.items():
            per=[coverage(r) for r in c["rows"]]
            metrics[m]={k:mean(r[k] for r in per) for k in per[0]}
            metrics[m].update(peak_gpu=max(r["memory"]["cuda_peak_allocated"] or 0 for r in c["rows"]),
                              peak_rss=max(r["memory"]["rss_peak_sampled"] for r in c["rows"]),
                              median_exhaust_s=median(r["elapsed"] for r in c["rows"]))
        detector_gate=True; monitors={}
        for name in ("native","int32","int64"):
            rows=by[name]["rows"]
            scores=detectors(rows,monitor)
            pop=detectors(rows,monitor,exact=True)
            causal=detectors(rows,monitor,surviving=True)
            curves=smooth(rows)
            monitors[name]={"sampled":scores,"exact_prior":pop,"conditional_on_rank":causal,"smooth_losing_winning":curves}
            detector_gate &= all(r["max_orientation_free"] is not None and r["max_orientation_free"]<=0.60+1e-12
                                 for r in (scores,pop,causal)) and min(curves)>=0.90
            detector_gate &= scores["first_leaf_max"] is not None and scores["first_leaf_max"]<=0.60+1e-12
        for r,n in zip(by["native"]["rows"],by["single"]["rows"]):
            if r["leaves"][0]["challenge_ones"]!=n["leaves"][0]["challenge_ones"] or sum(r["unchanged_model_accepts"])>1:
                raise ValueError("single-path or unchanged-model verifier control failed")
        for m in exact:
            for a,r,oracle in zip(by[m]["audits"],by[m]["rows"],by["native"]["rows"]):
                # Every distinct tree edge must have been physically unwound in the separate audit.
                edges={tuple(leaf["path"][:i]) for leaf in r["leaves"] for i in range(1,depth+1)}
                exact[m] &= (len(a["parent_errors"])==len(edges) and {tuple(x[0]) for x in a["parent_errors"]}==edges and
                             all(q==p==0 for _,q,p in a["parent_errors"]) and all(all(x) for x in a["bit_equal"])
                             and "first_whole_shear_lost" not in a["rounding"])
                ca,cr=coverage(r),coverage(oracle)
                fidelities[m] &= (ca["assigned_accuracy"]>=0.9 and ca["assigned_accuracy"]>=cr["assigned_accuracy"]-0.01-1e-12
                                  and ca["exact_prior"]>=cr["exact_prior"]-0.01-1e-12)
        utility=all(metrics[m]["exact_prior"]-max(1/(depth+1),metrics["single"]["exact_prior"])>=0.20-1e-12 and
                    metrics[m]["sampled"]>metrics["single"]["sampled"] for m in monitors)
        comparisons={m+"_vs_"+peer:{"exhaust_time_ratio":metrics[m]["median_exhaust_s"]/metrics[peer]["median_exhaust_s"],
                                    "time_to_stop_ratio":metrics[m]["mean_time_to_stop"]/metrics[peer]["mean_time_to_stop"],
                                    "gpu_peak_bytes_saved":metrics[peer]["peak_gpu"]-metrics[m]["peak_gpu"]}
                     for m in monitors for peer in ("sgd_cpu","adam_cpu","sgd_gpu","adam_gpu")}
        out["depths"][depth]={"methods":metrics,"monitors":monitors,"utility":utility,"signal_gate":detector_gate,
                              "system_comparisons":comparisons}
        all_b &= utility and detector_gate
    crossings={}
    for m in ("native","int32","int64"):
        for peer in ("sgd_gpu","adam_gpu"):
            wins=[d for d in DEPTHS if "methods" in out["depths"][d] and
                  out["depths"][d]["methods"][m]["peak_gpu"]<out["depths"][d]["methods"][peer]["peak_gpu"]]
            crossings[m+"_vs_"+peer]=wins
    memory_pass=all(4 in ds and 8 in ds for ds in crossings.values())
    same_quality=all("methods" in row and all(row["methods"]["int32"]["exact_prior"]>=row["methods"][p]["exact_prior"]-0.01
                     for p in ("sgd_cpu","adam_cpu","sgd_gpu","adam_gpu")) for row in out["depths"].values())
    checks=physics["cliff"]
    out.update(P3_A={"repeat_gradient_bitwise":physics["repeat"]["equal"],"exact":exact,"forward_fidelity":fidelities,
                     "native_growth":checks["pass"],"pass":all(exact.values()) and all(fidelities.values())
                     and checks["pass"] and physics["repeat"]["equal"] and physics["long_integer_pass"]},
               P3_B={"pass":all_b},P3_C={"depths_with_lower_peak_allocated":crossings,"memory_pass":memory_pass,
                     "forward_ratios":physics["speed"]["ratio"],"time_crossover":physics["copy"]["first_slower_K"],
                     "quality_matched":same_quality,"pass":memory_pass and same_quality and physics["speed"]["ratio"]["native"]<=1.05
                     and all("failure" not in c for c in cells.values())})
    return out
