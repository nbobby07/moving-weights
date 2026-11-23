"""Causal block or terminal feedback, policy controls, repair budgets and f32 horizons."""
import sys
from statistics import mean, pstdev

import torch
from eval.bench_continual import loss

from .protocol import (ForceCounter, HORIZONS, accuracy, batches, check, data, episode_tape, finite_phase,
                       id_buffer, l2, make, pack, unpack)


def validate(factory, prepared, seeds, device):
    clean_scores = {name: [] for name in ("glide", "adamw")}
    configs = {"glide": (0.06,0.12,0.18), "adamw": (0.0025,0.005,0.01)}
    receipts = []
    for p, seed in zip(prepared, seeds):
        tape = episode_tape(seed, "a", True, device)
        receipts.append({"clean": episode_tape(seed,"a",False,device).receipts, "dirty": tape.receipts})
    for name, grid in configs.items():
        for h in grid:
            scores = []
            for p, seed in zip(prepared, seeds):
                for regime in ("a","b"):
                    tape = episode_tape(seed,regime,False,device)
                    q = p[0].clone()
                    opt = make(factory, name, {"h":h}, q)
                    state = opt.init(q, torch.zeros_like(q))
                    ctr = ForceCounter()
                    for event in tape.arrivals:
                        q,state = ctr.step(opt,q,state,tape.fetch(event))
                    finite_phase(q,state)
                    scores.append(accuracy(q,p[2][1]))
            clean_scores[name].append({"h":h,"clean_live":scores})
    return {"rates":clean_scores,"audit_receipts":receipts}


def choose(val):
    rates = {name:max(grid,key=lambda r:mean(r["clean_live"]))["h"] for name,grid in val["rates"].items()}
    def rank(t):
        clean = sum(any(x>t for x in r["clean"]) for r in val["audit_receipts"])
        correct = sum(next((i for i,x in enumerate(r["dirty"]) if x>t),None)==2 for r in val["audit_receipts"])
        return -clean, correct, t
    return {**rates,"threshold":max((0.04,0.08,0.12),key=rank)}


POLICIES = {"rewind":("glide","rewind"), "glide_continue":("glide","continue"), "glide_skip":("glide","skip"),
            "adam_skip":("adamw","skip"), "adam_continue":("adamw","continue"),
            "always_skip":("adamw","always")}


def episode(factory, policy, cfg, regime, tape, prep, canary, continuation, fast_limit):
    method, action = POLICIES[policy]
    q0, _, tests = prep[:3]
    q = q0.clone()
    n = len(tape.arrivals)
    params = {"h":cfg[method]}
    opt = make(factory,method,params,q)
    state = opt.init(q,torch.zeros_like(q))
    counter, log = ForceCounter(), id_buffer()
    # Audit targets are invisible to the policy, which receives a receipt or terminal bit.
    mark = 16 if regime == "a" else 0
    target_q, target_p = q.clone(), state[0].clone()
    initial_canary = accuracy(q,canary) if regime == "b" else None
    stopped, alarm, cut = action == "always", None, None
    forward = replay = changed = 0
    qerr = perr = raw_at_alarm = None
    verifier = None
    first_fast_warning = None
    for i,event in enumerate(tape.arrivals):
        if not stopped:
            before = q
            batch = tape.fetch(event)
            if regime == "a" and first_fast_warning is None and loss(q,*batch).item() > fast_limit:
                first_fast_warning = i
            q,state = counter.step(opt,q,state,batch)
            forward += 1
            if action == "rewind":
                log.append(event)
            if i >= mark and alarm is None:
                changed += int(not torch.equal(q,before))
        completed = i + 1
        if completed == mark:
            target_q,target_p = q.clone(),state[0].clone()
        if regime == "a":
            fail = completed % 8 == 0 and tape.receipts[completed//8-1] > cfg["threshold"]
        elif completed == n:
            verifier = accuracy(q,canary)
            fail = round(len(canary[1]) * (initial_canary - verifier)) >= 9
        else:
            fail = False
        if fail and alarm is None:
            alarm = completed
            raw_at_alarm = accuracy(q,tests[0])
            if action in ("rewind","skip"):
                stopped = True
            if action == "rewind":
                cut = completed-8 if regime == "a" else 0
                opt = make(factory,method,params,q)
                state = (-state[0],)
                for j in range(completed-1,cut-1,-1):
                    q,state = counter.step(opt,q,state,tape.fetch(log[j]))
                    replay += 1
                state = (-state[0],)
                qerr,perr = l2(q,target_q),l2(state[0],target_p)
    finite_phase(q,state)
    endpoint, live = accuracy(q,tests[0]), accuracy(q,tests[1])
    # Every arm spends all remaining gradient credits on the same ordered clean-data prefix.
    repair = 2*n-counter.calls
    check(repair >= 0, "policy exceeded budget")
    for batch in continuation[:repair]:
        q,state = counter.step(opt,q,state,batch)
    check(counter.calls == 2*n == forward+replay+repair,"policy budget mismatch")
    finite_phase(q,state)
    s = q0.element_size()*q0.numel()
    return {"alarm_after_commits":alarm,"estimated_cut":cut,"changed_post_onset_before_alarm":changed,
            "target_acc":accuracy(target_q,tests[0]),"raw_acc_at_alarm":raw_at_alarm,
            "passive_loss_alarm_preupdate_index":first_fast_warning,
            "endpoint_acc":endpoint,"live_acc":live,"equal_budget_acc":accuracy(q,tests[0]),
            "q_l2_to_true_origin":qerr,"p_l2_to_true_origin":perr,
            "verifier_pre":initial_canary,"verifier_terminal":verifier,
            "calls":{"forward":forward,"replay":replay,"clean_repair":repair,"total":counter.calls},
            "memory_bytes":{"peak_id_payload":len(log)*log.itemsize,"weights_checkpoint":s,
                            "allocated_id_array_including_header":sys.getsizeof(log),
                            "full_state_checkpoint":2*s if method=="glide" else 3*s+8,
                            "immutable_external_event_payload":tape.payload_bytes(),
                            "optimizer_history":0}}


def trial(factory,cfg,prepared,seed,device):
    # Separate from calibration, validation, training and both scoring test sets.
    canary = data(seed*100+61,0,300,device)
    continuation = batches(data(seed*100+62,0,384,device),96,seed*100+63)
    calibration = batches(data(seed*100+64,1,384,device),32,seed*100+65)
    losses = [loss(prepared[0],*b).item() for b in calibration]
    fast_limit = mean(losses)+4*pstdev(losses)
    out = {}
    for regime in ("a","b"):
        out[regime] = {}
        for label,dirty in (("clean",False),("dirty",True)):
            tape = episode_tape(seed,regime,dirty,device)
            out[regime][label] = {"receipts":tape.receipts,"passive_loss_threshold":fast_limit if regime=="a" else None,
                "policies":{key:episode(factory,key,cfg,regime,tape,prepared,canary,continuation,fast_limit)
                            for key in POLICIES}}
    return out


def verdict(rows):
    result = {}
    for regime in ("a","b"):
        clean = [r[regime]["clean"]["policies"] for r in rows]
        dirty = [r[regime]["dirty"]["policies"] for r in rows]
        pairs = [m["rewind"] for m in dirty]
        at = 24 if regime == "a" else 32
        alarmed = [m for m in pairs if m["alarm_after_commits"] is not None]
        good = sum(m["alarm_after_commits"] >= at and m["changed_post_onset_before_alarm"] >= 8 and
                   m["q_l2_to_true_origin"] < 1e-11 and m["p_l2_to_true_origin"] < 1e-11 and
                   abs(m["endpoint_acc"]-m["target_acc"]) <= 0.01 for m in alarmed)
        endpoint = mean(m["rewind"]["endpoint_acc"]-m["adam_skip"]["endpoint_acc"] for m in dirty)
        own_skip = mean(m["rewind"]["endpoint_acc"]-m["glide_skip"]["endpoint_acc"] for m in dirty)
        equal = mean(m["rewind"]["equal_budget_acc"]-m["adam_skip"]["equal_budget_acc"] for m in dirty)
        utility = mean(m["rewind"]["live_acc"]-m["always_skip"]["live_acc"] for m in clean)
        material = sum(m["raw_acc_at_alarm"] is not None and
                       round(300*(m["target_acc"]-m["raw_acc_at_alarm"])) >= 6 for m in pairs)
        checks = {"delayed_true_origin_recoveries":good>=11,
                  "clean_alarms":sum(m["rewind"]["alarm_after_commits"] is not None for m in clean)<=1,
                  "outcome_verified_damage_at_least_2pp_in_11":material>=11,
                  "immediate_advantage_over_adam_skip_2pp":endpoint>=0.02-1e-12,
                  "immediate_advantage_over_glide_skip_2pp":own_skip>=0.02-1e-12,
                  "equal_budget_advantage_over_skip_2pp":equal>=0.02-1e-12,
                  "clean_utility_over_always_skip_1pp":utility>=0.01-1e-12}
        if regime=="a":
            early = [m["adam_continue"]["passive_loss_alarm_preupdate_index"] for m in dirty]
            checks["at_most_one_early_ungated_adam_loss_warning"] = sum(x is not None and x<24 for x in early)<=1
        result[regime] = {"pass":all(checks.values()),"checks":checks,"true_origin_recoveries":good,
                          "outcome_damaged":material,"clean_utility":utility,"endpoint_advantage":endpoint,
                          "equal_budget_advantage":equal,"within_glide_advantage":own_skip}
    return result


def horizon(factory, prep):
    q0 = prep[0].float()
    p0 = (0.02*torch.cos(torch.arange(q0.numel(),device=q0.device,dtype=torch.float64))).float()
    stream = [(x.float(),y) for task in prep[3] for x,y in task]
    test = (prep[2][0][0].float(),prep[2][0][1])
    clean_count = round(300*accuracy(q0,test))
    ends = {}
    rows = []
    for name in ("glide","lattice"):
        q,p = (q0,p0) if name=="glide" else (pack(q0),pack(p0))
        start_q,start_p = q.clone(),p.clone()
        opt = make(factory,name,{"h":0.12},q)
        state = opt.init(q,p)
        ctr = ForceCounter()
        for i in range(1,501):
            q,state = ctr.step(opt,q,state,stream[(i-1)%len(stream)])
            if i not in HORIZONS:
                continue
            rq,rs = q.clone(), (-state[0].clone(),)
            fresh = make(factory,name,{"h":0.12},rq)
            for j in range(i-1,-1,-1):
                rq,rs = ctr.step(fresh,rq,rs,stream[j%len(stream)])
            rp = -rs[0]
            exact = torch.equal(rq,start_q) and torch.equal(rp,start_p)
            decoded = (unpack(rq),unpack(rp)) if name=="lattice" else (rq,rp)
            forward = (unpack(q),unpack(state[0])) if name=="lattice" else (q,state[0])
            if name=="glide":
                ends[i] = tuple(x.clone() for x in forward)
            qe,pe = l2(decoded[0],q0),l2(decoded[1],p0)
            dc = round(300*accuracy(decoded[0],test))-clean_count
            rows.append({"method":name,"T":i,"q_l2":qe,"p_l2":pe,"change_in_correct":dc,
                         "within_tolerance":max(qe,pe)<1e-5 and abs(dc)<=1,"state_exact":exact,
                         "fwd_distance_from_native_q":l2(forward[0],ends[i][0]),
                         "fwd_distance_from_native_p":l2(forward[1],ends[i][1]),
                         "forward_acc":accuracy(forward[0],test),"phase_bytes":2*q.numel()*q.element_size()})
        check(ctr.calls==500+sum(HORIZONS),"horizon call count mismatch")
    return rows


def horizon_verdict(seeds):
    result = {}
    for name in ("glide","lattice"):
        passes = [all(r["within_tolerance"] if name=="glide" else r["state_exact"]
                      for seed in seeds for r in seed if r["method"]==name and r["T"]==t) for t in HORIZONS]
        first = next((j for j,v in enumerate(passes) if not v),len(passes))
        result[name] = {"passes_on_grid":passes,"T_max_grid_prefix":HORIZONS[first-1] if first else 0,
                        "first_failed_grid_horizon":HORIZONS[first] if first<len(passes) else None,
                        "right_censored":first==len(passes)}
    return result
