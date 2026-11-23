"""Iterative DFS. The timed reversible traversal retains no parent tensors."""
from array import array
import os
import threading
import time

import psutil
import torch

from .task import K, Problem, check


def sync(device):
    if device.type=="cuda":
        torch.cuda.synchronize(device)


def stamp(device):
    sync(device)
    return time.perf_counter()


def nbytes(state):
    return sum(v.numel()*v.element_size() if torch.is_tensor(v) else 8 for v in state)


class Memory:
    def __init__(self,dev):
        self.dev=dev
        self.proc=psutil.Process(os.getpid())
        self.stop=threading.Event()
        self.samples=0

    def sample(self):
        rss=self.proc.memory_info().rss
        self.peak=max(self.peak,rss)
        self.samples+=1
        if self.dev.type=="cuda":
            free,total=torch.cuda.mem_get_info(self.dev)
            self.used=max(self.used,total-free)

    def poll(self):
        while not self.stop.wait(0.003):
            self.sample()

    def __enter__(self):
        self.start=self.peak=self.proc.memory_info().rss
        self.used=0
        if self.dev.type=="cuda":
            torch.cuda.reset_peak_memory_stats()
        self.sample()
        self.thread=threading.Thread(target=self.poll,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        info=self.proc.memory_info()
        return {"rss_start":self.start,"rss_peak_sampled":self.peak,"rss_samples":self.samples,
                "os_process_lifetime_peak_rss":getattr(info,"peak_wset",None),
                "cuda_peak_allocated":torch.cuda.max_memory_allocated() if self.dev.type=="cuda" else None,
                "cuda_peak_reserved":torch.cuda.max_memory_reserved() if self.dev.type=="cuda" else None,
                "device_wide_used_sampled":self.used if self.dev.type=="cuda" else None}


def mode(method):
    if method=="single":
        return "native"
    return method.split("_")[0]


def copy_state(state,to,device):
    def transfer(x):
        if not torch.is_tensor(x):
            return x
        if to=="host":
            out=torch.empty(x.shape,dtype=x.dtype,device="cpu",pin_memory=device.type=="cuda")
            return out.copy_(x,non_blocking=True)
        return x.to(device,non_blocking=True,copy=True)
    result=tuple(transfer(x) for x in state)
    sync(device)
    return result


def features(edges):
    first=[e[0][0] for e in edges]
    last=[e[-1][0] for e in edges]
    changes=[b[0]-a[0] for e in edges for a,b in zip(e,e[1:])]
    return [sum(first)/len(first),sum(last)/len(last),last[-1],max(first),max(changes),
            sum(changes)/len(changes),sum(e[-1][1] for e in edges)/len(edges),
            sum(e[0][2] for e in edges)/len(edges),edges[-1][-1][2]]


def search(factory,model,seed,depth,method,h,shifts,*,audit=False):
    prob=Problem(seed,depth,model.device)
    engine=factory(mode(method),h=h,shifts=shifts)
    initial=model.initial(seed)
    initial_ones=prob.query(model,initial)
    state=engine.init(initial,torch.zeros_like(initial))
    del initial
    active=nbytes(state)
    reversible=method in ("native","int32","int64","single")
    offload=method.endswith("_cpu")
    saved={}
    parents=[]
    stats={}
    engine.diagnostics=stats if audit else None
    ids=array("q")
    traces=[]
    leaves=[]
    back_errors=[]
    bit_equal=[]
    forward=reverse=0
    peak_tape=peak_tape_alloc=peak_snapshot=peak_parent_count=transfer_bytes=0
    copy_seconds=0.0
    start=stamp(model.device)

    def step(event,direction,context):
        nonlocal state,forward,reverse
        engine.where=context
        calls=0
        def grad(x):
            nonlocal calls
            calls+=1
            return model.grad(x,prob.bank[event])
        state=engine.step(state,grad)
        check(calls==1,"each step must call the supplied batch gradient exactly once")
        if direction=="f":
            forward+=1
        else:
            reverse+=1

    def backtrack(level,old):
        nonlocal state,reverse,copy_seconds,transfer_bytes
        if reversible:
            state=(state[0],-state[1])
            for i in range(len(old)-1,level-1,-1):
                for _ in range(K):
                    event=ids.pop()
                    step(event,"r",["reverse",list(old[:i+1]),event])
                if audit:
                    parent=parents.pop()
                    errs=[]
                    bits=[]
                    for actual,desired,s in zip((state[0],-state[1]),parent,shifts):
                        bits.append(torch.equal(actual.view(torch.uint8),desired.view(torch.uint8)))
                        if torch.equal(actual,desired):
                            errs.append(0.0)
                        else:
                            diff=actual.double()-desired.double()
                            if actual.dtype!=torch.float32:
                                diff.mul_(2.0**-s)
                            errs.append(torch.linalg.vector_norm(diff).item())
                    back_errors.append([list(old[:i+1]),*errs])
                    bit_equal.append(bits)
            state=(state[0],-state[1])
        else:
            checkpoint=saved.pop(level)
            if offload:
                t=stamp(model.device)
                state=copy_state(checkpoint,"gpu",model.device)
                copy_seconds+=stamp(model.device)-t
                transfer_bytes+=sum(v.numel()*v.element_size() for v in state if torch.is_tensor(v))
            else:
                state=checkpoint
        del traces[level:]

    old=()
    for code in (prob.codes[:1] if method=="single" else prob.codes):
        common=next((i for i,(a,b) in enumerate(zip(old,code)) if a!=b),min(len(old),len(code)))
        if old:
            backtrack(common,old)
        for level in range(common,depth):
            bit=code[level]
            if not reversible and not any(code[:level+1]):
                # Only unresolved binary forks need a checkpoint. Immutable GPU snapshots need no copy.
                if offload:
                    t=stamp(model.device)
                    saved[level]=copy_state(state,"host",model.device)
                    copy_seconds+=stamp(model.device)-t
                    transfer_bytes+=sum(v.numel()*v.element_size() for v in state if torch.is_tensor(v))
                else:
                    saved[level]=state
                peak_snapshot=max(peak_snapshot,sum(nbytes(st) for st in saved.values()))
                peak_parent_count=max(peak_parent_count,len(saved))
            if audit:
                parents.append(tuple(v.clone() for v in state))
            origin=(level*2+bit)*(K+1)
            trace=[]
            if not audit:
                trace.append(model.observe(engine.weights(state),prob.bank[origin+K]))
            for k in range(K):
                step(origin+k,"f",["forward",list(code[:level+1]),k])
                if reversible and method!="single":
                    ids.append(origin+k)
                if not audit:
                    trace.append(model.observe(engine.weights(state),prob.bank[origin+K]))
            traces.append(trace)
            peak_tape=max(peak_tape,len(ids)*ids.itemsize)
            peak_tape_alloc=max(peak_tape_alloc,ids.__sizeof__())
        if not audit:
            ones=prob.query(model,engine.weights(state))
            leaves.append({"path":list(code),"challenge_ones":ones,"signals":features(traces),
                           "edge_curves":list(traces),"terminal_pass":[prob.verifier(ones,s) for s in range(depth+1)],
                           "assigned_correct":sum(v if y else 8-v for y,v in zip(prob.answers[len(leaves)],ones)),
                           "elapsed":stamp(model.device)-start,"forward":forward,"reverse":reverse})
        old=code
    if audit:
        backtrack(0,old)
    elapsed=stamp(model.device)-start
    return {"seed":seed,"D":depth,"method":method,"n":model.n,"K":K,"forward":forward,"reverse":reverse,
            "elapsed":elapsed,"active_state_bytes":active,"peak_tape_payload":peak_tape,
            "peak_id_array_allocated":peak_tape_alloc,"external_bank_bytes":sum(nbytes(b) for b in prob.bank.values()),
            "peak_parent_count":peak_parent_count,"peak_snapshot_bytes":peak_snapshot,
            "host_transfer_tensor_bytes":transfer_bytes,"synchronized_transfer_s":copy_seconds,
            "private_draws":prob.private_draws(),"unchanged_model_accepts":[prob.verifier(initial_ones,s) for s in range(depth+1)],
            "leaves":leaves,"parent_errors":back_errors,"bit_equal":bit_equal,"rounding":stats}
