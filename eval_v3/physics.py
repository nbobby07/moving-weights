"""Roundoff, directional tangent growth, checked integer replay and transfer timing."""
import math
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F

from .search import stamp,sync
from .task import K,Problem,check,rng

TIMES=(8,16,32,64,128,256,500)


def tape(problem,i):
    # Interleave contradictory hypotheses, slots and clean draws without any hidden secret.
    j=(1103515245*(i+71)+12345)%2147483647
    event=((j//13)%problem.depth*2+(j//113)%2)*(K+1)+i%K
    return problem.bank[event]


def pair_norm(q,p):
    return torch.sqrt(torch.linalg.vector_norm(q,dtype=torch.float64).square()+
                      torch.linalg.vector_norm(p,dtype=torch.float64).square())


def hvp(model,x,direction,batch):
    w=model.parameters(x,True)
    g=torch.autograd.grad(F.cross_entropy(model.logits(w,batch),batch[2]),w,create_graph=True)
    v=[a.view(s) for a,s in zip(direction.split(model.sizes),model.shapes)]
    h=torch.autograd.grad(g,w,grad_outputs=v)
    return torch.cat([a.reshape(-1) for a in h])


def tangent_probe(factory,model,seed):
    q=model.initial(seed);p=torch.sign(q)*2**-14
    r=rng(seed+174000)
    u,v=[torch.randn(model.n,generator=r).to(model.device) for _ in range(2)]
    n=pair_norm(u,v).item();u,v=u/n,v/n
    h=1/16; dx=u.add(v,alpha=h/2); b=Problem(seed,1,model.device).bank[0]
    vp=v.add(hvp(model,q.add(p,alpha=h/2),dx,b),alpha=-h)
    up=dx.add(vp,alpha=h/2)
    e=factory("native",h=h,shifts=[0,0]); eps=0.015625
    plus=e.step((q+eps*u,p+eps*v),lambda x:model.grad(x,b))
    minus=e.step((q-eps*u,p-eps*v),lambda x:model.grad(x,b))
    err=pair_norm((plus[0]-minus[0])/(2*eps)-up,(plus[1]-minus[1])/(2*eps)-vp).item()/pair_norm(up,vp).item()
    check(err<0.01,"tangent map disagrees with centered finite difference")
    return {"eps":eps,"relative_residual":err}


def repeat_probe(model,seed):
    q=model.initial(seed)
    b=Problem(seed,2,model.device).bank[0]
    p=torch.sign(q)*2**-14
    mid=q.add(p,alpha=1/32)
    a=model.grad(mid,b); other=model.grad(mid.clone(),b)
    counts=[int(g.count_nonzero()) for g in a.split(model.sizes)]
    same=torch.equal(a.view(torch.int32),other.view(torch.int32))
    check(same and all(counts),"repeated midpoint force differs or a weight block is inactive")
    return {"equal":same,"max_difference":(a-other).abs().max().item(),"nonzero_by_parameter_tensor":counts}


def trajectory(factory,model,seed,h,shifts,method,times):
    q0=model.initial(seed)
    p0=torch.sign(q0)*2**-14
    problem=Problem(seed,8,model.device)
    e=factory(method,h=h,shifts=shifts)
    if method!="native":
        e.diagnostics={}
    st=e.init(q0,p0)
    start=st
    norm=pair_norm(q0,p0).item()
    gen=rng(seed*733+9)
    u,v=[torch.randn(model.n,generator=gen).to(model.device) for _ in range(2)]
    scale=pair_norm(u,v).item()
    u,v=u/scale,v/scale
    growth=0.0
    local=[]
    rows=[]
    for i in range(1,max(times)+1):
        batch=tape(problem,i-1)
        if method=="native":
            dm=u.add(v,alpha=h/2)
            mid=st[0].add(st[1],alpha=h/2)
            v=v.add(hvp(model,mid,dm,batch),alpha=-h)
            u=dm.add(v,alpha=h/2)
            scale=pair_norm(u,v).item()
            check(math.isfinite(scale) and scale>0,"invalid tangent propagation")
            growth+=math.log(scale)
            local.append(math.log(scale))
            u,v=u/scale,v/scale
        e.where=["long_forward",seed,i]
        st=e.step(st,lambda x:model.grad(x,batch))
        if i in times:
            rev=factory(method,h=h,shifts=shifts)
            back=(st[0],-st[1])
            for j in range(i-1,-1,-1):
                rev.where=["long_reverse",seed,i,j]
                back=rev.step(back,lambda x:model.grad(x,tape(problem,j)))
            eq=torch.equal(back[0],start[0]) and torch.equal(-back[1],start[1])
            diffs=[torch.linalg.vector_norm(x.double()-y.double()).item()/(2**s if method!="native" else 1)
                   for x,y,s in zip((back[0],-back[1]),start,shifts)]
            rows.append({"T":i,"q_l2":diffs[0],"p_l2":diffs[1],"relative_error":math.hypot(*diffs)/norm,
                         "exact":eq,"log_gain":growth if method=="native" else None,
                         "finite_time_rate":growth/(i*h) if method=="native" else None})
    return {"seed":seed,"h":h,"method":method,"samples":rows,"local_log_stretch":local,
            "forward_rounding":e.diagnostics if method!="native" else None}


def quality(run):
    r=run["samples"]
    use=[x for x in r if 0<x["relative_error"]<0.1]
    crossed=next((s["T"] for s in r if s["relative_error"]>=1e-5),None)
    out={"first_grid_miss":crossed,"usable":len(use),"pass":False}
    if len(use)<4:
        return out
    t=np.array([s["T"] for s in use]); yy=np.log([s["relative_error"] for s in use]); g=np.array([s["log_gain"] for s in use])
    def fit(x):
        k,b=np.polyfit(x,yy,1)
        return float(k),float(1-((yy-k*x-b)**2).sum()/max(1e-30,((yy-yy.mean())**2).sum()))
    slope,re=fit(t); _,rp=fit(np.log(t))
    corr=float(np.corrcoef(g,yy)[0,1]) if np.std(g)>0 and np.std(yy)>0 else 0.0
    ok=(crossed is not None and r[0]["relative_error"]<1e-5 and slope>0 and re>=0.85 and re>=rp+0.05
        and corr>=0.8 and max(g)>=math.log(4))
    return {**out,"slope_per_step":slope,"r2_exponential":re,"r2_power":rp,"corr_loggain_logerror":corr,"pass":bool(ok)}


def speed_and_copy(factory,model,cfg,seed):
    problem=Problem(seed,2,model.device)
    q=model.initial(seed); p=torch.zeros_like(q)
    names=("native","int32","int64","sgd","adam")
    times={m:[] for m in names}
    for rep in range(5):
        for m in names[rep:]+names[:rep]:
            e=factory(m,h=cfg["rates"].get(m,cfg["rates"]["native"]),shifts=cfg["shifts"].get(m,[0,0]))
            st=e.init(q,p)
            calls=0
            def force(x):
                nonlocal calls
                calls+=1
                return model.grad(x,tape(problem,calls-1))
            for _ in range(2):
                st=e.step(st,force)
            a=stamp(model.device)
            for _ in range(12):
                st=e.step(st,force)
            times[m].append(stamp(model.device)-a)
            check(calls==14,"forward call budget")
    speed={"seconds_12_forward":times,"ratio":{m:median(x/y for x,y in zip(s,times["sgd"])) for m,s in times.items()}}
    if model.device.type!="cuda":
        return speed,{"first_slower_K":{}}
    transfers={}
    for m,buffers in (("sgd",2),("adam",3)):
        dev=[torch.empty_like(q) for _ in range(buffers)]
        host=[torch.empty_like(q,device="cpu",pin_memory=True) for _ in dev]
        pairs=[]
        for rep in range(6):
            t=stamp(model.device)
            for dst,src in zip(host,dev):
                dst.copy_(src,non_blocking=True)
            mid=stamp(model.device)
            for dst,src in zip(dev,host):
                dst.copy_(src,non_blocking=True)
            end=stamp(model.device)
            if rep:
                pairs.append([mid-t,end-mid])
        transfers[m]={"bytes_one_way":buffers*model.n*4,"D2H_H2D_seconds":pairs}
    rewinds={}
    crossing={}
    for m in names[:3]:
        e=factory(m,h=cfg["rates"]["native"],shifts=cfg["shifts"].get(m,[0,0]))
        per={}
        for k in (1,2,4,8):
            samples=[]
            for _ in range(3):
                st=e.init(q,p)
                for j in range(k):
                    st=e.step(st,lambda x:model.grad(x,tape(problem,j)))
                t=stamp(model.device)
                st=(st[0],-st[1])
                for j in range(k-1,-1,-1):
                    st=e.step(st,lambda x:model.grad(x,tape(problem,j)))
                st=(st[0],-st[1])
                samples.append(stamp(model.device)-t)
            per[k]=samples
        rewinds[m]=per
        for peer,tr in transfers.items():
            for metric,baseline in (("roundtrip",median(sum(x) for x in tr["D2H_H2D_seconds"])),
                                    ("restore",median(x[1] for x in tr["D2H_H2D_seconds"]))):
                crossing[f"{m}_vs_{peer}_{metric}"]=next((k for k,ts in per.items() if median(ts)>baseline),None)
    return speed,{"transfers":transfers,"reverse_seconds":rewinds,"first_slower_K":crossing}


def physics(factory,model,cfg,seeds,smoke=False):
    repeat=repeat_probe(model,seeds[0])
    base=cfg["rates"]["native"]
    runs=[]; ok=True
    for seed in seeds[:1 if smoke else 2]:
        triple=[]
        for h in (base/2,base,base*2):
            run=trajectory(factory,model,seed,h,[0,0],"native",(2,4,8,16) if smoke else TIMES)
            run["fit"]=quality(run);runs.append(run);triple.append(run["fit"])
        lo,_,hi=[r["first_grid_miss"] for r in triple]
        ok &= triple[2]["pass"] and hi is not None and (lo is None or hi<lo)
    longs={}
    for m in ("int32","int64"):
        engine=factory(m,h=1,shifts=[0,0]);engine.where=["overflow_control"]
        lim=2**31-1 if m=="int32" else 2**50-1
        z=torch.tensor([lim],dtype=getattr(torch,m),device=model.device)
        try:
            engine.step((z,torch.full_like(z,4)),lambda x:torch.zeros_like(x))
        except ArithmeticError:
            pass
        else:
            raise ValueError("lattice silently accepted an out-of-range shear")
        try:
            longs[m]=trajectory(factory,model,seeds[0],base,cfg["shifts"][m],m,(16,) if smoke else (500,))
        except ArithmeticError as exc:
            longs[m]={"failure":str(exc)}
    speed,copy=speed_and_copy(factory,model,cfg,seeds[0])
    return {"repeat":repeat,"tangent_probe":tangent_probe(factory,model,seeds[0]),"native_runs":runs,"cliff":{"pass":bool(ok)},"long_integer":longs,
            "long_integer_pass":all("failure" not in r and r["samples"][0]["exact"]
                                    and "first_whole_shear_lost" not in r["forward_rounding"] for r in longs.values()),
            "speed":speed,"copy":copy}
