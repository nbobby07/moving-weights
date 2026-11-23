"""Black-box conformance and wrong-force controls for the actual submitted adapter."""
import math

import torch

from .protocol import check, l2, make, pack, unpack


def verify(factory, device):
    q = torch.tensor([0.83,-0.56,0.31],dtype=torch.float64,device=device)
    p = q.new_tensor([0.14,0.61,-0.32])
    f = q.new_tensor([0.01,0.7,4.0])
    W = f/f.mean()
    h = 0.125
    K = q.new_tensor([[1.3,0.1,0.2],[0.1,0.9,0.05],[0.2,0.05,1.1]])
    I = torch.eye(6,device=device,dtype=q.dtype)
    J = torch.cat((I[3:],-I[:3]))
    def force(i):
        return lambda x: K@x + (i*7%19-8)*0.07
    probes = {}
    for name in ("glide","mass","generic","iso","low","high"):
        a = 3.0 if name in ("mass","generic") else 0.0
        cfg = {"h":h,"alpha":a,"strength":0.4}
        opt = make(factory,name,cfg,q,f)
        inv = 1/(1+a*W)
        mid = q + h*inv*p/2
        ends = []
        def record(x):
            ends.append(x.clone())
            return force(0)(x)
        expected_p = p-h*force(0)(mid)
        q1,(p1,) = opt.step(q,(p,),record,gate=0)
        check(len(ends)==1 and l2(ends[0],mid)<1e-13,"wrong DKD force site")
        err = max(l2(q1,mid+h*inv*expected_p/2),l2(p1,expected_p))
        def mapping(z):
            x,v=z.chunk(2)
            x,(v,)=opt.step(x,(v,),force(1),gate=0)
            return torch.cat((x,v))
        z=torch.cat((q,p))
        jac=torch.stack([(mapping(z+1e-5*e)-mapping(z-1e-5*e))/2e-5 for e in I],1)
        residual = l2(jac.T@J@jac,J)
        x,st=q.clone(),opt.init(q,p)
        for i in range(500):
            x,st=opt.step(x,st,force(i),gate=0)
        st=(-st[0],)
        fresh=make(factory,name,cfg,q,f)
        for i in range(499,-1,-1):
            x,st=fresh.step(x,st,force(i),gate=0)
        back=max(l2(x,q),l2(-st[0],p))
        check(err<1e-13 and residual<1e-8 and back<1e-11,name+" mechanics failed")
        probes[name]={"site_l2":l2(ends[0],mid),"endpoint_l2":err,"form_residual":residual,"rollback_500":back}
    # Two batches with different affine gradients at exactly the same q.
    gap=l2(force(0)(q),force(1)(q))
    check(gap>0.1,"stale-FSAL control is uninformative")
    probes["wrong_VV_Fsal_gradient_l2"]=gap
    anchor=q.new_tensor([0.1,-0.2,-0.1])
    def prior(x):
        return (W*(x-anchor).square()).sum()/2
    def energy(x):
        return x@(K@x)/2
    opt=make(factory,"generic",{"h":h,"alpha":3,"strength":0.4},anchor,f)
    y=opt.anchor_flow(q,h/2)
    yf=anchor+torch.exp(-h/2*0.4*W)*(q-anchor)
    flux=energy(q)-energy(y)
    ds=(prior(q)-prior(y)).item()
    actual,st=opt.step(q,(p,),force(3),gate=1)
    mid=yf+h/2*p/(1+3*W)
    vp=p-h*force(3)(mid)
    xf=mid+h/2*vp/(1+3*W)
    exact=anchor+torch.exp(-h/2*0.4*W)*(xf-anchor)
    anchored=max(l2(y,yf),l2(actual,exact),l2(st[0],vp))
    balance=abs((energy(y)+flux-energy(q)).item())
    check(anchored<1e-13 and balance<1e-13 and ds>0,"GENERIC anchor failed")
    probes["anchor"]={"split_error":anchored,"brake_energy_balance":balance,"entropy_increase":ds}
    energy_probes={}
    for name,mask in (("iso",torch.ones_like(f,dtype=torch.bool)),("low",f<=f.median()),("high",f>f.median())):
        o=make(factory,name,{"h":h},q,f)
        cap=p[mask].square().sum()/2
        target=0.4*cap
        end=o.brake(p,target)
        delta=(p.square()-end.square()).sum()/2
        expected=torch.where(mask,math.sqrt(0.6)*p,p)
        check(l2(end,expected)<1e-13 and abs(float(delta-target))<1e-13,"brake orientation failed")
        check(torch.equal(o.brake(torch.zeros_like(p),target*0),torch.zeros_like(p)),"zero-capacity brake")
        energy_probes[name]={"target":target.item(),"removed":delta.item(),"endpoint_l2":l2(end,expected)}
    probes["energy"]=energy_probes
    o=make(factory,"lattice",{"h":h},q)
    Q,P=pack(q.float()),pack(p.float())
    def discrete(x):
        check(x.dtype==torch.float32,"lattice loss force must use f32")
        return x.sin()+0.1
    a,ss=o.step(Q,o.init(Q,P),discrete,gate=0)
    b,(v,)=o.step(a,(-ss[0],),discrete,gate=0)
    check(torch.equal(b,Q) and torch.equal(-v,P) and not torch.equal(a,Q),"integer round trip")
    probes["lattice_one_step"]=True
    for name in ("sgdm","adamw","ewc"):
        opt=make(factory,name,{"h":0.013,"strength":0.3},q,f)
        x=q.clone()
        st=opt.init(x,p)
        native=torch.nn.Parameter(q.clone())
        base=(torch.optim.AdamW([native],lr=0.013,weight_decay=0.01,foreach=False,fused=False) if name=="adamw"
              else torch.optim.SGD([native],lr=0.013,momentum=0.9,foreach=False))
        for i in range(7):
            grad=force(i)(native.detach())
            native.grad=grad+0.3*W*(native.detach()-q) if name=="ewc" else grad
            base.step()
            x,st=opt.step(x,st,force(i),gate=0)
        probes[name]={"native_l2":l2(x,native.detach())}
        check(probes[name]["native_l2"]<1e-13,name+" baseline failed")
    return probes
