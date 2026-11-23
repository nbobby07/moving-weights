"""Functional reference adapter. All arrays in state belong to the caller."""
import torch


class RangeFault(ArithmeticError):
    pass


class Engine:
    def __init__(self,method,h,shifts):
        self.method,self.h,self.shifts=method,h,shifts
        self.diagnostics=None
        self.where=None

    def init(self,q,p):
        if self.method in ("int32","int64"):
            dtype=getattr(torch,self.method)
            out=[]
            for x,s in zip((q,p),self.shifts):
                z=x.double()*2**s
                limit=2**31-1 if dtype==torch.int32 else 2**50-1
                if not bool((z.abs()<=limit).all()):
                    raise RangeFault("initial range")
                a=z.round().to(dtype)
                if not torch.equal((a.float()*2.0**-s).view(torch.int32),x.view(torch.int32)):
                    raise RangeFault("initial state cannot be encoded losslessly")
                out.append(a)
            return tuple(out)
        if self.method=="adam":
            return q,p,torch.zeros_like(q),0
        return q,p

    def weights(self,st):
        return st[0] if st[0].dtype==torch.float32 else st[0].float()*2.0**-self.shifts[0]

    def shear(self,old,source,factor,name):
        limit=2**31-1 if old.dtype==torch.int32 else 2**50-1
        out=torch.empty_like(old)
        lost=nonzero=0
        d=self.diagnostics
        # Bound temporary double precision workspace independently of parameter count.
        for start in range(0,old.numel(),2**21):
            s=slice(start,start+2**21)
            raw=source[s].double()*factor
            inc=raw.round()
            z=old[s].double()+inc
            peak=torch.maximum(inc.abs().amax(),z.abs().amax()).item()
            if not peak<=limit:
                bad=~((inc.abs()<=limit)&(z.abs()<=limit))
                j=int(bad.to(torch.int8).argmax())
                raise RangeFault(f"{self.where} {name} index={start+j} increment={float(inc[j])} attempted={float(z[j])} limit={limit}")
            out[s].copy_(z)
            if d is not None:
                zero=(inc==0)&(raw!=0)
                counts=torch.stack((zero.sum(),(raw!=0).sum())).tolist()
                lost+=counts[0];nonzero+=counts[1]
                if counts[0] and "first_sub_lsb" not in d:
                    j=int(zero.to(torch.int8).argmax())
                    d["first_sub_lsb"]=[self.where,name,start+j,float(raw[j])]
        if d is not None:
            d["lost_components"]=d.get("lost_components",0)+lost
            d["nonzero_components"]=d.get("nonzero_components",0)+nonzero
            if lost and lost==nonzero and "first_whole_shear_lost" not in d:
                d["first_whole_shear_lost"]=[self.where,name]
        return out

    def step(self,st,grad):
        q,p=st[:2]
        h=self.h
        if self.method=="sgd":
            v=p.mul(0.9).add_(grad(q))
            return q.add(v,alpha=-h),v
        if self.method=="adam":
            v,t=st[2:]
            g=grad(q)
            m=p.lerp(g,0.1)
            v=v.mul(0.999).addcmul_(g,g,value=0.001)
            t+=1
            denom=v.sqrt().div_((1-0.999**t)**0.5).add_(1e-8)
            return q.mul(1-h*0.01).addcdiv_(m,denom,value=-h/(1-0.9**t)),m,v,t
        if self.method=="native":
            mid=q.add(p,alpha=h/2)
            pn=p.add(grad(mid),alpha=-h)
            return mid.add(pn,alpha=h/2),pn
        sq,sp=self.shifts
        drift=h/2*2.0**(sq-sp)
        mid=self.shear(q,p,drift,"drift1")
        pn=self.shear(p,grad(mid.float()*2.0**-sq),-h*2.0**sp,"kick")
        return self.shear(mid,pn,drift,"drift2"),pn


def make_engine(method,*,h,shifts):
    return Engine(method,h,shifts)
