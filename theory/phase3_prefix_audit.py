"""Post-test localization: replay the prefix immediately before each long-tape range guard."""
import hashlib
import json
import re

from eval_v3.physics import trajectory
from eval_v3.reference import make_engine
from eval_v3.run import configure
from eval_v3.task import ROOT,Transformer,tree_hash


def main():
    folder=ROOT.parent/"theory/evidence/phase3"
    p=folder/"test_parts/physics.json"
    report=json.loads(p.read_bytes())
    receipt=(folder/"validate.json").read_bytes()
    assert hashlib.sha256(receipt).hexdigest()==report["validation_sha256"]
    assert tree_hash(ROOT)==report["eval_sha256"]=="4cf1e4d5208bba25b42f2a34f92205044e3ccf4804b1512e2c49d5fc2903195d"
    cfg=json.loads(receipt)["chosen"]
    model=Transformer(configure("cuda"))
    rows={}
    for m in ("int32","int64"):
        fault=report["physics"]["long_integer"][m].get("failure","")
        match=re.search(r"\['long_forward', (\d+), (\d+)\]",fault)
        if match:
            s,t=map(int,match.groups())
            row=trajectory(make_engine,model,s,cfg["rates"]["native"],cfg["shifts"][m],m,(t-1,))
            assert row["samples"][0]["exact"]
            rows[m]={"first_rejected":fault,"last_completed_prefix":row}
    out={"scope":"Post-hoc boundary localization with the original lock, model, tape and range limits",
         "physics_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"formats":rows}
    with (folder/"last_safe_prefix.json").open("x",encoding="utf-8") as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print({m:v["last_completed_prefix"]["samples"] for m,v in rows.items()})


if __name__=="__main__":
    main()
