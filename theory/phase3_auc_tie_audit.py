"""Rescore the locked pre-verifier signals with one ridge score per bit-identical feature row."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval_v3.metrics import draws
from theory.phase3_reference_audit import auc


def auc_at_ranks(score, y, rank):
    weighted = 0.0
    pairs = 0
    for k in np.unique(rank):
        take = rank == k
        positives = int(y[take].sum())
        n = positives * (int(take.sum()) - positives)
        if n:
            weighted += n * auc(score[take], y[take])
            pairs += n
    return weighted / pairs if pairs else None


def score_scheme(rows, weights, scheme, frozen):
    conditional = scheme == "conditional_on_rank"
    x, y, rank = draws(rows, exact=(scheme == "exact_prior"), surviving=conditional)
    norm = (x - weights["mu"]) / weights["sd"]
    batched = norm @ weights["w"]
    per_vector = {}
    raw_ranges = {}
    stable = np.empty_like(batched)
    for i,row in enumerate(x):
        key = row.tobytes()
        z = float(batched[i])
        if key not in per_vector:
            per_vector[key] = math.fsum(v*w for v,w in zip(norm[i], weights["w"]))
            raw_ranges[key] = [z,z]
        stable[i] = per_vector[key]
        raw_ranges[key][0] = min(raw_ranges[key][0],z)
        raw_ranges[key][1] = max(raw_ranges[key][1],z)
    def reduce(score):
        return auc_at_ranks(score,y,rank) if conditional else auc(score,y)
    columns = np.column_stack((x,stable)).T
    corrected = [reduce(col) for col in columns]
    assert np.allclose(corrected[:-1], frozen["auc"][:-1], atol=1e-12,rtol=0)
    first = rank == 0
    early = [auc(col[first],y[first]) for col in columns]
    if scheme == "exact_prior":
        assert all(sum(l["terminal_pass"])==1 for r in rows for l in r["leaves"])
        assert all(v == 0.5 for v in corrected)
    return {"frozen_ridge_auc":frozen["auc"][-1], "batched_ridge_auc_this_process":reduce(batched),
            "one_score_per_vector_ridge_auc":corrected[-1], "unique_bit_vectors":len(per_vector),
            "split_bitwise_tie_groups":sum(lo!=hi for lo,hi in raw_ranges.values()),
            "max_within_identical_row_spread":max(hi-lo for lo,hi in raw_ranges.values()),
            "frozen_max":frozen["max_orientation_free"],
            "corrected_max":max(max(v,1-v) for v in corrected if v is not None),
            "corrected_first_leaf_max":max(max(v,1-v) for v in early if v is not None)}


def measure(folder):
    test=json.loads((folder/"test.json").read_bytes())
    monitor=test["chosen"]["monitor"]
    result={}
    for d in (1,2,4,8):
        result[d]={}
        for method in ("native","int32","int64"):
            key=f"{method}_{d}"
            rows=json.loads((folder/test["parts"][key]["path"]).read_bytes())["cell"]["rows"]
            saved=test["decisions"]["depths"][str(d)]["monitors"][method]
            result[d][method]={s:score_scheme(rows,monitor,s,saved[s])
                               for s in ("sampled","exact_prior","conditional_on_rank")}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    print(json.dumps(measure(parser.parse_args().folder),allow_nan=False))
