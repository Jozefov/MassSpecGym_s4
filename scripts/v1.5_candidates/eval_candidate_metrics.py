#!/usr/bin/env python
"""Score a candidate JSON on the three structure-only shortcuts, over the
high-separation debug subset of the test fold. Morgan r2/2048 Tanimoto.

Per query the candidate set = [answer] + decoys (answer at index 0). Metrics:

  a15 (test-twin)  : rank each candidate by its max Tanimoto to OTHER test-fold
                     GTs (leave-one-out by 2D-InChIKey). top-20 = answer lands
                     in top 20. PUSH DOWN toward ~10%.
  a45 (centroid)   : rank each candidate by mean Tanimoto to the OTHER
                     candidates in its own set. top-20 = answer in top 20. Keep
                     FLAT (~random, 20/|set|). Rises if decoys cluster on the GT.
  dup (solvable)   : fraction of sets containing a decoy with Tanimoto >= 0.99
                     to the answer (degenerate/unsolvable). Want ~0.

Debug subset = top-N by separation (closest_test_T - closest_decoy_T) with
closest_test_T < 0.95 (drops FP-saturated lipids). closest_test_T is fixed
(answer vs nearest real test molecule); closest_decoy_T comes from this JSON.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RADIUS, NBITS, DUP_T = 2, 2048, 0.99
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)

# Globals shared with forked workers.
TEST_FPS: list = []
TEST_IK: np.ndarray = np.array([])


def _fp(smi: str):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    return _GEN.GetFingerprint(m) if m is not None else None


def _ik2d(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    try:
        k = Chem.MolToInchiKey(m)
        return k[:14] if k else None
    except Exception:
        return None


def _score_query(args) -> tuple | None:
    """Compute (hit20_a15, hit20_a45, dup, closest_decoy_T) for one query."""
    answer_smi, answer_ik, decoys = args
    a_fp = _fp(answer_smi)
    if a_fp is None:
        return None
    cand_smis = [answer_smi] + list(decoys)
    cand_ik = [answer_ik] + [_ik2d(s) for s in decoys]
    cand_fps = [a_fp] + [_fp(s) for s in decoys]
    keep = [i for i, f in enumerate(cand_fps) if f is not None]
    cand_fps = [cand_fps[i] for i in keep]
    cand_ik = [cand_ik[i] for i in keep]
    n = len(cand_fps)
    if n < 2:
        return None

    # a15: max Tanimoto to other test GTs (LOO by ik2d).
    a15 = np.empty(n)
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(cand_fps[i], TEST_FPS))
        sims[TEST_IK == cand_ik[i]] = -1.0
        a15[i] = sims.max()
    # a45: mean Tanimoto to other candidates in the set.
    a45 = np.empty(n)
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(cand_fps[i], cand_fps))
        a45[i] = (sims.sum() - 1.0) / (n - 1)  # drop self (sim=1)
    # answer is index 0 in the kept order (it was first and always valid).
    rank_a15 = int((a15 > a15[0]).sum())  # 0-based rank of the answer
    rank_a45 = int((a45 > a45[0]).sum())
    decoy_sims = np.asarray(DataStructs.BulkTanimotoSimilarity(cand_fps[0], cand_fps[1:]))
    closest_decoy_T = float(decoy_sims.max()) if len(decoy_sims) else 0.0
    dup = closest_decoy_T >= DUP_T
    return (answer_ik, rank_a15 < 20, rank_a45 < 20, dup, closest_decoy_T)


def _init(test_fps, test_ik):
    global TEST_FPS, TEST_IK
    RDLogger.DisableLog("rdApp.*")
    TEST_FPS, TEST_IK = test_fps, test_ik


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries-parquet", required=True)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--fold", default="test")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--debug-iks", default=None,
                    help="JSON list of query ik2d to score (fixed subset); else top-N by separation")
    ap.add_argument("--out-csv", default=None, help="per-query results CSV for tier breakdown")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")

    q = pd.read_parquet(args.queries_parquet)
    q = q[q["fold"] == args.fold].reset_index(drop=True)
    qfps = [_fp(s) for s in q["smiles"]]
    keep = [i for i, f in enumerate(qfps) if f is not None]
    q = q.iloc[keep].reset_index(drop=True)
    test_fps = [qfps[i] for i in keep]
    test_ik = np.asarray(q["inchikey_2d"].tolist(), dtype=object)
    smis = q["smiles"].tolist()
    n = len(q)
    print(f"[eval] {n} {args.fold}-fold GTs", flush=True)

    ik_codes = pd.factorize(test_ik)[0]
    closest_test_T = np.zeros(n)
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(test_fps[i], test_fps))
        sims[ik_codes == ik_codes[i]] = -1.0
        closest_test_T[i] = sims.max()

    cands: dict[str, list[str]] = json.load(open(args.candidates_json))
    test_ik_set = set(test_ik)
    ik_to_decoys: dict[str, list[str]] = {}
    for k, vals in cands.items():
        kk = _ik2d(k)
        if kk in test_ik_set and kk not in ik_to_decoys:
            ik_to_decoys[kk] = vals[1:]
    print(f"[eval] matched {len(ik_to_decoys)}/{n} to candidate JSON", flush=True)

    closest_decoy_T = np.zeros(n)
    for i in range(n):
        d = [f for f in (_fp(s) for s in ik_to_decoys.get(test_ik[i], [])) if f is not None]
        closest_decoy_T[i] = max(DataStructs.BulkTanimotoSimilarity(test_fps[i], d)) if d else 0.0

    sep = closest_test_T - closest_decoy_T
    df = pd.DataFrame({"smiles": smis, "ik": test_ik, "formula": q["formula"].tolist(),
                       "closest_test_T": closest_test_T, "closest_decoy_T": closest_decoy_T,
                       "separation": sep, "n_decoys": [len(ik_to_decoys.get(k, [])) for k in test_ik]})
    v = df[df["formula"] == "C48H81N9O8"]
    if len(v):
        r = v.iloc[0]
        print(f"[eval] VERIFY C48H81N9O8: test_T={r.closest_test_T:.3f} decoy_T={r.closest_decoy_T:.3f} "
              f"sep={r.separation:.3f} n_decoys={int(r.n_decoys)}", flush=True)

    if args.debug_iks:
        order = {ik: i for i, ik in enumerate(json.load(open(args.debug_iks)))}
        debug = df[df["ik"].isin(order)].copy()
        debug = debug.iloc[debug["ik"].map(order).argsort().to_numpy()]
        print(f"[eval] debug subset: {len(debug)}/{len(order)} from --debug-iks", flush=True)
    else:
        debug = df[df["closest_test_T"] < 0.95].sort_values("separation", ascending=False).head(args.top_n)
        print(f"[eval] debug subset: {len(debug)} queries (top-{args.top_n}, test_T<0.95); "
              f"sep range {debug.separation.min():.3f}..{debug.separation.max():.3f}", flush=True)

    tasks = [(r.smiles, r.ik, ik_to_decoys.get(r.ik, [])) for r in debug.itertuples()]
    with mp.get_context("fork").Pool(args.n_workers, initializer=_init,
                                     initargs=(test_fps, test_ik)) as pool:
        res = [r for r in pool.map(_score_query, tasks) if r is not None]
    rdf = pd.DataFrame(res, columns=["ik", "hit_a15", "hit_a45", "dup", "cdt"]).merge(
        debug[["ik", "separation", "closest_test_T"]], on="ik", how="left")
    a15, a45, dup, cdt = rdf.hit_a15.mean(), rdf.hit_a45.mean(), rdf.dup.mean(), rdf.cdt.mean()
    print(f"\n[eval] === {args.label} (n={len(rdf)}) ===")
    print(f"[eval] a15 test-twin top-20 : {a15*100:.1f}%   (push toward ~10%)")
    print(f"[eval] a45 centroid  top-20 : {a45*100:.1f}%   (keep flat ~ random {20/512*100:.1f}%)")
    print(f"[eval] dup>=0.99 in set     : {dup*100:.1f}%   (want ~0)")
    print(f"[eval] mean closest_decoy_T : {cdt:.3f}   (ceiling = per-query closest_test_T)")
    for lab, m in [("sep>0.4 (hard)", rdf.separation > 0.4),
                   ("sep 0.2-0.4", (rdf.separation > 0.2) & (rdf.separation <= 0.4)),
                   ("sep<=0.2 (easy)", rdf.separation <= 0.2)]:
        s = rdf[m]
        if len(s):
            print(f"[eval]   tier {lab} (n={len(s)}): a15={s.hit_a15.mean()*100:.1f}% "
                  f"a45={s.hit_a45.mean()*100:.1f}% dup={s.dup.mean()*100:.1f}%")
    if args.out_csv:
        rdf.to_csv(args.out_csv, index=False)
        print(f"[eval] wrote per-query {args.out_csv}")


if __name__ == "__main__":
    main()
