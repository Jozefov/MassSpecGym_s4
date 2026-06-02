#!/usr/bin/env python
"""Sweep S4-pool decoy-sampling strategies on the debug subset, to push the
test-twin shortcut (a15) DOWN without lifting the centroid metric (a45).

All decoys come from the S4-generated pool (+PubChem/Molpher only as already
sliced) -- no new molecule source. The lever is *which* pool molecules we pick.

Strategies (each builds a 511-decoy set; answer at index 0; Morgan r2/2048):
  random           random from the mass-window union (~ current baseline)
  testlike         top-511 by max-Tanimoto-to-the-test-fold (measured-like): makes
                   many decoys as test-twin-close as the GT -> a15 down; they sit
                   near *various* test mols (not the GT) -> a45 should stay flat
  testlike_spread  farthest-first 511 among the top measured-like pool (extra
                   anti-clustering insurance for a45)
  unionnearest64   64 nearest-to-GT (cap=closest_test_T) + 447 random (reference;
                   expected to spike a45 -- included to confirm the contrast)

Reuses the cached pool slices written by select_hybrid_candidates._slice_pool
(falls back to streaming if absent). Dedup by 2D-InChIKey; drop Tanimoto>=0.99.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from select_hybrid_candidates import (
    _ik2d, _merge_windows, _packed, _slice_pool, _tanimoto, _POP, CAP, DUP_T, PPM, POOLS,
)


def _max_sim_to_test(fps: np.ndarray, testQ: np.ndarray) -> np.ndarray:
    """For each packed FP in `fps` (n,256), max Tanimoto to any test FP (T,256)."""
    if len(fps) == 0:
        return np.zeros(0)
    bpop = _POP[fps].sum(1)
    tpop = _POP[testQ].sum(1)
    best = np.zeros(len(fps))
    for t in range(len(testQ)):
        inter = _POP[testQ[t] & fps].sum(1)
        sim = inter / np.maximum(tpop[t] + bpop - inter, 1)
        np.maximum(best, sim, out=best)
    return best


def _pick(ik, smi, score_desc, k, gt_ik, seen=None):
    """Take up to k by descending `score_desc`, dedup ik2d, exclude gt + seen."""
    seen = set(seen or ()) | {gt_ik}
    out = []
    for i in np.argsort(-score_desc):
        if len(out) >= k:
            break
        if ik[i] and ik[i] not in seen:
            seen.add(ik[i])
            out.append(smi[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-parquet", required=True)
    ap.add_argument("--mass-json", required=True)
    ap.add_argument("--cache-dir", required=True, help="pool_cache_topN dir from the selector")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--strategies", default="random,testlike,testlike_spread,unionnearest64")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    data = Path(args.data_dir)
    strategies = args.strategies.split(",")
    rng = np.random.RandomState(0)

    # queries + closest_test_T + separation -> debug subset (same as selector)
    q = pd.read_parquet(args.queries_parquet)
    q = q[q["fold"] == "test"].reset_index(drop=True)
    qp = [_packed(s) for s in q["smiles"]]
    keep = [i for i, f in enumerate(qp) if f is not None]
    q = q.iloc[keep].reset_index(drop=True)
    Q = np.stack([qp[i] for i in keep])
    iks = q["inchikey_2d"].to_numpy()
    code = pd.factorize(iks)[0]
    ctest = np.zeros(len(q))
    for i in range(len(q)):
        s = _tanimoto(Q[i], Q); s[code == code[i]] = -1; ctest[i] = s.max()
    cands = json.load(open(args.mass_json))
    tset = set(iks); ik_dec = {}
    for k, v in cands.items():
        kk = _ik2d(k)
        if kk in tset and kk not in ik_dec:
            ik_dec[kk] = v[1:]
    cdecoy = np.zeros(len(q))
    for i in range(len(q)):
        d = [p for p in (_packed(s) for s in ik_dec.get(iks[i], [])) if p is not None]
        cdecoy[i] = _tanimoto(Q[i], np.stack(d)).max() if d else 0.0
    q = q.assign(closest_test_T=ctest, separation=ctest - cdecoy)
    debug = (q[q["closest_test_T"] < 0.95].sort_values("separation", ascending=False)
             .head(args.top_n).reset_index(drop=True))
    qp_by_smi = {s: _packed(s) for s in debug["smiles"]}
    print(f"[sweep] {len(debug)} debug queries; strategies={strategies}", flush=True)

    # pool slices (cached) + their measured-likeness (max sim to test), computed once
    intervals = _merge_windows(debug["exact_mass"].to_numpy())
    cache = Path(args.cache_dir)
    pools = {}
    for name, cp, fp in POOLS:
        mass, ik, smi, fps = _slice_pool(data / cp, data / fp, intervals, cache / name)
        print(f"[sweep] pool {name}: {len(mass):,} rows; computing measured-likeness ...", flush=True)
        simtest = _max_sim_to_test(fps, Q)
        pools[name] = (mass, ik, smi, fps, simtest)
        print(f"[sweep]   {name} max-sim-to-test: median={np.median(simtest):.3f} p90={np.percentile(simtest,90):.3f}", flush=True)

    out = {s: {} for s in strategies}
    for r in debug.itertuples():
        qp_ = qp_by_smi[r.smiles]
        tol = r.exact_mass * PPM * 1e-6
        cap = float(r.closest_test_T)
        ik_a, smi_a, simgt_a, simtest_a = [], [], [], []
        for name, _, _ in POOLS:
            mass, ik, smi, fps, simtest = pools[name]
            lo = np.searchsorted(mass, r.exact_mass - tol, "left")
            hi = np.searchsorted(mass, r.exact_mass + tol, "right")
            if hi <= lo:
                continue
            ik_a.append(ik[lo:hi]); smi_a.append(smi[lo:hi])
            simgt_a.append(_tanimoto(qp_, fps[lo:hi])); simtest_a.append(simtest[lo:hi])
        if not ik_a:
            for s in strategies:
                out[s][r.smiles] = [r.smiles]
            continue
        bik = np.concatenate(ik_a); bsmi = np.concatenate(smi_a)
        bsgt = np.concatenate(simgt_a); bstest = np.concatenate(simtest_a)
        ok = bsgt < DUP_T  # never include near-identical decoys (solvable)
        bik, bsmi, bsgt, bstest = bik[ok], bsmi[ok], bsgt[ok], bstest[ok]
        for s in strategies:
            if s == "random":
                order = rng.permutation(len(bik)).astype(float)
                decoys = _pick(bik, bsmi, order, CAP, r.inchikey_2d)
            elif s == "testlike":
                # measured-like (high max-sim-to-test) but NOT hugging the GT
                # (sim-to-GT <= closest_test_T) so we don't recreate the a45
                # centroid leak; this prefers decoys near *other* test molecules.
                m = bsgt <= cap
                decoys = _pick(bik[m], bsmi[m], bstest[m], CAP, r.inchikey_2d)
            elif s == "testlike_spread":
                m = bsgt <= cap
                mik, msmi, mstest = bik[m], bsmi[m], bstest[m]
                top = np.argsort(-mstest)[:min(len(mik), CAP * 4)]  # measured-like shortlist
                order = rng.permutation(len(top)).astype(float)   # spread within it
                decoys = _pick(mik[top], msmi[top], order, CAP, r.inchikey_2d)
            elif s == "unionnearest64":
                capok = np.nonzero((bsgt <= cap) & (bsgt > 0))[0]
                near = capok[np.argsort(-bsgt[capok])]
                picks, seen = [], {r.inchikey_2d}
                for i in near:
                    if len(picks) >= 64:
                        break
                    if bik[i] and bik[i] not in seen:
                        seen.add(bik[i]); picks.append(bsmi[i])
                fill = _pick(bik, bsmi, rng.permutation(len(bik)).astype(float),
                             CAP - len(picks), r.inchikey_2d, seen=seen)
                decoys = picks + fill
            else:
                raise SystemExit(f"unknown strategy {s}")
            out[s][r.smiles] = [r.smiles] + decoys

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    json.dump(debug["inchikey_2d"].tolist(), open(outd / "sweep_debug_iks.json", "w"))
    for s in strategies:
        p = outd / f"sweep_{s}.json"
        json.dump(out[s], open(p, "w"))
        sizes = [len(v) - 1 for v in out[s].values()]
        print(f"[sweep] {s}: wrote {p.name} (mean decoys={np.mean(sizes):.1f})", flush=True)


if __name__ == "__main__":
    main()
