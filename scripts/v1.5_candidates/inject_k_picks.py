#!/usr/bin/env python
"""Inject K similarity-picked decoys into the REAL v1.5 baseline candidate sets.

Exactly the minority design: keep (511-K) of the existing MassSpecGym S4
pipeline decoys (gen_pool -> pubchem -> molpher) UNTOUCHED, swap only K (32/64)
for similarity picks. Methods for the K picks:
  twin       Molpher morphs of the GT's iso-mass TEST neighbours (near OTHER test
             molecules -> lowers a15; not near the GT -> a45 stays at baseline)
  testlike   pool molecules nearer to another test molecule than to the GT (LOO)
  unionnear  nearest-to-GT (CONTROL; expected to break a45)
Writes one candidate JSON per (method, K). Morgan r2/2048. Reuses cached slices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from select_hybrid_candidates import (
    _ik2d, _merge_windows, _packed, _slice_pool, _tanimoto, DUP_T, PPM, POOLS,
)
from sweep_sampling_strategies import _max_sim_to_test


def _distinct_take(smis, n, gt_ik, seen=None):
    """First n distinct-ik2d SMILES (excl gt_ik / seen)."""
    seen = set(seen or ()) | {gt_ik}
    out = []
    for sm in smis:
        if len(out) >= n:
            break
        ik = _ik2d(sm)
        if ik and ik not in seen:
            seen.add(ik)
            out.append(sm)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-parquet", required=True)
    ap.add_argument("--mass-json", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--morphs-json", required=True)
    ap.add_argument("--methods", default="twin,testlike,unionnear")
    ap.add_argument("--ks", default="32,64")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    data = Path(args.data_dir)
    methods = args.methods.split(",")
    ks = sorted(int(x) for x in args.ks.split(","))
    kmax = max(ks)
    CAP = 511
    rng = np.random.RandomState(0)

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
    print(f"[inject] {len(debug)} debug queries; methods={methods} ks={ks}", flush=True)

    # twin source + pool slices
    morphs_json = json.load(open(args.morphs_json))
    torder = np.argsort(q["exact_mass"].to_numpy())
    tmass_s = q["exact_mass"].to_numpy()[torder]; tsmi_s = q["smiles"].to_numpy()[torder]; tik_s = iks[torder]
    cache = Path(args.cache_dir)
    pools = {}
    for name, cp, fp in POOLS:
        mass, ik, smi, fps = _slice_pool(data / cp, data / fp, _merge_windows(debug["exact_mass"].to_numpy()), cache / name)
        simtest = _max_sim_to_test(fps, Q) if "testlike" in methods else None
        pools[name] = (mass, ik, smi, fps, simtest)
        print(f"[inject] pool {name}: {len(mass):,} rows", flush=True)

    out = {f"base_{m}_k{k}": {} for m in methods for k in ks}
    for r in debug.itertuples():
        qp_ = qp_by_smi[r.smiles]
        tol = r.exact_mass * PPM * 1e-6
        cap = float(r.closest_test_T)
        base = list(ik_dec.get(r.inchikey_2d, []))
        rng.shuffle(base)
        picks = {}
        if "twin" in methods:
            gathered = []
            jlo = np.searchsorted(tmass_s, r.exact_mass - tol, "left")
            jhi = np.searchsorted(tmass_s, r.exact_mass + tol, "right")
            for j in range(jlo, jhi):
                if tik_s[j] != r.inchikey_2d:
                    gathered.extend(morphs_json.get(tsmi_s[j], []))
            gathered = [sm for sm in gathered
                        if (p := _packed(sm)) is not None and _tanimoto(qp_, p[None, :])[0] < DUP_T]
            picks["twin"] = _distinct_take(gathered, kmax, r.inchikey_2d)
        if "testlike" in methods or "unionnear" in methods:
            ik_a, smi_a, sgt_a, st_a = [], [], [], []
            for name, _, _ in POOLS:
                mass, ik, smi, fps, simtest = pools[name]
                lo = np.searchsorted(mass, r.exact_mass - tol, "left")
                hi = np.searchsorted(mass, r.exact_mass + tol, "right")
                if hi <= lo:
                    continue
                ik_a.append(ik[lo:hi]); smi_a.append(smi[lo:hi]); sgt_a.append(_tanimoto(qp_, fps[lo:hi]))
                st_a.append(simtest[lo:hi] if simtest is not None else np.zeros(hi - lo))
            if ik_a:
                bsmi = np.concatenate(smi_a); bsgt = np.concatenate(sgt_a); bst = np.concatenate(st_a)
                if "unionnear" in methods:
                    m = (bsgt <= cap) & (bsgt > 0)
                    picks["unionnear"] = _distinct_take([bsmi[i] for i in np.argsort(-bsgt) if m[i]], kmax, r.inchikey_2d)
                if "testlike" in methods:
                    m = (bst > bsgt) & (bsgt <= cap)
                    picks["testlike"] = _distinct_take([bsmi[i] for i in np.argsort(-bst) if m[i]], kmax, r.inchikey_2d)
        for mth in methods:
            pk = picks.get(mth, [])
            for K in ks:
                kp = pk[:K]
                fill = base[: CAP - len(kp)]
                out[f"base_{mth}_k{K}"][r.smiles] = [r.smiles] + kp + fill

    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    json.dump(debug["inchikey_2d"].tolist(), open(outd / "inject_debug_iks.json", "w"))
    for key, d in out.items():
        npk = int(np.mean([sum(1 for _ in v) for v in d.values()]))  # mean list len
        json.dump(d, open(outd / f"{key}.json", "w"))
        print(f"[inject] wrote {key}.json ({len(d)} queries, mean len {npk})", flush=True)


if __name__ == "__main__":
    main()
