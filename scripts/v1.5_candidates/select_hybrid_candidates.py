#!/usr/bin/env python
"""Prototype hybrid mass-candidate selector on the debug subset.

For each debug query, decoy set (cap 511 + answer at index 0) =
  - K UNION-nearest-by-Tanimoto-to-answer across S4 u PubChem u Molpher (source
    agnostic; Molpher morphs naturally dominate), each pick capped at Tanimoto
    <= the query's closest_test_T (and < 0.99) so decoys approach the real
    test-neighbour density without becoming unsolvable; then
  - (511-K) random from the same mass-window union, S4->PubChem->Molpher order
    (the structure-blind base).
Dedup by 2D-InChIKey; exclude the answer's own ik2d. K=0 is the control.

Pools are sliced to the UNION of the debug queries' +/-10 ppm mass windows and
CACHED (so re-running with a different K-pick policy skips the slow stream).
Morgan r2/2048 Tanimoto. Writes one candidate JSON per K (debug subset only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RADIUS, NBITS, PPM, DUP_T, CAP = 2, 2048, 10.0, 0.99, 511
PW = NBITS // 8
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)

POOLS = [  # order = S4 -> PubChem -> Molpher (used for the random base)
    ("s4", "pools/s4/s4_generated_pool_clean.parquet",
     "pools/s4/s4_generated_pool_morgan_r2_2048_fps.bin"),
    ("pubchem", "pools/pubchem/pubchem_rdkit_canon_pool_clean.parquet",
     "pools/pubchem/pubchem_rdkit_canon_pool_morgan_r2_2048_fps.bin"),
    ("molpher", "pools/molpher/molpher_morphs_pool_clean.parquet",
     "pools/molpher/molpher_morphs_pool_morgan_r2_2048_fps.bin"),
]


def _packed(smi: str) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    bits = np.zeros(NBITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(_GEN.GetFingerprint(m), bits)
    return np.packbits(bits)


def _tanimoto(qp: np.ndarray, M: np.ndarray) -> np.ndarray:
    if len(M) == 0:
        return np.zeros(0)
    inter = _POP[qp & M].sum(1)
    return inter / np.maximum(int(_POP[qp].sum()) + _POP[M].sum(1) - inter, 1)


def _ik2d(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    try:
        k = Chem.MolToInchiKey(m)
        return k[:14] if k else None
    except Exception:
        return None


def _merge_windows(masses: np.ndarray) -> np.ndarray:
    tol = masses * PPM * 1e-6
    iv = np.stack([masses - tol, masses + tol], axis=1)
    iv = iv[np.argsort(iv[:, 0])]
    out = [iv[0].copy()]
    for lo, hi in iv[1:]:
        if lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append(np.array([lo, hi]))
    return np.array(out)


def _slice_pool(clean_path: Path, fps_path: Path, intervals: np.ndarray, cache: Path):
    """Rows (mass, ik2d, smiles, packed_fps) in any interval, sorted by mass.
    Cached to <cache>.parquet + <cache>_fps.npy for fast re-runs."""
    cp, fpn = cache.with_suffix(".parquet"), cache.parent / (cache.name + "_fps.npy")
    if cp.exists() and fpn.exists():
        t = pq.read_table(cp)
        return (t.column("mass").to_numpy(), np.asarray(t.column("ik").to_pylist(), object),
                np.asarray(t.column("smi").to_pylist(), object), np.load(fpn))
    los, his = intervals[:, 0], intervals[:, 1]
    fp_mm = np.memmap(fps_path, dtype=np.uint8, mode="r").reshape(-1, PW)
    idx_parts, mass_parts, ik, smi = [], [], [], []
    off = 0
    for b in pq.ParquetFile(clean_path).iter_batches(
            batch_size=2_000_000, columns=["exact_mass", "inchikey_2d", "smiles"]):
        m = b.column("exact_mass").to_numpy()
        j = np.searchsorted(los, m, side="right") - 1
        hit = (j >= 0) & (m <= his[np.clip(j, 0, len(his) - 1)])
        if hit.any():
            loc = np.nonzero(hit)[0]
            sub = b.take(pa.array(loc))  # materialise only matched rows
            idx_parts.append(off + loc)
            mass_parts.append(m[loc])
            ik.extend(sub.column("inchikey_2d").to_pylist())
            smi.extend(sub.column("smiles").to_pylist())
        off += b.num_rows
    if not idx_parts:
        mass, ikv, smiv, fps = np.zeros(0), np.array([], object), np.array([], object), np.zeros((0, PW), np.uint8)
    else:
        gidx = np.concatenate(idx_parts)
        mass = np.concatenate(mass_parts)
        order = np.argsort(mass)
        mass = mass[order]
        ikv = np.asarray(ik, object)[order]
        smiv = np.asarray(smi, object)[order]
        fps = np.ascontiguousarray(fp_mm[gidx][order])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"mass": mass, "ik": ikv.tolist(), "smi": smiv.tolist()}), cp)
    np.save(fpn, fps)
    return mass, ikv, smiv, fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries-parquet", required=True)
    ap.add_argument("--mass-json", required=True)
    ap.add_argument("--ks", default="0,32,64")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    data = Path(args.data_dir)
    ks = [int(x) for x in args.ks.split(",")]
    rng = np.random.RandomState(0)
    cache_dir = Path(args.out_dir) / f"pool_cache_top{args.top_n}"

    q = pd.read_parquet(args.queries_parquet)
    q = q[q["fold"] == "test"].reset_index(drop=True)
    qpacked = [_packed(s) for s in q["smiles"]]
    keep = [i for i, f in enumerate(qpacked) if f is not None]
    q = q.iloc[keep].reset_index(drop=True)
    Q = np.stack([qpacked[i] for i in keep])
    iks = q["inchikey_2d"].to_numpy()
    code = pd.factorize(iks)[0]
    ctest = np.zeros(len(q))
    for i in range(len(q)):
        s = _tanimoto(Q[i], Q)
        s[code == code[i]] = -1
        ctest[i] = s.max()
    cands = json.load(open(args.mass_json))
    tset = set(iks)
    ik_dec: dict[str, list[str]] = {}
    for k, v in cands.items():
        kk = _ik2d(k)
        if kk in tset and kk not in ik_dec:
            ik_dec[kk] = v[1:]
    cdecoy = np.zeros(len(q))
    for i in range(len(q)):
        d = [p for p in (_packed(s) for s in ik_dec.get(iks[i], [])) if p is not None]
        cdecoy[i] = _tanimoto(Q[i], np.stack(d)).max() if d else 0.0
    q = q.assign(closest_test_T=ctest, separation=ctest - cdecoy)
    debug = (q[q["closest_test_T"] < 0.95]
             .sort_values("separation", ascending=False).head(args.top_n).reset_index(drop=True))
    qp_by_smi = {s: _packed(s) for s in debug["smiles"]}
    print(f"[sel] debug subset {len(debug)} queries; mass {debug['exact_mass'].min():.1f}..{debug['exact_mass'].max():.1f}", flush=True)

    intervals = _merge_windows(debug["exact_mass"].to_numpy())
    pools = {}
    for name, cp, fp in POOLS:
        pools[name] = _slice_pool(data / cp, data / fp, intervals, cache_dir / name)
        print(f"[sel] pool {name}: {len(pools[name][0]):,} rows in window union", flush=True)

    out = {k: {} for k in ks}
    npicks = {k: [] for k in ks}
    for r in debug.itertuples():
        qp = qp_by_smi[r.smiles]
        tol = r.exact_mass * PPM * 1e-6
        cap = float(r.closest_test_T)
        per_pool = []
        for name, _, _ in POOLS:
            mass, ik, smi, fps = pools[name]
            lo = np.searchsorted(mass, r.exact_mass - tol, "left")
            hi = np.searchsorted(mass, r.exact_mass + tol, "right")
            sims = _tanimoto(qp, fps[lo:hi]) if hi > lo else np.zeros(0)
            per_pool.append((ik[lo:hi], smi[lo:hi], sims))
        # union of all window members for the nearest-K pick
        u_ik = np.concatenate([p[0] for p in per_pool]) if per_pool else np.array([], object)
        u_smi = np.concatenate([p[1] for p in per_pool]) if per_pool else np.array([], object)
        u_sim = np.concatenate([p[2] for p in per_pool]) if per_pool else np.zeros(0)
        cap_ok = np.nonzero((u_sim <= cap) & (u_sim < DUP_T) & (u_sim > 0))[0]
        nearest = cap_ok[np.argsort(-u_sim[cap_ok])]  # closest-under-cap first, union
        for K in ks:
            seen = {r.inchikey_2d}
            picks: list[str] = []
            for idx in nearest:
                ik = u_ik[idx]
                if ik and ik not in seen:
                    seen.add(ik)
                    picks.append(u_smi[idx])
                    if len(picks) >= K:
                        break
            fill: list[str] = []
            for sl_ik, sl_smi, sims in per_pool:  # random base, S4 -> PubChem -> Molpher
                need = CAP - len(picks) - len(fill)
                if need <= 0:
                    break
                avail = [i for i in range(len(sl_ik))
                         if sl_ik[i] and sl_ik[i] not in seen and sims[i] < DUP_T]
                rng.shuffle(avail)
                for i in avail[:need]:
                    seen.add(sl_ik[i])
                    fill.append(sl_smi[i])
            out[K][r.smiles] = [r.smiles] + picks + fill
            npicks[K].append(len(picks))

    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    json.dump(debug["inchikey_2d"].tolist(), open(outd / "debug_subset_iks.json", "w"))
    for K in ks:
        p = outd / f"hybrid_mass_debug_k{K}.json"
        json.dump(out[K], open(p, "w"))
        print(f"[sel] K={K}: wrote {p.name} ({len(out[K])} queries); mean picks={np.mean(npicks[K]):.1f}", flush=True)


if __name__ == "__main__":
    main()
