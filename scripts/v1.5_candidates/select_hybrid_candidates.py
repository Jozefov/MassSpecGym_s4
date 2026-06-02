#!/usr/bin/env python
"""Prototype hybrid mass-candidate selector on the debug subset.

For each debug query, decoy set (cap 511 + answer at index 0) =
  - K nearest-by-Tanimoto-to-answer drawn S4->PubChem->Molpher, each pick capped
    at Tanimoto <= the query's closest_test_T (and < 0.99) so decoys approach the
    real test-neighbour density without becoming unsolvable; then
  - (511-K) random from the same mass-window union (S4->PubChem->Molpher order).
Dedup by 2D-InChIKey; exclude the answer's own ik2d.

K=0 is the control (pure random from the new pools). Pools are sliced to the
UNION of the debug queries' +/-10 ppm mass windows (cheap); FP rows are read
from the aligned *_morgan_r2_2048_fps.bin via mmap. Morgan r2/2048 Tanimoto.

Writes one candidate JSON per K (debug subset only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RADIUS, NBITS, PPM, DUP_T, CAP = 2, 2048, 10.0, 0.99, 511
PW = NBITS // 8
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)

# order = S4 -> PubChem -> Molpher
POOLS = [
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
    """Tanimoto of one packed FP (256,) vs a packed matrix (n,256)."""
    if len(M) == 0:
        return np.zeros(0)
    inter = _POP[qp & M].sum(1)
    qpop = int(_POP[qp].sum())
    mpop = _POP[M].sum(1)
    return inter / np.maximum(qpop + mpop - inter, 1)


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
    """Merge +/-PPM windows of the masses into sorted disjoint [lo,hi] intervals."""
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


def _slice_pool(clean_path: Path, fps_path: Path, intervals: np.ndarray):
    """Return (mass, ik2d, smiles, packed_fps) for pool rows in any interval,
    sorted by mass."""
    los, his = intervals[:, 0], intervals[:, 1]
    pf = pq.ParquetFile(clean_path)
    fp_mm = np.memmap(fps_path, dtype=np.uint8, mode="r").reshape(-1, PW)
    idx_parts, mass_parts, ik, smi = [], [], [], []
    off = 0
    for b in pf.iter_batches(batch_size=2_000_000,
                             columns=["exact_mass", "inchikey_2d", "smiles"]):
        m = b.column("exact_mass").to_numpy()
        j = np.searchsorted(los, m, side="right") - 1
        hit = (j >= 0) & (m <= his[np.clip(j, 0, len(his) - 1)])
        if hit.any():
            loc = np.nonzero(hit)[0]
            idx_parts.append(off + loc)
            mass_parts.append(m[loc])
            bik, bsmi = b.column("inchikey_2d").to_pylist(), b.column("smiles").to_pylist()
            ik.extend(bik[k] for k in loc)
            smi.extend(bsmi[k] for k in loc)
        off += b.num_rows
    if not idx_parts:
        return np.zeros(0), np.array([], object), np.array([], object), np.zeros((0, PW), np.uint8)
    gidx = np.concatenate(idx_parts)
    fps = np.ascontiguousarray(fp_mm[gidx])
    mass = np.concatenate(mass_parts)
    order = np.argsort(mass)
    return mass[order], np.asarray(ik, object)[order], np.asarray(smi, object)[order], fps[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help=".../_data/v1.5")
    ap.add_argument("--queries-parquet", required=True)
    ap.add_argument("--mass-json", required=True, help="baseline mass JSON for the debug subset")
    ap.add_argument("--ks", default="0,32,64")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    data = Path(args.data_dir)
    ks = [int(x) for x in args.ks.split(",")]
    rng = np.random.RandomState(0)

    # test queries + closest_test_T + separation -> debug subset
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
    print(f"[sel] debug subset {len(debug)} queries; "
          f"mass {debug['exact_mass'].min():.1f}..{debug['exact_mass'].max():.1f}", flush=True)

    # slice the 3 pools to the union of debug mass windows
    intervals = _merge_windows(debug["exact_mass"].to_numpy())
    pools = {}
    for name, cp, fp in POOLS:
        pools[name] = _slice_pool(data / cp, data / fp, intervals)
        print(f"[sel] pool {name}: {len(pools[name][0]):,} rows in window union", flush=True)

    out = {k: {} for k in ks}
    npicks = {k: [] for k in ks}
    ndec = {k: [] for k in ks}
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
        for K in ks:
            seen = {r.inchikey_2d}
            picks: list[str] = []
            for sl_ik, sl_smi, sims in per_pool:  # K nearest-under-cap, in pool order
                if len(picks) >= K:
                    break
                ok = np.nonzero((sims <= cap) & (sims < DUP_T) & (sims > 0))[0]
                for idx in ok[np.argsort(-sims[ok])]:
                    ikk = sl_ik[idx]
                    if ikk and ikk not in seen:
                        seen.add(ikk)
                        picks.append(sl_smi[idx])
                        if len(picks) >= K:
                            break
            fill: list[str] = []
            for sl_ik, sl_smi, sims in per_pool:  # random fill, in pool order
                need = CAP - len(picks) - len(fill)
                if need <= 0:
                    break
                avail = [i for i in range(len(sl_ik))
                         if sl_ik[i] and sl_ik[i] not in seen and sims[i] < DUP_T]
                rng.shuffle(avail)
                for i in avail[:need]:
                    seen.add(sl_ik[i])
                    fill.append(sl_smi[i])
            decoys = picks + fill
            out[K][r.smiles] = [r.smiles] + decoys
            npicks[K].append(len(picks))
            ndec[K].append(len(decoys))

    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    # fixed debug subset (ik2d, in separation order) so baseline + every K are scored on the same queries
    json.dump(debug["inchikey_2d"].tolist(), open(outd / "debug_subset_iks.json", "w"))
    for K in ks:
        p = outd / f"hybrid_mass_debug_k{K}.json"
        json.dump(out[K], open(p, "w"))
        print(f"[sel] K={K}: wrote {p.name} ({len(out[K])} queries); "
              f"mean picks={np.mean(npicks[K]):.1f} mean decoys={np.mean(ndec[K]):.1f}", flush=True)


if __name__ == "__main__":
    main()
