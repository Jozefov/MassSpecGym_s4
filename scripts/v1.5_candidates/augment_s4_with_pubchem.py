#!/usr/bin/env python
"""Augment MassSpecGym S4 retrieval candidates with PubChem candidates for
queries where the S4 bucket has fewer than 8 entries.

Rule (per Roman):
  - Keep ALL existing S4 candidates as-is (query at index 0, then S4-generated).
  - Append PubChem candidates whose 2D-InChIKey is NOT already present in the
    S4 list. Append in PubChem's existing order.
  - Stop once total list length reaches 512.
  - Apply only to queries where the original S4 list length < 8.

Source PubChem: ``MassSpecGym1.5_retrieval_candidates_{formula,mass}.json``
(re-canonicalised v1.5 release; keys align with S4 JSONs).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")

S4_FORMULA  = DATA / "MassSpecGym_S4_retrieval_candidates_formula.json"
S4_MASS     = DATA / "MassSpecGym_S4_retrieval_candidates_mass.json"
PC_FORMULA  = DATA / "MassSpecGym1.5_retrieval_candidates_formula.json"
PC_MASS     = DATA / "MassSpecGym1.5_retrieval_candidates_mass.json"

OUT_FORMULA = DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula.json"
OUT_MASS    = DATA / "MassSpecGym_S4plusPC_retrieval_candidates_mass.json"

THRESHOLD = 8       # augment when len(s4_list) < this
CAP = 512           # final list cap; must match MAX_CANDIDATES in build_msg_s4_candidates.py


# Worker pool globals.
_WORKER_SMILES_TO_IK: dict[str, str | None] = {}


def _worker_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _worker_ik(smi: str):
    """Compute 2D-InChIKey for a SMILES; return (smi, ik2d_or_None)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi, None
    try:
        ik = Chem.MolToInchiKey(m)
        return smi, ik[:14] if ik else None
    except Exception:
        return smi, None


def _ik_for_set(smiles_set: set[str], n_workers: int) -> dict[str, str | None]:
    if not smiles_set:
        return {}
    import multiprocessing as mp
    items = list(smiles_set)
    if n_workers <= 1:
        return dict(_worker_ik(s) for s in items)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        return dict(pool.imap_unordered(_worker_ik, items, chunksize=512))


def _augment(label: str, s4_path: Path, pc_path: Path, out_path: Path, n_workers: int) -> dict:
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    with s4_path.open() as f:
        s4 = json.load(f)
    print(f"[load] S4 {s4_path.name}: {len(s4):,} keys in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    with pc_path.open() as f:
        pc = json.load(f)
    print(f"[load] PubChem v1.5 {pc_path.name}: {len(pc):,} keys in {time.perf_counter()-t0:.1f}s")

    # Identify queries needing augmentation.
    short_keys = [q for q in s4 if len(s4[q]) < THRESHOLD]
    short_in_pc = [q for q in short_keys if q in pc]
    print(f"[scan] queries with S4 len < {THRESHOLD}: {len(short_keys):,}")
    print(f"[scan]   ...of which present in PubChem v1.5: {len(short_in_pc):,}")

    # Collect all SMILES we need to compute 2D-InChIKey for: the S4 candidates of
    # affected queries + the PubChem candidates we'd consider adding.
    smiles_to_resolve: set[str] = set()
    for q in short_in_pc:
        smiles_to_resolve.update(s4[q])
        smiles_to_resolve.update(pc[q][:CAP])  # don't bother beyond the cap
    print(f"[resolve] computing 2D-InChIKey for {len(smiles_to_resolve):,} unique SMILES "
          f"using {n_workers} workers ...")
    t0 = time.perf_counter()
    smi2ik = _ik_for_set(smiles_to_resolve, n_workers)
    print(f"[resolve]   done in {time.perf_counter()-t0:.1f}s; "
          f"{sum(1 for v in smi2ik.values() if v):,} resolved, "
          f"{sum(1 for v in smi2ik.values() if not v):,} unresolved")

    # Build augmented JSON. Copy untouched keys verbatim; augment short ones.
    out: dict[str, list[str]] = {}
    n_aug = n_added_total = 0
    list_size_before = list_size_after = 0
    for q, cands in s4.items():
        if len(cands) >= THRESHOLD or q not in pc:
            out[q] = cands
            continue
        existing_iks = {smi2ik.get(s) for s in cands if smi2ik.get(s)}
        merged = list(cands)
        for c in pc[q]:
            if len(merged) >= CAP:
                break
            ik = smi2ik.get(c)
            if not ik or ik in existing_iks or c in merged:
                continue
            merged.append(c)
            existing_iks.add(ik)
        n_added = len(merged) - len(cands)
        if n_added:
            n_aug += 1
            n_added_total += n_added
        list_size_before += len(cands)
        list_size_after += len(merged)
        out[q] = merged

    print(f"[aug] queries augmented: {n_aug:,} / {len(short_in_pc):,}")
    print(f"[aug] total PubChem candidates added: {n_added_total:,}")
    if n_aug:
        print(f"[aug] mean list size: before={list_size_before/len(short_in_pc):.1f} "
              f"after={list_size_after/len(short_in_pc):.1f}")

    # Stats on the full output JSON.
    import numpy as np
    L = np.asarray([len(v) for v in out.values()])
    n_trivial = int((L == 1).sum())
    n_lt8 = int((L < THRESHOLD).sum())
    print(f"[stats] OUTPUT — n_queries={len(out):,} "
          f"median={int(np.median(L))} mean={L.mean():.1f} p90={int(np.percentile(L,90))} max={int(L.max())}")
    print(f"[stats] trivial (len=1): {n_trivial:,} ({n_trivial/len(out)*100:.2f}%)")
    print(f"[stats] len<{THRESHOLD}: {n_lt8:,} ({n_lt8/len(out)*100:.2f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[write] wrote {out_path} ({size_mb:.1f} MB)")

    return {
        "label": label,
        "n_queries": len(out),
        "n_short_input": len(short_keys),
        "n_short_in_pc": len(short_in_pc),
        "n_augmented": n_aug,
        "n_pc_added_total": n_added_total,
        "median_after": int(np.median(L)),
        "mean_after": float(L.mean()),
        "trivial_after": n_trivial,
        "lt8_after": n_lt8,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-workers", type=int, default=int(os.environ.get("WORKERS", 32)))
    args = p.parse_args()

    results = []
    results.append(_augment("FORMULA", S4_FORMULA, PC_FORMULA, OUT_FORMULA, args.n_workers))
    results.append(_augment("MASS",    S4_MASS,    PC_MASS,    OUT_MASS,    args.n_workers))

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
