#!/usr/bin/env python
"""Per-query separation table — baseline for the hybrid decoy strategy.

separation = closest_test_T - closest_decoy_T
  closest_test_T  : max Morgan r2/2048 Tanimoto from a test GT to any OTHER
                    test-fold GT (different 2D-InChIKey). The "real-neighbour"
                    density the decoys are up against.
  closest_decoy_T : max Tanimoto from the GT to its CURRENT decoys (existing
                    candidate JSON, GT at index 0).

Big separation = GT resembles a real test molecule but no decoy -> the
structure-only shortcut works. The hybrid sampler shrinks this without (a)
making the GT the set centroid or (b) pushing closest_decoy_T toward 1.0.

Writes <out_csv> sorted by separation desc:
  separation, gt_smiles, gt_inchikey_2d, gt_formula, n_decoys,
  closest_test_T, closest_test_smiles, closest_decoy_T
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RADIUS, NBITS = 2, 2048
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries-parquet", required=True)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--fold", default="test")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")

    q = pd.read_parquet(args.queries_parquet)
    q = q[q["fold"] == args.fold].reset_index(drop=True)
    qfps = [_fp(s) for s in q["smiles"]]
    keep = [i for i, f in enumerate(qfps) if f is not None]
    q = q.iloc[keep].reset_index(drop=True)
    qfps = [qfps[i] for i in keep]
    iks = q["inchikey_2d"].tolist()
    smis = q["smiles"].tolist()
    n = len(q)
    print(f"[sep] {n} {args.fold}-fold queries with valid FPs", flush=True)

    # closest_test_T: nearest OTHER test GT (exclude same 2D-InChIKey via codes).
    ik_codes = pd.factorize(iks)[0]
    closest_test_T = np.zeros(n)
    closest_test_smiles = [""] * n
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(qfps[i], qfps))
        sims[ik_codes == ik_codes[i]] = -1.0  # exclude self + same-ik2d dupes
        j = int(sims.argmax())
        closest_test_T[i] = sims[j]
        closest_test_smiles[i] = smis[j]
    print("[sep] closest_test_T done", flush=True)

    # closest_decoy_T from the current candidate JSON, matched by 2D-InChIKey
    # (robust to canonicalisation differences in the old/spoiled JSON keys).
    cands: dict[str, list[str]] = json.load(open(args.candidates_json))
    test_ik = set(iks)
    ik_to_decoys: dict[str, list[str]] = {}
    for k, vals in cands.items():
        kk = _ik2d(k)
        if kk in test_ik and kk not in ik_to_decoys:
            ik_to_decoys[kk] = vals[1:]  # GT at index 0
    print(f"[sep] matched {len(ik_to_decoys)}/{n} test queries to candidate JSON", flush=True)

    closest_decoy_T = np.zeros(n)
    n_decoys = np.zeros(n, dtype=int)
    for i in range(n):
        decoys = ik_to_decoys.get(iks[i], [])
        n_decoys[i] = len(decoys)
        dfps = [f for f in (_fp(s) for s in decoys) if f is not None]
        closest_decoy_T[i] = max(DataStructs.BulkTanimotoSimilarity(qfps[i], dfps)) if dfps else 0.0
    print("[sep] closest_decoy_T done", flush=True)

    out = pd.DataFrame({
        "separation": closest_test_T - closest_decoy_T,
        "gt_smiles": smis,
        "gt_inchikey_2d": iks,
        "gt_formula": q["formula"].tolist(),
        "n_decoys": n_decoys,
        "closest_test_T": closest_test_T,
        "closest_test_smiles": closest_test_smiles,
        "closest_decoy_T": closest_decoy_T,
    }).sort_values("separation", ascending=False).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[sep] wrote {args.out_csv}  ({len(out)} rows)", flush=True)
    print("[sep] TOP 8 by separation:\n" + out.head(8).to_string(), flush=True)
    print("[sep] VERIFY C48H81N9O8 (expect test_T~0.95, decoy_T~0.17, sep~0.78, n_decoys~28):\n"
          + out[out["gt_formula"] == "C48H81N9O8"].to_string(), flush=True)


if __name__ == "__main__":
    main()
