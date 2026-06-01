#!/usr/bin/env python
"""Post-process Molpher morphs in the main env (rdkit==2023.9.4).

Reads provisional per-query morphs from generate_molpher_morphs.py and:
  - re-canonicalises each morph with ``rdkit_canonical_smiles`` (pipeline canon),
  - parse-checks (drops anything RDKit cannot parse),
  - validates formula == query formula (RerouteBond preserves formula; recanon
    rarely alters it — drop mismatches, mirroring the MSnGym pipeline),
  - dedups by 2D-InChIKey within a query and excludes the query's own ik2d,
  - preserves the Tanimoto-to-query ranking from generation.

Writes (labels recomputed and SAVED so emit never recomputes):
  <out_json>          {query_smiles: [canonical morphs, most-similar first]}
  <out_pool_parquet>  unique morphs across all queries:
                      smiles, inchikey_2d, formula, exact_mass  (for FP step)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


def _worker_init() -> None:
    RDLogger.DisableLog("rdApp.*")


def _canon(smi: str) -> tuple[str, str, str, float] | None:
    """(canonical_smiles, inchikey_2d, formula, exact_mass) or None."""
    from massspecgym.utils import rdkit_canonical_smiles
    try:
        std = rdkit_canonical_smiles(smi)
    except Exception:
        return None
    if not std:
        return None
    mol = Chem.MolFromSmiles(std)
    if mol is None:
        return None
    try:
        ik2d = Chem.MolToInchiKey(mol)[:14]
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mass = float(rdMolDescriptors.CalcExactMolWt(mol))
    except Exception:
        return None
    if not ik2d:
        return None
    return std, ik2d, formula, mass


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonicalise + validate Molpher morphs.")
    ap.add_argument("--in-json", type=Path, required=True)
    ap.add_argument("--queries-parquet", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-pool-parquet", type=Path, required=True)
    ap.add_argument("--n-workers", type=int, default=48)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")

    gen: dict[str, list[str]] = json.loads(args.in_json.read_text())
    qdf = pq.read_table(args.queries_parquet,
                        columns=["smiles", "formula", "inchikey_2d"]).to_pandas()
    q_formula = dict(zip(qdf["smiles"], qdf["formula"]))
    q_ik = dict(zip(qdf["smiles"], qdf["inchikey_2d"]))

    # Canonicalise every unique raw morph once, in parallel.
    uniq = sorted({m for ms in gen.values() for m in ms})
    print(f"[post] {len(gen):,} queries, {len(uniq):,} unique raw morphs", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.n_workers, initializer=_worker_init) as pool:
        results = pool.map(_canon, uniq, chunksize=1000)
    canon = {raw: r for raw, r in zip(uniq, results) if r is not None}
    print(f"[post] canonicalised {len(canon):,}/{len(uniq):,}", flush=True)

    out: dict[str, list[str]] = {}
    pool_rows: dict[str, tuple[str, str, float]] = {}  # ik2d -> (smiles, formula, mass)
    n_drop_parse = n_drop_formula = 0
    for q, morphs in gen.items():
        qf = q_formula.get(q)
        qik = q_ik.get(q)
        seen: set[str] = set()
        kept: list[str] = []
        for raw in morphs:  # already Tanimoto-ranked (most similar first)
            r = canon.get(raw)
            if r is None:
                n_drop_parse += 1
                continue
            std, ik2d, formula, mass = r
            if qf is not None and formula != qf:
                n_drop_formula += 1
                continue
            if ik2d == qik or ik2d in seen:
                continue
            seen.add(ik2d)
            kept.append(std)
            pool_rows.setdefault(ik2d, (std, formula, mass))
        out[q] = kept

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out))

    sm, ik, fo, ma = [], [], [], []
    for ik2d, (std, formula, mass) in pool_rows.items():
        sm.append(std); ik.append(ik2d); fo.append(formula); ma.append(mass)
    pq.write_table(
        pa.table({"smiles": sm, "inchikey_2d": ik, "formula": fo, "exact_mass": ma}),
        args.out_pool_parquet, compression="zstd",
    )

    sizes = np.array([len(v) for v in out.values()]) if out else np.zeros(0)
    print(f"[post] dropped parse={n_drop_parse:,} formula_mismatch={n_drop_formula:,}", flush=True)
    print(f"[post] morphs/query median={int(np.median(sizes))} mean={sizes.mean():.1f} "
          f"zero={int((sizes == 0).sum()):,}", flush=True)
    print(f"[post] unique morph pool={len(pool_rows):,} -> {args.out_pool_parquet}", flush=True)
    print(f"[post] per-query -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
