#!/usr/bin/env python
"""Re-canonicalise raw PubChem with RDKit into an attribute-complete,
2D-InChIKey-deduplicated candidate pool for MassSpecGym.

Input : raw PubChem TSV with a ``smiles`` column. PubChem's own ``formula`` /
        ``mass`` are IGNORED — its mass is average MW, not monoisotopic.
Output: ``<out>.parquet`` with columns recomputed and SAVED on disk:
            smiles        RDKit-canonical SMILES (rdkit==2023.9.4)
            inchikey_2d   first 14 chars of InChIKey (connectivity layer)
            formula       CalcMolFormula
            exact_mass    CalcExactMolWt (monoisotopic)

Labels are recomputed FROM the canonical SMILES (canonicalisation first, since
it can alter the molecule), using the SAME canonicaliser as the S4 pool
(``massspecgym.utils.rdkit_canonical_smiles``) so all three pools are mutually
consistent. Persisting the labels means merge / candidate emission never
recompute them.

Dedup keeps the first occurrence per 2D-InChIKey in input order (deterministic).
Streams the input in chunks; pure-RDKit canon of ~123 M rows is a single-shot
~1-2 h job, so give the PBS job ample walltime rather than resuming.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger

_SCHEMA = pa.schema([
    ("smiles", pa.string()),
    ("inchikey_2d", pa.string()),
    ("formula", pa.string()),
    ("exact_mass", pa.float64()),
])


def _worker_init() -> None:
    RDLogger.DisableLog("rdApp.*")


def _canon_full(smi: str) -> tuple[str, str, str, float] | None:
    """Canonicalise then recompute (smiles, inchikey_2d, formula, exact_mass).

    Returns None if RDKit cannot canonicalise/parse the SMILES or derive a key.
    """
    if not isinstance(smi, str) or not smi:
        return None
    from massspecgym.utils import rdkit_canonical_smiles
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
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
        ik = Chem.MolToInchiKey(mol)
        ik2d = ik[:14] if ik else None
        formula = rdMolDescriptors.CalcMolFormula(mol)
        exact_mass = float(rdMolDescriptors.CalcExactMolWt(mol))
    except Exception:
        return None
    if not ik2d:
        return None
    return std, ik2d, formula, exact_mass


def main() -> None:
    p = argparse.ArgumentParser(description="Build RDKit-canonical PubChem pool.")
    p.add_argument("--input-tsv", type=Path, required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--out-parquet", type=Path, required=True)
    p.add_argument("--n-workers", type=int, default=48)
    p.add_argument("--chunk-size", type=int, default=2_000_000)
    args = p.parse_args()

    RDLogger.DisableLog("rdApp.*")
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()  # 2D-InChIKeys already written (dedup, keep-first)
    n_read = 0
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    reader = pd.read_csv(args.input_tsv, sep="\t", usecols=[args.smiles_col],
                         chunksize=args.chunk_size, dtype=str, na_filter=False)
    with pq.ParquetWriter(args.out_parquet, _SCHEMA, compression="zstd") as writer, \
            ctx.Pool(args.n_workers, initializer=_worker_init) as pool:
        for chunk in reader:
            # pool.map preserves order so keep-first dedup is deterministic.
            results = pool.map(_canon_full, chunk[args.smiles_col].tolist(), chunksize=1000)
            keep_smiles, keep_ik, keep_formula, keep_mass = [], [], [], []
            for r in results:
                if r is None:
                    continue
                std, ik2d, formula, mass = r
                if ik2d in seen:
                    continue
                seen.add(ik2d)
                keep_smiles.append(std)
                keep_ik.append(ik2d)
                keep_formula.append(formula)
                keep_mass.append(mass)
            if keep_smiles:
                writer.write_table(pa.table({
                    "smiles": keep_smiles, "inchikey_2d": keep_ik,
                    "formula": keep_formula, "exact_mass": keep_mass,
                }, schema=_SCHEMA))
            n_read += len(chunk)
            rate = n_read / max(1e-9, time.perf_counter() - t0)
            print(f"[pubchem] read={n_read:,} unique={len(seen):,} "
                  f"({rate:.0f} rows/s, {time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"[pubchem] DONE read={n_read:,} unique={len(seen):,} -> {args.out_parquet}", flush=True)


if __name__ == "__main__":
    main()
