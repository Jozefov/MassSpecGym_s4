#!/usr/bin/env python
"""Compute packed Morgan fingerprints for a candidate-pool parquet.

Reads a parquet with a SMILES column in batches, parses each SMILES (dropping
rows RDKit cannot parse), computes Morgan r=2 / 2048-bit fingerprints, and
writes FP-aligned artefacts:

  <prefix>_morgan_r2_2048_fps.bin      uint8  (N, 256)  np.packbits(2048 bits)
  <prefix>_morgan_r2_2048_bitsums.bin  uint16 (N,)       popcount per fingerprint
  <prefix>_morgan_r2_2048_meta.json    run metadata
  <prefix>_clean.parquet               surviving input rows, in FP order

Row i of each binary corresponds to row i of <prefix>_clean.parquet, so the
binaries can be memory-mapped and indexed directly for fast Tanimoto search.

SMILES are assumed already RDKit-canonical (every pool canonicalises upstream
with rdkit==2023.9.4); this script only parse-validates them. Deduplication by
2D-InChIKey is the pool builder's responsibility and is verified separately.

The 2048-bit recipe matches the MSnGym pool and the retrieval_hacking_challenge
audit, so decoy selection and leakage measurement use the same metric.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
N_BITS = 2048
PACKED_WIDTH = N_BITS // 8  # 256 bytes per fingerprint

# Bits set per byte value 0..255, for vectorised popcount over packed FPs.
_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint16)
)

_GEN = None  # per-worker Morgan generator (set in _worker_init)


def _worker_init() -> None:
    global _GEN
    RDLogger.DisableLog("rdApp.*")
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def _pack(smi: str) -> bytes | None:
    """Return the 256-byte packed Morgan FP for ``smi``, or None if unparseable."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None:
        return None
    bits = np.zeros((N_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(_GEN.GetFingerprint(mol), bits)
    return np.packbits(bits).tobytes()


def main() -> None:
    p = argparse.ArgumentParser(description="Pack Morgan FPs for a pool parquet.")
    p.add_argument("--input-parquet", type=Path, required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="Output path stem, e.g. .../pools/s4/s4_generated_pool")
    p.add_argument("--n-workers", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=1_000_000)
    args = p.parse_args()

    RDLogger.DisableLog("rdApp.*")
    tag = f"morgan_r{RADIUS}_{N_BITS}"
    stem = args.out_prefix.name
    fps_path = args.out_prefix.with_name(f"{stem}_{tag}_fps.bin")
    bitsums_path = args.out_prefix.with_name(f"{stem}_{tag}_bitsums.bin")
    meta_path = args.out_prefix.with_name(f"{stem}_{tag}_meta.json")
    clean_path = args.out_prefix.with_name(f"{stem}_clean.parquet")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.input_parquet)
    total_in = pf.metadata.num_rows
    print(f"[fps] {args.input_parquet} rows={total_in:,} cols={pf.schema_arrow.names}", flush=True)

    ctx = mp.get_context("fork")
    n_valid = n_invalid = 0
    bitsums_chunks: list[np.ndarray] = []
    writer: pq.ParquetWriter | None = None
    t0 = time.perf_counter()
    with open(fps_path, "wb") as f_fps, \
            ctx.Pool(args.n_workers, initializer=_worker_init) as pool:
        for bi, batch in enumerate(pf.iter_batches(batch_size=args.batch_size)):
            df = batch.to_pandas()
            # pool.map preserves order, so packed[i] aligns with df row i.
            packed = pool.map(_pack, df[args.smiles_col].tolist(), chunksize=2000)
            mask = np.fromiter((b is not None for b in packed), dtype=bool, count=len(packed))
            valid_bytes = b"".join(b for b in packed if b is not None)
            f_fps.write(valid_bytes)
            arr = np.frombuffer(valid_bytes, dtype=np.uint8).reshape(-1, PACKED_WIDTH)
            bitsums_chunks.append(_POPCOUNT[arr].sum(axis=1).astype(np.uint16))

            valid_table = pa.Table.from_pandas(df.loc[mask].reset_index(drop=True),
                                               preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(clean_path, valid_table.schema, compression="zstd")
            writer.write_table(valid_table)

            n_valid += int(mask.sum())
            n_invalid += int((~mask).sum())
            print(f"[fps] batch {bi}: +{int(mask.sum()):,} valid / {int((~mask).sum()):,} invalid"
                  f"  (cum {n_valid:,}/{total_in:,}, {time.perf_counter()-t0:.0f}s)", flush=True)

    if writer is not None:
        writer.close()
    bitsums = np.concatenate(bitsums_chunks) if bitsums_chunks else np.zeros(0, np.uint16)
    bitsums.tofile(bitsums_path)

    meta = {
        "n_molecules": n_valid, "n_invalid": n_invalid, "total_rows_in": total_in,
        "radius": RADIUS, "n_bits": N_BITS, "packed_width": PACKED_WIDTH,
        "smiles_col": args.smiles_col, "input_parquet": str(args.input_parquet),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[fps] DONE valid={n_valid:,} invalid={n_invalid:,}", flush=True)
    print(f"[fps] -> {fps_path}\n        {bitsums_path}\n        {clean_path}\n        {meta_path}")


if __name__ == "__main__":
    main()
