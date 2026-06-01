#!/usr/bin/env python
"""Generate per-query Molpher morphs for MassSpecGym queries (molpher env).

For each unique query molecule, run Molpher's RerouteBond operator to produce
formula-preserving structural isomers, then keep the TOP-N most similar to the
query by Tanimoto (Morgan r=2/2048). Output JSON {query_smiles: [morph, ...]}.

Runs in the molpher-lib env (RDKit 2021). Morphs are PROVISIONAL here — they are
re-canonicalised with the main env's rdkit==2023.9.4 in
postprocess_molpher_morphs.py, so consistency is enforced there. Morgan/Tanimoto
ranking is version-stable, so selecting top-N here is safe.

Reads queries from a plain text file (one SMILES per line) so this env needs no
pyarrow. Multi-day on ~28.7k queries → checkpoint/resume every interval.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

from molpher.core import MolpherMol
from molpher.core.morphing import Molpher
from molpher.core.morphing.operators import RerouteBond

MOLPHER_ATTEMPTS = 10_000
TOP_N = 512  # keep the 512 most-similar morphs per query
RADIUS = 2
N_BITS = 2048


def _fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def _morphs_for(query: str) -> list[str]:
    """Return up to TOP_N formula-preserving morphs most similar to ``query``."""
    try:
        source = MolpherMol(query)
        original = source.asRDMol()
    except Exception:
        return []
    if original is None:
        return []
    try:
        ref_formula = rdMolDescriptors.CalcMolFormula(original)
        query_ik2d = Chem.MolToInchiKey(original).split("-")[0]
        query_fp = _fp(original)
    except Exception:
        return []

    collected: dict[str, "Chem.Mol"] = {}

    def collector(morph, operator):
        # Keep only same-formula morphs that are not the query's own 2D structure.
        try:
            m = morph.asRDMol()
            if m is None or rdMolDescriptors.CalcMolFormula(m) != ref_formula:
                return
            if Chem.MolToInchiKey(m).split("-")[0] == query_ik2d:
                return
            smi = Chem.MolToSmiles(m, canonical=True)
            if smi not in collected:
                collected[smi] = m
        except Exception:
            return

    try:
        Molpher(source, operators=[RerouteBond()],
                attempts=MOLPHER_ATTEMPTS, collectors=[collector])()
    except Exception:
        return []
    if not collected:
        return []

    smis = list(collected.keys())
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, [_fp(collected[s]) for s in smis])
    ranked = sorted(range(len(smis)), key=lambda i: sims[i], reverse=True)
    return [smis[i] for i in ranked[:TOP_N]]


def _process(query: str) -> tuple[str, list[str]]:
    return query, _morphs_for(query)


def _load_ckpt(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _save_ckpt(d: dict, p: Path) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.rename(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-query Molpher morphs.")
    ap.add_argument("--queries-txt", type=Path, required=True, help="one SMILES per line")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--n-workers", type=int, default=32)
    ap.add_argument("--checkpoint-interval", type=int, default=200)
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    queries = [ln.strip() for ln in args.queries_txt.read_text().splitlines() if ln.strip()]
    queries = list(dict.fromkeys(queries))  # unique, order-preserving
    logging.info("%d unique queries", len(queries))

    ckpt_path = args.out_json.with_suffix(".checkpoint.json")
    done = _load_ckpt(ckpt_path)
    todo = [q for q in queries if q not in done]
    logging.info("resume: %d done, %d todo", len(done), len(todo))

    if todo:
        workers = min(args.n_workers, multiprocessing.cpu_count())
        with multiprocessing.Pool(workers) as pool:
            for i, (q, morphs) in enumerate(pool.imap_unordered(_process, todo)):
                done[q] = morphs
                if (i + 1) % args.checkpoint_interval == 0:
                    _save_ckpt(done, ckpt_path)
                    logging.info("checkpoint %d/%d (last query: %d morphs)", i + 1, len(todo), len(morphs))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(done))
    import numpy as np
    sizes = np.array([len(v) for v in done.values()])
    logging.info("DONE %d queries; morphs/query median=%d mean=%.1f zero=%d",
                 len(done), int(np.median(sizes)), sizes.mean(), int((sizes == 0).sum()))
    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
