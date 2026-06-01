#!/usr/bin/env python
"""Augment MSG S4+PubChem formula candidates with Molpher RerouteBond morphs
for queries that still have len<8 after the PubChem fallback step.

Rule (per Roman):
  - Keep ALL existing S4+PubChem candidates as-is.
  - For queries with len(list) < 8, generate Molpher formula-preserving morphs
    (RerouteBond operator), filter by 2D-InChIKey uniqueness vs the existing
    list AND the query itself, and select via greedy farthest-first Tanimoto
    diversity.
  - Append until total list reaches the 512 cap.

Requires the conda molpher env at
``/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/molpher/.venv``.

Modelled on `molpher/scripts/fill_formula_candidates_molpher.py`.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors

from molpher.core import MolpherMol
from molpher.core.morphing import Molpher
from molpher.core.morphing.operators import RerouteBond
from tqdm import tqdm

DATA = Path("/pfs/lustrep2/scratch/project_465002061/rbushuie/DreaMS-Mol_dev/MassSpecGym/data")
IN_JSON = DATA / "MassSpecGym_S4plusPC_retrieval_candidates_formula.json"
OUT_JSON = DATA / "MassSpecGym_S4plusPCplusMol_retrieval_candidates_formula.json"
CHECKPOINT_JSON = DATA / "MassSpecGym_S4plusPCplusMol_checkpoint.json"

THRESHOLD = 8
CAP = 512  # must match MAX_CANDIDATES in build_msg_s4_candidates.py
MOLPHER_ATTEMPTS = 10_000
RANDOM_SEED = 42
N_WORKERS = int(os.environ.get("WORKERS", 32))
CHECKPOINT_INTERVAL = 50


def canonicalize(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else smi


def inchikey_2d(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        ik = Chem.MolToInchiKey(m)
        return ik[:14] if ik else None
    except Exception:
        return None


def pick_diverse_indices(fps, k: int) -> list[int]:
    """Greedy farthest-first selection by Tanimoto distance."""
    if not fps or k <= 0:
        return []
    k = min(k, len(fps))
    chosen = [0]
    available = list(range(1, len(fps)))
    while len(chosen) < k and available:
        best_idx, best_min_dist = None, -1.0
        chosen_fps = [fps[i] for i in chosen]
        for idx in available:
            sims = DataStructs.BulkTanimotoSimilarity(fps[idx], chosen_fps)
            min_dist = min(1.0 - s for s in sims)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        if best_idx is None:
            break
        chosen.append(best_idx)
        available.remove(best_idx)
    return chosen


def generate_morphs(query: str, existing_canon: set[str], max_needed: int) -> list[str]:
    """Generate formula-preserving Molpher morphs, pick diverse subset."""
    if max_needed <= 0:
        return []
    try:
        source = MolpherMol(query)
    except Exception:
        return []
    original = source.asRDMol()
    if original is None:
        return []
    try:
        ref_formula = rdMolDescriptors.CalcMolFormula(original)
        query_ik2d = Chem.MolToInchiKey(original).split("-")[0]
    except Exception:
        return []
    collected: dict[str, Chem.Mol] = {}

    def collector(morph, operator):
        try:
            rd_mol = morph.asRDMol()
            if rd_mol is None:
                return
            if rdMolDescriptors.CalcMolFormula(rd_mol) != ref_formula:
                return
            ik2d = Chem.MolToInchiKey(rd_mol).split("-")[0]
            if ik2d == query_ik2d:
                return
            smi = Chem.MolToSmiles(rd_mol, canonical=True)
            if smi in existing_canon or smi in collected:
                return
            collected[smi] = rd_mol
        except Exception:
            return

    try:
        molpher = Molpher(source, operators=[RerouteBond()], attempts=MOLPHER_ATTEMPTS, collectors=[collector])
        molpher()
    except Exception:
        return []

    if not collected:
        return []
    smiles_list = list(collected.keys())
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        m = collected[smi]
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        except Exception:
            continue
        fps.append(fp)
        valid_smiles.append(smi)
    if not fps:
        return []
    idx = pick_diverse_indices(fps, min(max_needed, len(fps)))
    return [valid_smiles[i] for i in idx]


def process_one(args):
    """Run for one (query, current_list) pair. Returns (query, new_list, n_added)."""
    query, current = args
    max_needed = CAP - len(current)
    if max_needed <= 0:
        return query, current, 0
    existing_canon = {canonicalize(s) for s in current}
    existing_iks = {inchikey_2d(s) for s in current if inchikey_2d(s)}
    morphs = generate_morphs(query, existing_canon, max_needed)
    added = []
    for m in morphs:
        ik = inchikey_2d(m)
        if ik and ik not in existing_iks:
            existing_iks.add(ik)
            added.append(m)
            if len(current) + len(added) >= CAP:
                break
    return query, current + added, len(added)


def load_checkpoint(p: Path) -> dict:
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {}


def save_checkpoint(data: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    tmp.rename(p)


def main() -> None:
    random.seed(RANDOM_SEED)
    RDLogger.DisableLog("rdApp.*")
    logging.info(f"Loading {IN_JSON} ...")
    with IN_JSON.open() as f:
        cands: dict[str, list[str]] = json.load(f)
    logging.info(f"Loaded {len(cands):,} queries")

    short = [q for q in cands if len(cands[q]) < THRESHOLD]
    logging.info(f"queries with len<{THRESHOLD}: {len(short):,}")

    checkpoint = load_checkpoint(CHECKPOINT_JSON)
    if checkpoint:
        logging.info(f"Resumed from checkpoint: {len(checkpoint):,} entries")

    tasks = [(q, cands[q]) for q in short if q not in checkpoint]
    logging.info(f"Remaining tasks: {len(tasks):,} (already done: {len(checkpoint):,})")

    updated = dict(cands)
    updated.update(checkpoint)

    total_added = 0
    n_aug = 0
    if tasks:
        workers = min(N_WORKERS, multiprocessing.cpu_count())
        logging.info(f"using {workers} workers")
        with multiprocessing.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(process_one, tasks)
            for i, (q, new_list, n_added) in enumerate(tqdm(iterator, total=len(tasks))):
                updated[q] = new_list
                checkpoint[q] = new_list
                total_added += n_added
                if n_added > 0:
                    n_aug += 1
                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(checkpoint, CHECKPOINT_JSON)
                    logging.info(f"Checkpoint: {i + 1}/{len(tasks)} processed, {total_added:,} morphs added")

    logging.info(f"Molpher fill done. {n_aug:,}/{len(short):,} queries got new morphs; "
                 f"{total_added:,} morphs added total.")

    L = np.asarray([len(v) for v in updated.values()])
    n_trivial = int((L == 1).sum())
    n_lt8 = int((L < THRESHOLD).sum())
    logging.info(f"FINAL — n_queries={len(updated):,} median={int(np.median(L))} mean={L.mean():.1f} "
                 f"p90={int(np.percentile(L,90))} max={int(L.max())}")
    logging.info(f"FINAL — trivial (len=1): {n_trivial:,} ({n_trivial/len(updated)*100:.2f}%)")
    logging.info(f"FINAL — len<{THRESHOLD}: {n_lt8:,} ({n_lt8/len(updated)*100:.2f}%)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(updated, f)
    logging.info(f"Wrote {OUT_JSON} ({os.path.getsize(OUT_JSON)/1e6:.1f} MB)")
    if CHECKPOINT_JSON.exists():
        CHECKPOINT_JSON.unlink()
        logging.info("Removed checkpoint")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
