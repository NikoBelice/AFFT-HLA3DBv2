#!/usr/bin/env python3
"""
Validation-like inference with fine-tuned AlphaFold parameters, without native structures.

Goal
----
Predict new structures while matching the fine-tuning validation configuration as
closely as possible, but without requiring:
  - native_pdbfile
  - native_alignstring
  - native coordinates
  - anchor_class_file
  - D-score evaluation

Important
---------
This is NOT mathematically identical to the original --exact_validation branch,
because exact validation used predict_utils.create_batch_for_training(), which
requires native structure information. This script instead uses the normal
prediction feature pathway while preserving validation-like model/preprocessing
settings and deterministic RNG behavior.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import os
import pickle
import random
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from alphafold.model import config
from alphafold.model import model

import predict_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-like AlphaFold inference with fine-tuned weights "
            "without native structures."
        )
    )

    parser.add_argument(
        "--targets",
        required=True,
        help=(
            "TSV containing target_chainseq and templates_alignfile. "
            "Optional targetid is used for output naming."
        ),
    )
    parser.add_argument(
        "--params_file",
        required=True,
        help="Fine-tuned parameter pickle saved by the training script.",
    )
    parser.add_argument(
        "--outfile_prefix",
        required=True,
        help="Prefix used for output files.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for predicted PDB, metrics, and final TSV outputs.",
    )

    # Defaults chosen to mirror the fine-tuning validation settings in run_predictionv3.py.
    parser.add_argument("--model_name", default="model_2_ptm")
    parser.add_argument("--crop_size", type=int, default=190)
    parser.add_argument("--msa_clusters", type=int, default=5)
    parser.add_argument("--extra_msa", type=int, default=1)
    parser.add_argument("--num_evo_blocks", type=int, default=48)
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=None,
        help="Override AlphaFold configured number of recycles.",
    )
    parser.add_argument("--struc_viol_weight", type=float, default=1.0)

    # Fine-tuning exact validation forced MSA resampling on.
    parser.add_argument(
        "--no_resample_msa",
        action="store_true",
        help=(
            "Disable MSA resampling. By default it is ENABLED to match "
            "fine-tuning validation."
        ),
    )

    parser.add_argument(
        "--validation_index_col",
        default=None,
        help=(
            "Optional TSV column containing the original 0-based validation index. "
            "If omitted, current row order is used. This index is used for "
            "deterministic validation-like RNG seeding."
        ),
    )

    parser.add_argument(
        "--ignore_identities",
        action="store_true",
        help="Ignore expected template identity checks.",
    )
    parser.add_argument(
        "--no_pdbs",
        action="store_true",
        help="Do not write predicted PDB files.",
    )
    parser.add_argument(
        "--terse",
        action="store_true",
        help="Reduce auxiliary output written by predict_utils.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def seed_everything(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def load_finetuned_runner(args):
    model_config = config.model_config(args.model_name)

    resample_msa = not args.no_resample_msa

    model_config.data.common.resample_msa_in_recycling = resample_msa
    model_config.model.resample_msa_in_recycling = resample_msa
    model_config.data.common.max_extra_msa = args.extra_msa
    model_config.data.eval.max_msa_clusters = args.msa_clusters
    model_config.data.eval.crop_size = args.crop_size

    model_config.model.embeddings_and_evoformer.evoformer_num_block = (
        args.num_evo_blocks
    )
    model_config.model.heads.structure_module.structural_violation_loss_weight = (
        args.struc_viol_weight
    )

    if args.num_recycle is not None:
        model_config.model.num_recycle = args.num_recycle
        if hasattr(model_config.data.common, "num_recycle"):
            model_config.data.common.num_recycle = args.num_recycle

    with open(args.params_file, "rb") as handle:
        params = pickle.load(handle)

    runner = model.RunModel(model_config, params)
    return {args.model_name: runner}


def build_template_features(target_row, query_sequence, ignore_identities):
    alignfile = Path(target_row.templates_alignfile)

    if not alignfile.exists():
        raise FileNotFoundError(
            f"Template alignment file not found: {alignfile}"
        )

    alignment_df = pd.read_table(alignfile)
    template_features_list = []

    for template_number, row in alignment_df.iterrows():
        if int(row.target_len) != len(query_sequence):
            raise ValueError(
                f"target_len mismatch in {alignfile}: "
                f"{row.target_len} != {len(query_sequence)}"
            )

        target_to_template_alignment = {
            int(pair.split(":")[0]): int(pair.split(":")[1])
            for pair in str(row.target_to_template_alignstring).split(";")
            if pair
        }

        expected_identities = (
            None if ignore_identities else row.identities
        )

        template_features = predict_utils.create_single_template_features(
            query_sequence,
            row.template_pdbfile,
            target_to_template_alignment,
            f"T{template_number:03d}",
            allow_chainbreaks=True,
            allow_skipped_lines=True,
            expected_identities=expected_identities,
            expected_template_len=row.template_len,
        )
        template_features_list.append(template_features)

    return predict_utils.compile_template_features(template_features_list)


def run_prediction_compatibly(**kwargs):
    """
    Pass a deterministic seed when the local predict_utils implementation
    supports random_seed or seed.
    """
    signature = inspect.signature(
        predict_utils.run_alphafold_prediction
    )

    if "random_seed" in signature.parameters:
        kwargs["random_seed"] = kwargs.pop("_seed")
    elif "seed" in signature.parameters:
        kwargs["seed"] = kwargs.pop("_seed")
    else:
        kwargs.pop("_seed")

    return predict_utils.run_alphafold_prediction(**kwargs)


def find_prediction_pdb(row_prefix, model_name):
    prefix = Path(row_prefix)
    candidates = sorted(prefix.parent.glob(prefix.name + "*.pdb"))

    model_hits = [
        p for p in candidates
        if model_name in p.name
    ]

    if model_hits:
        return model_hits[-1]

    return candidates[-1] if candidates else None


def validation_like_seed(validation_index):
    """
    Match the spirit of exact-validation RNG:
        jax.random.fold_in(jax.random.PRNGKey(0), validation_index)

    run_alphafold_prediction accepts an integer seed in most local implementations,
    so collapse the folded JAX key deterministically into uint32.
    """
    key = jax.random.fold_in(
        jax.random.PRNGKey(0),
        int(validation_index),
    )
    key_np = np.asarray(key, dtype=np.uint32)
    return int(np.bitwise_xor.reduce(key_np))


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_table(args.targets)

    required = {"target_chainseq", "templates_alignfile"}
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(
            f"Targets TSV is missing columns: {sorted(missing)}"
        )

    target_lengths = [
        len(str(row.target_chainseq).replace("/", ""))
        for row in targets.itertuples()
    ]

    if max(target_lengths) > args.crop_size:
        raise ValueError(
            f"At least one target has {max(target_lengths)} residues, "
            f"which exceeds crop_size={args.crop_size}. "
            f"To mirror fine-tuning validation, keep crop_size=190 only if "
            f"all targets fit."
        )

    model_runners = load_finetuned_runner(args)

    if args.verbose:
        print("command:", " ".join(sys.argv))
        print("device:", jax.local_devices()[0])
        print("model_name:", args.model_name)
        print("params_file:", args.params_file)
        print("crop_size:", args.crop_size)
        print("msa_clusters:", args.msa_clusters)
        print("extra_msa:", args.extra_msa)
        print("num_evo_blocks:", args.num_evo_blocks)
        print("resample_msa:", not args.no_resample_msa)

    final_rows = []

    for counter, target_row in targets.iterrows():
        target_id = str(
            target_row.get("targetid", f"T{counter}")
        ).strip()

        validation_index = (
            int(target_row[args.validation_index_col])
            if args.validation_index_col
            else counter
        )

        seed = validation_like_seed(validation_index)
        seed_everything(seed)

        print(
            f"START: {counter + 1}/{len(targets)} "
            f"{target_id} validation_index={validation_index} seed={seed}",
            flush=True,
        )

        query_chainseq = str(target_row.target_chainseq)
        query_sequence = query_chainseq.replace("/", "")

        template_features = build_template_features(
            target_row,
            query_sequence,
            args.ignore_identities,
        )

        # Query-only MSA, as in the original prediction pathway.
        msa = [query_sequence]
        deletion_matrix = [[0] * len(query_sequence)]

        row_prefix_value = target_row.get(
            "outfile_prefix",
            f"{args.outfile_prefix}_{target_id}",
        )

        row_prefix = str(
            output_dir / Path(str(row_prefix_value)).name
        )

        all_metrics = run_prediction_compatibly(
            query_sequence=query_sequence,
            msa=msa,
            deletion_matrix=deletion_matrix,
            chainbreak_sequence=query_chainseq,
            template_features=template_features,
            model_runners=model_runners,
            out_prefix=row_prefix,
            crop_size=args.crop_size,
            dump_pdbs=not (args.no_pdbs or args.terse),
            dump_metrics=not args.terse,
            _seed=seed,
        )

        output_row = target_row.copy()
        output_row["validation_index_used"] = validation_index
        output_row["prediction_seed"] = seed

        predicted_pdb = find_prediction_pdb(
            row_prefix,
            args.model_name,
        )

        if predicted_pdb is not None:
            output_row["predicted_pdbfile"] = str(predicted_pdb)

        chains = query_chainseq.split("/")
        chain_stops = list(
            itertools.accumulate(
                len(chain) for chain in chains
            )
        )
        chain_starts = [0] + chain_stops[:-1]
        nres = chain_stops[-1]

        for model_name, metrics in all_metrics.items():
            plddt = np.asarray(metrics["plddt"])
            pae = metrics.get("predicted_aligned_error")
            pae = (
                None if pae is None
                else np.asarray(pae)
            )

            output_row[f"{model_name}_plddt"] = float(
                np.mean(plddt[:nres])
            )

            if pae is not None:
                output_row[f"{model_name}_pae"] = float(
                    np.mean(pae[:nres, :nres])
                )

            for chain1, (start1, stop1) in enumerate(
                zip(chain_starts, chain_stops)
            ):
                output_row[
                    f"{model_name}_plddt_{chain1}"
                ] = float(
                    np.mean(plddt[start1:stop1])
                )

                if pae is not None:
                    for chain2, (start2, stop2) in enumerate(
                        zip(chain_starts, chain_stops)
                    ):
                        output_row[
                            f"{model_name}_pae_{chain1}_{chain2}"
                        ] = float(
                            np.mean(
                                pae[
                                    start1:stop1,
                                    start2:stop2,
                                ]
                            )
                        )

        final_rows.append(output_row)

        print(
            f"DONE: {counter + 1}/{len(targets)} {target_id}",
            flush=True,
        )

    outfile = output_dir / (
        f"{Path(args.outfile_prefix).name}_final.tsv"
    )

    pd.DataFrame(final_rows).to_csv(
        outfile,
        sep="\t",
        index=False,
    )

    print("made:", outfile)


if __name__ == "__main__":
    main()
