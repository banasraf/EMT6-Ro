#!/usr/bin/env python3
"""
Replay a protocol set on the EMT6-Ro GPU backend.

For each protocol in the input file, run M replicates per tumor across
all 10 tumors in data/tumor-lib/. Emit one row per protocol with
(M * n_tumors) comma-separated final living-cell counts, matching the
layout of the historical results-gpu-*.txt files.

Memory check (Titan V 12 GB): a batch of P protocols * T tumors * M reps
holds ~60 B/site * 53*53 sites ~ 170 KB/sim. M*T*P = 1000 -> 170 MB.
Comfortable up to ~50000 sims in flight.

Usage:
    python3 tools/run_protocols_gpu.py \
        --protocols protocols-30.06.txt \
        --tumors data/tumor-lib/tumor-{1..10}.txt \
        --params data/default-parameters.json \
        --reps 100 --steps 144000 \
        --out /home/rafal/runs/audit-gpu-30.06/results.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure we can import the emt6ro module from the repo's python/ dir
HERE = Path(__file__).parent.resolve()
REPO = HERE.parent
sys.path.insert(0, str(REPO / "python"))

import numpy as np
from emt6ro.simulation import Experiment, load_parameters, load_state

from convert_protocols import parse_two_line, to_emt6_python_form


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocols", type=Path, required=True)
    p.add_argument("--tumors", nargs="+", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--steps", type=int, default=144000)
    p.add_argument("--resolution", type=int, default=300)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta-out", type=Path, default=None)
    p.add_argument("--protocol-chunk", type=int, default=10,
                   help="How many protocols to fuse into one Experiment batch")
    p.add_argument("--limit-protocols", type=int, default=None,
                   help="Run only the first N protocols (debug/pilot)")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.meta_out is None:
        args.meta_out = args.out.with_suffix(args.out.suffix + ".meta.json")

    print(f"Loading params: {args.params}")
    params = load_parameters(str(args.params))

    print(f"Loading {len(args.tumors)} tumors")
    tumors = [load_state(str(p), params) for p in args.tumors]

    print(f"Parsing protocols: {args.protocols}")
    protocols = parse_two_line(args.protocols)
    protocols = to_emt6_python_form(protocols)
    if args.limit_protocols:
        protocols = protocols[: args.limit_protocols]
    print(f"  {len(protocols)} protocols after canonicalization")

    n_t = len(tumors)
    n_p = len(protocols)
    M = args.reps

    # Pre-fill output rows (one per protocol).
    rows = [None] * n_p

    t0 = time.time()

    for pi_start in range(0, n_p, args.protocol_chunk):
        chunk = protocols[pi_start : pi_start + args.protocol_chunk]
        n_chunk = len(chunk)
        # Per-tumor: run all protocols in this chunk together.
        # Experiment expects num_protocols and num_tests. We loop tumors
        # outside because Experiment takes a list of tumors and re-uses them
        # internally for every protocol -- output shape is
        # (n_protocols, n_tumors, n_runs).
        # Use one Experiment per chunk to avoid retained state across chunks.

        tc0 = time.time()
        exp = Experiment(
            params,
            tumors,
            runs=M,
            protocols_num=n_chunk,
            simulation_steps=args.steps,
            protocol_resolution=args.resolution,
        )
        exp.add_irradiations(chunk)
        exp.run(args.steps)
        results = np.asarray(exp.get_results())  # (n_protocols, n_tumors, n_runs)
        # The historical file emits per protocol all (tumor, rep) values
        # as one row of (n_tumors * n_reps) values.
        for ci in range(n_chunk):
            row = results[ci].reshape(-1)  # n_tumors * n_runs
            rows[pi_start + ci] = row
        tc1 = time.time()
        print(
            f"  chunk {pi_start}..{pi_start+n_chunk-1} of {n_p}: "
            f"{n_chunk} protocols x {n_t} tumors x {M} reps in {tc1-tc0:.1f}s "
            f"({n_chunk*n_t*M/(tc1-tc0):.0f} sims/sec)"
        )

    t1 = time.time()
    print(f"Total: {n_p} protocols in {t1-t0:.1f}s")

    # Write rows to the output file in the historical layout.
    with args.out.open("w") as f:
        for row in rows:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    print(f"Wrote {args.out}: {n_p} rows")

    # Metadata
    args.meta_out.write_text(json.dumps({
        "n_protocols": n_p,
        "n_tumors": n_t,
        "reps_per_tumor": M,
        "steps": args.steps,
        "resolution": args.resolution,
        "tumors": [str(p) for p in args.tumors],
        "protocols_file": str(args.protocols),
        "params_file": str(args.params),
        "wall_time_sec": t1 - t0,
        "row_layout": "values per row are tumor-major then rep-major: [t0r0, t0r1, ..., t0r{M-1}, t1r0, ...]",
    }, indent=2))
    print(f"Wrote {args.meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
