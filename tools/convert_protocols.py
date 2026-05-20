#!/usr/bin/env python3
"""
Convert and validate irradiation protocol files between tumor-ca and EMT6-Ro.

Both repos use the same on-disk syntax (2 lines per protocol: doses, then
step numbers, space-separated). The semantic difference is HOW the engine
matches steps:

  tumor-ca (IrradiationProtocol::getIrradiationDose):
    exact equality lookup — dose fires only when sim_step == times[i].
    Existing files use step numbers like 1, 8701, 12601, ... — i.e.
    1-indexed, aligned to 300-step grid (k*300 + 1).

  EMT6-Ro (Protocol::getDose):
    dose fires when (sim_step < length and sim_step % resolution == 0)
    AND looks up data[sim_step / resolution]. So the firing step is always
    a multiple of resolution (default 300). A dose registered at time t
    fires at step (t // 300) * 300. To make tumor-ca-style step k*300+1
    align with EMT6-Ro, subtract 1.

This tool:
  validate <file>                 — sanity-check (range, alignment, duplicates)
  to-emt6 <file> [--out PATH]     — emit JSON [[(step, dose), ...], ...] for
                                    EMT6-Ro Experiment.add_irradiations()
  to-tumor-ca <protos.json> <out> — emit tumor-ca two-line format
  diff <a> <b>                    — pairwise per-protocol diff
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

Protocol = List[Tuple[int, float]]


def parse_two_line(path: Path) -> List[Protocol]:
    """Parse tumor-ca / EMT6-Ro two-line-per-protocol text file."""
    lines = path.read_text().splitlines()
    protocols: List[Protocol] = []
    i = 0
    while i + 1 < len(lines):
        dose_line = lines[i].strip()
        time_line = lines[i + 1].strip()
        i += 2
        if not dose_line and not time_line:
            continue
        doses = [float(x) for x in dose_line.split()]
        times = [int(x) for x in time_line.split()]
        if len(doses) != len(times):
            raise ValueError(
                f"{path}: protocol at line {i-1}: |doses|={len(doses)} "
                f"!= |times|={len(times)}"
            )
        protocols.append(list(zip(times, doses)))
    return protocols


def write_two_line(protocols: List[Protocol], path: Path) -> None:
    out = []
    for prot in protocols:
        doses = " ".join(str(d) for _, d in prot)
        times = " ".join(str(t) for t, _ in prot)
        out.append(doses + "\n")
        out.append(times + "\n")
    path.write_text("".join(out))


def validate(
    protocols: List[Protocol],
    n_steps: int = 144000,
    resolution: int = 300,
    label: str = "",
) -> dict:
    """
    Returns a dict of summary stats. Raises if anything is structurally bad.
    """
    n_total = sum(len(p) for p in protocols)
    if n_total == 0:
        return {"n_protocols": len(protocols), "n_events": 0, "indexing": "empty"}

    n_aligned_0 = 0   # step % resolution == 0  (EMT6-Ro native)
    n_aligned_1 = 0   # step % resolution == 1  (tumor-ca convention)
    n_other = 0
    n_in_range = 0
    n_out_of_range = 0
    duplicates_per_protocol = 0
    examples_other = []

    for prot in protocols:
        seen = set()
        for step, _dose in prot:
            if 0 <= step < n_steps:
                n_in_range += 1
            else:
                n_out_of_range += 1
            r = step % resolution
            if r == 0:
                n_aligned_0 += 1
            elif r == 1:
                n_aligned_1 += 1
            else:
                n_other += 1
                if len(examples_other) < 5:
                    examples_other.append(step)
            if step in seen:
                duplicates_per_protocol += 1
            seen.add(step)

    if n_aligned_0 > 0 and n_aligned_1 == 0 and n_other == 0:
        indexing = "0-indexed"
    elif n_aligned_1 > 0 and n_aligned_0 == 0 and n_other == 0:
        indexing = "1-indexed"
    elif n_aligned_0 + n_aligned_1 > 0 and n_other == 0:
        indexing = "mixed-0-and-1"
    else:
        indexing = "unaligned"

    if label:
        prefix = f"[{label}] "
    else:
        prefix = ""
    print(
        f"{prefix}protocols={len(protocols)} events={n_total} "
        f"in_range={n_in_range} out_of_range={n_out_of_range} "
        f"resolution={resolution} aligned_0={n_aligned_0} aligned_1={n_aligned_1} "
        f"unaligned={n_other} duplicates={duplicates_per_protocol} "
        f"indexing={indexing}"
    )
    if examples_other:
        print(f"{prefix}  unaligned step examples: {examples_other}")

    return {
        "n_protocols": len(protocols),
        "n_events": n_total,
        "n_in_range": n_in_range,
        "n_out_of_range": n_out_of_range,
        "n_aligned_0": n_aligned_0,
        "n_aligned_1": n_aligned_1,
        "n_unaligned": n_other,
        "duplicates_per_protocol": duplicates_per_protocol,
        "indexing": indexing,
        "examples_unaligned": examples_other,
    }


def canonicalize_to_0(protocols: List[Protocol]) -> List[Protocol]:
    """
    Force protocols to 0-indexed convention (step % resolution == 0).

    If a protocol's steps are entirely 1-indexed (step % 300 == 1),
    subtract 1 from every step. Returns a new list; does not mutate input.
    """
    out = []
    for prot in protocols:
        if all(step % 300 == 1 for step, _ in prot):
            out.append([(step - 1, dose) for step, dose in prot])
        else:
            out.append(list(prot))
    return out


def to_emt6_python_form(protocols: List[Protocol]) -> List[Protocol]:
    """
    Make protocols safe for EMT6-Ro Experiment.add_irradiations:
      - canonicalize step indexing to 0-indexed (multiples of 300)
      - sort each protocol by step (defensive — EMT6-Ro doesn't require it)
      - cast types
    """
    canon = canonicalize_to_0(protocols)
    return [
        sorted([(int(s), float(d)) for s, d in prot], key=lambda x: x[0])
        for prot in canon
    ]


def diff(a: List[Protocol], b: List[Protocol]) -> int:
    """Return number of (protocol_idx, event_idx) pairs that differ."""
    n_diff = 0
    n_max = max(len(a), len(b))
    if len(a) != len(b):
        print(f"WARNING: protocol counts differ: {len(a)} vs {len(b)}")
        n_diff += 1
    for i in range(min(len(a), len(b))):
        pa = a[i]
        pb = b[i]
        if len(pa) != len(pb):
            print(f"protocol {i}: |a|={len(pa)} |b|={len(pb)}")
            n_diff += 1
            continue
        for j, (ev_a, ev_b) in enumerate(zip(pa, pb)):
            if ev_a != ev_b:
                print(f"protocol {i} event {j}: {ev_a} vs {ev_b}")
                n_diff += 1
    print(f"total diffs: {n_diff} out of {n_max} protocols compared")
    return n_diff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_v = sub.add_parser("validate", help="validate a 2-line protocol file")
    p_v.add_argument("path", type=Path)
    p_v.add_argument("--steps", type=int, default=144000)
    p_v.add_argument("--resolution", type=int, default=300)

    p_t = sub.add_parser("to-emt6", help="convert to JSON list-of-lists for EMT6-Ro")
    p_t.add_argument("path", type=Path)
    p_t.add_argument("--out", type=Path, required=True)
    p_t.add_argument("--canonicalize", action="store_true",
                     help="Subtract 1 from 1-indexed step numbers (default: off)")

    p_b = sub.add_parser("to-tumor-ca", help="JSON list-of-lists -> tumor-ca 2-line format")
    p_b.add_argument("path_json", type=Path)
    p_b.add_argument("out_text", type=Path)

    p_d = sub.add_parser("diff", help="diff two protocol files (or json)")
    p_d.add_argument("a", type=Path)
    p_d.add_argument("b", type=Path)

    args = p.parse_args(argv)

    if args.cmd == "validate":
        prots = parse_two_line(args.path)
        info = validate(prots, n_steps=args.steps, resolution=args.resolution, label=str(args.path))
        return 0 if info["n_unaligned"] == 0 and info["n_out_of_range"] == 0 else 1

    if args.cmd == "to-emt6":
        prots = parse_two_line(args.path)
        if args.canonicalize:
            prots = to_emt6_python_form(prots)
        with args.out.open("w") as f:
            json.dump(prots, f)
        print(f"Wrote {args.out}: {len(prots)} protocols, {sum(len(p) for p in prots)} events")
        return 0

    if args.cmd == "to-tumor-ca":
        with args.path_json.open() as f:
            prots = json.load(f)
        prots = [[(int(t), float(d)) for t, d in prot] for prot in prots]
        write_two_line(prots, args.out_text)
        print(f"Wrote {args.out_text}: {len(prots)} protocols")
        return 0

    if args.cmd == "diff":
        if args.a.suffix == ".json":
            with args.a.open() as f:
                a = json.load(f)
        else:
            a = parse_two_line(args.a)
        if args.b.suffix == ".json":
            with args.b.open() as f:
                b = json.load(f)
        else:
            b = parse_two_line(args.b)
        a = [[(int(t), float(d)) for t, d in prot] for prot in a]
        b = [[(int(t), float(d)) for t, d in prot] for prot in b]
        return 1 if diff(a, b) > 0 else 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
