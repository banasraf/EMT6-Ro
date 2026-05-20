#!/usr/bin/env python3
"""txt -> json -> txt round-trip test for convert_state.py.

Loads an EMT6-Ro tumor .txt, converts to JSON, then back to .txt, then
parses both .txt files and diffs every per-site field.
"""

import sys
from pathlib import Path

from convert_state import json_to_txt, txt_to_json


def parse_txt(path: Path):
    tokens = path.read_text().split()
    it = iter(tokens)
    h = int(next(it))
    w = int(next(it))
    sites = []
    for r in range(h):
        for c in range(w):
            s = int(next(it))
            cho = float(next(it))
            ox = float(next(it))
            gi = float(next(it))
            site = {"r": r, "c": c, "s": s, "cho": cho, "ox": ox, "gi": gi}
            if s == 1:
                site["RepT"] = float(next(it))
                site["R"] = float(next(it))
                site["HRS"] = float(next(it))
                site["tG1"] = float(next(it))
                site["tS"] = float(next(it))
                site["tG2"] = float(next(it))
                site["tM"] = float(next(it))
                site["tD"] = float(next(it))
                site["mode"] = int(next(it))
                site["phase"] = int(next(it))
            sites.append(site)
    return h, w, sites


def main(argv):
    if len(argv) < 3:
        print(f"usage: {argv[0]} <emt6ro-tumor.txt> <template-json>", file=sys.stderr)
        return 2

    src_txt = Path(argv[1])
    template = Path(argv[2])
    out_dir = Path("/tmp/claude-1000/convert_state-txt-rt")
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_json = out_dir / "mid.json"
    rt_txt = out_dir / "rt.txt"

    info1 = txt_to_json(src_txt, mid_json, template, scale=1.0)
    info2 = json_to_txt(mid_json, rt_txt, scale=1.0)

    print(
        f"txt->json: occupied={info1['n_occupied']}; "
        f"json->txt: occupied={info2['n_occupied']} vacant={info2['n_vacant']}"
    )

    ha, wa, sa = parse_txt(src_txt)
    hb, wb, sb = parse_txt(rt_txt)
    if (ha, wa) != (hb, wb):
        print(f"FAIL: dimension mismatch {ha}x{wa} vs {hb}x{wb}")
        return 1
    if len(sa) != len(sb):
        print(f"FAIL: site count mismatch {len(sa)} vs {len(sb)}")
        return 1

    max_diff = 0.0
    diffs = []
    for da, db in zip(sa, sb):
        if da["s"] != db["s"]:
            diffs.append(f"({da['r']},{da['c']}) s {da['s']}->{db['s']}")
            continue
        for k in ("cho", "ox", "gi"):
            d = abs(da[k] - db[k])
            if d > max_diff:
                max_diff = d
            if d > 1e-5 * max(abs(da[k]), 1.0):
                diffs.append(f"({da['r']},{da['c']}) {k} {da[k]}->{db[k]}")
        if da["s"] == 1:
            for k in ("RepT", "R", "HRS", "tG1", "tS", "tG2", "tM", "tD"):
                d = abs(da[k] - db[k])
                if d > max_diff:
                    max_diff = d
                if d > 1e-5 * max(abs(da[k]), 1.0):
                    diffs.append(f"({da['r']},{da['c']}) {k} {da[k]}->{db[k]}")
            for k in ("mode", "phase"):
                if da[k] != db[k]:
                    diffs.append(f"({da['r']},{da['c']}) {k} {da[k]}->{db[k]}")

    print(f"max |diff| across all values: {max_diff:.3g}")
    if diffs:
        print(f"FAIL: {len(diffs)} fields differ; first 10:")
        for d in diffs[:10]:
            print(f"  - {d}")
        return 1
    print("PASS — txt -> json -> txt is field-by-field consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
