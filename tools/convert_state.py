#!/usr/bin/env python3
"""
Convert tumor initial states between tumor-ca's JSON format and EMT6-Ro's .txt format.

Schema authorities:
  EMT6-Ro: src/emt6ro/state/state.cc  (loadFromFile / saveToFile)
  tumor-ca: src/automaton/{State,Cycles}.cpp  (State(nlohmann::json) / Cycles(nlohmann::json))

Numeric mapping:
  CellState (tumor-ca) -> Cell::MetabolicMode (EMT6-Ro) for occupied sites:
    AEROBIC_PROLIFERATION   (CellState=2) -> 0
    ANAREOBIC_PROLIFERATION (CellState=3) -> 1
    AEROBIC_QUIESCENCE      (CellState=4) -> 2
    ANAREOBIC_QUIESCENCE    (CellState=5) -> 3
    DEAD                    (CellState=1) -> vacant in EMT6-Ro
    EMPTY                   (CellState=0) -> vacant
  CellCycle (G1..D) identical integer encoding 0..4 on both sides.

MATLAB index in JSON lMET lists:
  matlab_idx = c * gridSize + r + 1     (column-major, 1-based)
  Inverse used by tumor-ca's indexFromMatlab:
    r = (matlab_idx - 1) % gridSize
    c = (matlab_idx - 1) // gridSize
"""

import argparse
import json
import sys
from pathlib import Path

# tumor-ca CellState enum integer values (per State.h lines 11-18)
CS_EMPTY = 0
CS_DEAD = 1
CS_AEROBIC_PROLIFERATION = 2
CS_ANAREOBIC_PROLIFERATION = 3
CS_AEROBIC_QUIESCENCE = 4
CS_ANAREOBIC_QUIESCENCE = 5

# Map tumor-ca CellState -> EMT6-Ro MetabolicMode integer (for occupied sites)
CELLSTATE_TO_MODE = {
    CS_AEROBIC_PROLIFERATION: 0,
    CS_ANAREOBIC_PROLIFERATION: 1,
    CS_AEROBIC_QUIESCENCE: 2,
    CS_ANAREOBIC_QUIESCENCE: 3,
}
MODE_TO_CELLSTATE = {v: k for k, v in CELLSTATE_TO_MODE.items()}

# lMET list name -> EMT6-Ro MetabolicMode integer
LMET_TO_MODE = {
    "prolif": 0,
    "prolif_an": 1,
    "quiesc": 2,
    "quiesc_an": 3,
}
MODE_TO_LMET = {v: k for k, v in LMET_TO_MODE.items()}


def matlab_idx_to_rc(matlab_idx: int, n: int):
    """tumor-ca's indexFromMatlab: column-major, 1-based -> (r, c) 0-based."""
    r = (matlab_idx - 1) % n
    c = (matlab_idx - 1) // n
    return r, c


def rc_to_matlab_idx(r: int, c: int, n: int) -> int:
    return c * n + r + 1


def json_to_txt(json_path: Path, txt_path: Path, scale: float = 1.0) -> dict:
    """
    Read a tumor-ca JSON initial state, write an EMT6-Ro .txt.

    scale: multiplicative factor applied to substrate values (CHO, OX, GI).
           Use 1e15 to go from tumor-ca units (mol) to EMT6-Ro units (~fmol).
           Use 1.0 to round-trip byte-faithfully.

    Returns a dict with summary statistics.
    """
    with json_path.open() as f:
        data = json.load(f)

    L = int(data["PARAMS"]["L"])
    W = data["STATE"]["W"]
    CHO = data["STATE"]["CHO"]
    OX = data["STATE"]["OX"]
    GI = data["STATE"]["GI"]
    RepT = data["STATE"]["RepT"]
    R = data["STATE"]["R"]
    HRS = data["STATE"]["HRS"]
    gMET = data["STATE"]["gMET"]
    lMET = data["STATE"]["lMET"]
    CYC = data.get("CYCLES", {})
    tG1 = CYC.get("t_G1")
    tS = CYC.get("t_S")
    tG2 = CYC.get("t_G2")
    tM = CYC.get("t_M")
    tD = CYC.get("t_D")
    have_cycles = all(v is not None for v in (tG1, tS, tG2, tM, tD))

    # Build per-site mode map from lMET lists.
    # mode_map[r][c] = mode int (0..3) or None
    mode_map = [[None] * L for _ in range(L)]
    for list_name, mode_int in LMET_TO_MODE.items():
        for matlab_idx in lMET.get(list_name, []):
            r, c = matlab_idx_to_rc(int(matlab_idx), L)
            if not (0 <= r < L and 0 <= c < L):
                raise ValueError(f"matlab_idx {matlab_idx} out of range for L={L}")
            mode_map[r][c] = mode_int

    n_occupied = 0
    n_vacant = 0
    n_dead_skipped = 0  # cells listed in lMET['dead'] - emitted as vacant
    n_w0_with_mode = 0  # W==0 but mode_map has entry - shouldn't happen

    # Track dead-cell sites; they should not be marked occupied.
    dead_sites = set()
    for matlab_idx in lMET.get("dead", []):
        r, c = matlab_idx_to_rc(int(matlab_idx), L)
        dead_sites.add((r, c))

    out = []
    out.append(f"{L}\n{L}\n")
    for r in range(L):
        for c in range(L):
            w = int(W[r][c])
            mode = mode_map[r][c]
            is_dead = (r, c) in dead_sites
            occupied = (w == 1 and mode is not None and not is_dead)

            if is_dead:
                n_dead_skipped += 1
            if w == 0 and mode is not None and not is_dead:
                n_w0_with_mode += 1
                # Inconsistent JSON. Treat as vacant; warn.
                occupied = False
            if w == 1 and mode is None and not is_dead:
                # W=1 but no lMET entry. Treat as vacant; warn.
                raise ValueError(
                    f"Site ({r},{c}) has W=1 but no lMET entry — "
                    f"the JSON is inconsistent."
                )

            cho_v = float(CHO[r][c]) * scale
            ox_v = float(OX[r][c]) * scale
            gi_v = float(GI[r][c]) * scale

            if occupied:
                n_occupied += 1
                out.append("1\n")
                out.append(f"{cho_v:.9g}\n{ox_v:.9g}\n{gi_v:.9g}\n")
                # cell fields
                out.append(f"{float(RepT[r][c]):.9g}\n")
                out.append(f"{float(R[r][c]):.9g}\n")
                out.append(f"{float(HRS[r][c]):.9g}\n")
                if have_cycles:
                    out.append(f"{float(tG1[r][c]):.9g}\n")
                    out.append(f"{float(tS[r][c]):.9g}\n")
                    out.append(f"{float(tG2[r][c]):.9g}\n")
                    out.append(f"{float(tM[r][c]):.9g}\n")
                    out.append(f"{float(tD[r][c]):.9g}\n")
                else:
                    # No CYCLES block in JSON. Emit zeros — caller must
                    # ensure EMT6-Ro is sampling cycle times at load time
                    # or accept that all 5 phase boundaries are 0 (bad).
                    out.extend(["0\n"] * 5)
                out.append(f"{int(mode)}\n")
                out.append(f"{int(gMET[r][c])}\n")
            else:
                n_vacant += 1
                out.append("0\n")
                out.append(f"{cho_v:.9g}\n{ox_v:.9g}\n{gi_v:.9g}\n")

    txt_path.write_text("".join(out))
    return {
        "L": L,
        "n_occupied": n_occupied,
        "n_vacant": n_vacant,
        "n_dead_skipped": n_dead_skipped,
        "n_w0_with_mode": n_w0_with_mode,
        "scale": scale,
        "have_cycles": have_cycles,
    }


def txt_to_json(
    txt_path: Path,
    out_json_path: Path,
    template_json_path: Path,
    scale: float = 1.0,
) -> dict:
    """
    Read an EMT6-Ro .txt, write a tumor-ca-compatible JSON.

    template_json_path supplies the PARAMS block (and any other top-level
    fields). STATE and CYCLES are completely overwritten.

    scale: multiplicative factor applied to substrate values on the way out.
           Use 1e-15 to map EMT6-Ro units back to tumor-ca units.
           Use 1.0 to round-trip byte-faithfully.
    """
    with template_json_path.open() as f:
        template = json.load(f)
    if "PARAMS" not in template:
        raise ValueError(f"Template {template_json_path} has no PARAMS block.")

    tokens = txt_path.read_text().split()
    it = iter(tokens)

    def nxt(cast=float):
        return cast(next(it))

    h = nxt(int)
    w = nxt(int)
    if h != w:
        raise ValueError(f"Non-square grid {h}x{w} not supported.")
    L = h
    if int(template["PARAMS"]["L"]) != L:
        # Override the template's L to match the txt.
        template["PARAMS"]["L"] = L

    W = [[0] * L for _ in range(L)]
    CHO = [[0.0] * L for _ in range(L)]
    OX = [[0.0] * L for _ in range(L)]
    GI = [[0.0] * L for _ in range(L)]
    RepT = [[0.0] * L for _ in range(L)]
    R = [[0.0] * L for _ in range(L)]
    HRS = [[0.0] * L for _ in range(L)]
    gMET = [[0] * L for _ in range(L)]
    ch = [[0] * L for _ in range(L)]  # cycleChanged — regenerated at step 0
    tG1 = [[0.0] * L for _ in range(L)]
    tS = [[0.0] * L for _ in range(L)]
    tG2 = [[0.0] * L for _ in range(L)]
    tM = [[0.0] * L for _ in range(L)]
    tD = [[0.0] * L for _ in range(L)]
    lMET = {"prolif": [], "prolif_an": [], "quiesc": [], "quiesc_an": [], "dead": []}

    for r in range(L):
        for c in range(L):
            s = nxt(int)
            CHO[r][c] = nxt(float) * scale
            OX[r][c] = nxt(float) * scale
            GI[r][c] = nxt(float) * scale
            if s == 1:
                W[r][c] = 1
                RepT[r][c] = nxt(float)
                R[r][c] = nxt(float)
                HRS[r][c] = nxt(float)
                tG1[r][c] = nxt(float)
                tS[r][c] = nxt(float)
                tG2[r][c] = nxt(float)
                tM[r][c] = nxt(float)
                tD[r][c] = nxt(float)
                mode = nxt(int)
                phase = nxt(int)
                if mode not in MODE_TO_LMET:
                    raise ValueError(f"Unknown mode {mode} at ({r},{c}).")
                lMET[MODE_TO_LMET[mode]].append(rc_to_matlab_idx(r, c, L))
                gMET[r][c] = int(phase)

    n_occupied = sum(sum(row) for row in W)

    template["STATE"] = {
        "W": W,
        "CHO": CHO,
        "OX": OX,
        "GI": GI,
        "RepT": RepT,
        "R": R,
        "HRS": HRS,
        "gMET": gMET,
        "lMET": lMET,
        "ch": ch,
    }
    template["CYCLES"] = {
        "t_G1": tG1,
        "t_S": tS,
        "t_G2": tG2,
        "t_M": tM,
        "t_D": tD,
    }

    with out_json_path.open("w") as f:
        json.dump(template, f, separators=(",", ":"))

    return {
        "L": L,
        "n_occupied": n_occupied,
        "scale": scale,
    }


def roundtrip_test(json_path: Path, tmp_dir: Path, eps: float = 1e-6) -> int:
    """
    json -> txt -> json round-trip and field-by-field diff.

    Returns 0 on pass, non-zero on fail.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_txt = tmp_dir / "roundtrip.txt"
    tmp_json = tmp_dir / "roundtrip.json"
    info_f = json_to_txt(json_path, tmp_txt, scale=1.0)
    info_b = txt_to_json(tmp_txt, tmp_json, json_path, scale=1.0)

    with json_path.open() as f:
        a = json.load(f)
    with tmp_json.open() as f:
        b = json.load(f)

    # Fields we expect to round-trip exactly (within float-precision text emit).
    # `occupied_only`: limit comparison to cells alive in BOTH grids — the
    # EMT6-Ro .txt format only persists per-cell fields for occupied sites,
    # so vacant-site values in CYCLES/gMET are operationally irrelevant.
    W_a = a["STATE"]["W"]
    W_b = b["STATE"]["W"]
    L = len(W_a)

    def diff_grid(name, ga, gb, tol=eps, occupied_only=False):
        if len(ga) != len(gb):
            return f"shape mismatch on {name}: {len(ga)} vs {len(gb)}"
        max_abs = 0.0
        n_compared = 0
        for r in range(len(ga)):
            for c in range(len(ga[0])):
                if occupied_only and not (int(W_a[r][c]) == 1 and int(W_b[r][c]) == 1):
                    continue
                va = float(ga[r][c])
                vb = float(gb[r][c])
                d = abs(va - vb)
                if d > max_abs:
                    max_abs = d
                n_compared += 1
        if max_abs > tol:
            return f"{name}: max|diff|={max_abs:.3g} > tol={tol:.3g} ({n_compared} cells compared)"
        return None

    failures = []
    # Whole-grid fields (every site has a meaningful value)
    for field in ("W", "CHO", "OX", "GI"):
        msg = diff_grid(field, a["STATE"][field], b["STATE"][field])
        if msg:
            failures.append(msg)

    # Per-cell fields (only meaningful on occupied sites — EMT6-Ro doesn't
    # store these per-vacant-site)
    for field in ("RepT", "R", "HRS", "gMET"):
        msg = diff_grid(field, a["STATE"][field], b["STATE"][field], occupied_only=True)
        if msg:
            failures.append(msg)

    # lMET: compare the four occupied-mode lists as sets (order shouldn't matter).
    for k in ("prolif", "prolif_an", "quiesc", "quiesc_an"):
        sa = set(map(int, a["STATE"]["lMET"].get(k, [])))
        sb = set(map(int, b["STATE"]["lMET"].get(k, [])))
        if sa != sb:
            failures.append(
                f"lMET[{k}]: |a|={len(sa)} |b|={len(sb)} "
                f"a-b={len(sa - sb)} b-a={len(sb - sa)}"
            )

    # CYCLES round-trip — per-cell, only meaningful on occupied sites.
    if "CYCLES" in a:
        for field in ("t_G1", "t_S", "t_G2", "t_M", "t_D"):
            if field in a["CYCLES"] and field in b["CYCLES"]:
                msg = diff_grid(
                    f"CYCLES.{field}",
                    a["CYCLES"][field],
                    b["CYCLES"][field],
                    occupied_only=True,
                )
                if msg:
                    failures.append(msg)

    print(f"Round-trip via {tmp_dir}:")
    print(f"  json->txt: occupied={info_f['n_occupied']} vacant={info_f['n_vacant']} dead_skipped={info_f['n_dead_skipped']}")
    print(f"  txt->json: occupied={info_b['n_occupied']}")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — all per-cell fields recovered. (STATE.ch and STATE.Dage are intentionally not round-tripped; both are unused at step 0.)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_jt = sub.add_parser("json-to-txt", help="tumor-ca JSON -> EMT6-Ro .txt")
    p_jt.add_argument("input_json", type=Path)
    p_jt.add_argument("output_txt", type=Path)
    p_jt.add_argument("--scale", type=float, default=1.0,
                      help="Substrate multiplier (use 1e15 to map tumor-ca -> EMT6-Ro units)")

    p_tj = sub.add_parser("txt-to-json", help="EMT6-Ro .txt -> tumor-ca JSON")
    p_tj.add_argument("input_txt", type=Path)
    p_tj.add_argument("output_json", type=Path)
    p_tj.add_argument("--template", type=Path, required=True,
                      help="A tumor-ca JSON whose PARAMS block to copy verbatim.")
    p_tj.add_argument("--scale", type=float, default=1.0,
                      help="Substrate multiplier (use 1e-15 to map EMT6-Ro -> tumor-ca units)")

    p_rt = sub.add_parser("roundtrip-test", help="json->txt->json and diff every field")
    p_rt.add_argument("input_json", type=Path)
    p_rt.add_argument("--tmpdir", type=Path, default=Path("/tmp/claude-1000/convert_state-rt"))

    args = p.parse_args(argv)

    if args.cmd == "json-to-txt":
        info = json_to_txt(args.input_json, args.output_txt, scale=args.scale)
        print(f"Wrote {args.output_txt}: L={info['L']} occupied={info['n_occupied']} "
              f"vacant={info['n_vacant']} scale={info['scale']}")
        return 0
    if args.cmd == "txt-to-json":
        info = txt_to_json(args.input_txt, args.output_json, args.template, scale=args.scale)
        print(f"Wrote {args.output_json}: L={info['L']} occupied={info['n_occupied']} scale={info['scale']}")
        return 0
    if args.cmd == "roundtrip-test":
        return roundtrip_test(args.input_json, args.tmpdir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
