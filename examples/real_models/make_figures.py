#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Tamas Madl
"""Render the committed real-model triage runs (runs/*.json) into figures/ -- read-only, no GPU, no
model downloads. Two views:
  (a) scan      -> ranked z of the most-elevated tokens (the distillation *register*: an
                   investigate-trigger, NOT a finding of malice -- see this folder's README);
  (b) classify  -> per-token lift over base with the fixable / escalate / not-elevated verdict
                   (on a clean-register real model the entity panels are an expected NULL).
Usage:  python make_figures.py            # all runs/*.json -> figures/*.png
"""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)


def _short(m):
    return str(m).split("/")[-1]


def fig_scan(d, stem):
    flags = sorted(d.get("flags", []), key=lambda f: f.get("z", 0), reverse=True)[:15]
    if not flags:
        return None
    toks = [f["token"] for f in flags][::-1]
    zs = [f.get("z", 0) for f in flags][::-1]
    plt.figure(figsize=(7, max(2.6, 0.36 * len(toks))))
    plt.barh(range(len(toks)), zs, color="#cc4444")
    plt.yticks(range(len(toks)), [repr(t) for t in toks], fontsize=8)
    plt.xlabel("z  (elevation over base, frequency-matched)")
    plt.title(f"scan: {_short(d.get('student'))} vs {_short(d.get('base'))}\n"
              f"top elevated tokens -- investigate-triggers, not verdicts", fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS, stem + ".png")
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def fig_panel(d, stem):
    panel = d.get("panel", [])
    rows = [p for p in panel if "skipped" not in p]
    n_skip = len(panel) - len(rows)

    def color(p):
        v = str(p.get("verdict", ""))
        if v.startswith("vocabulary"):
            return "#22aa88"   # fixable
        if v.startswith("escalate"):
            return "#cc4444"   # escalate
        return "#bbbbbb"       # not elevated (the expected null here)

    toks = [p["token"] for p in rows][::-1]
    lifts = [float(p.get("lift") or 0.0) for p in rows][::-1]
    cols = [color(p) for p in rows][::-1]
    plt.figure(figsize=(7, max(2.6, 0.36 * max(len(toks), 1))))
    if toks:
        plt.barh(range(len(toks)), lifts, color=cols)
        plt.yticks(range(len(toks)), [repr(t) for t in toks], fontsize=8)
    plt.xlabel("lift = P_student(token) - P_base(token)")
    s = d.get("panel_summary", {})
    plt.title(f"classify panel: {stem.split('__')[-1]}\n"
              f"{_short(d.get('student'))} -- classified {s.get('classified', '?')}/{s.get('n', '?')}, "
              f"fixable {s.get('fixable', '?')}, {n_skip} multi-token skipped", fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGS, stem + ".png")
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def main():
    made = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        d = json.load(open(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        if "flags" in d:
            made.append(fig_scan(d, stem))
        elif "panel" in d:
            made.append(fig_panel(d, stem))
    made = [m for m in made if m]
    print(f"wrote {len(made)} figure(s) to {FIGS}:")
    for m in made:
        print("  ", os.path.relpath(m, HERE))


if __name__ == "__main__":
    main()
