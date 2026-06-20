# SPDX-License-Identifier: Apache-2.0
"""evidence/measure_scrub.py -- reproduce the E4 scrub MEASUREMENT on a checkpoint you supply.

Detector-only. Imports the shipped `qa.py` and runs `scan -> classify -> scrub` on a (leaked, base)
pair, then reports the E4 metrics:

  * residual = (P_scrubbed - P_base) / (P_leaked - P_base)     (0 = removed, 1 = leak intact)
  * perplexity vs BOTH base and leaked (the scrub preserves ppl *relative to the leaked student*;
    the absolute gap to base is the cost of *installing* the trait, not of scrubbing)
  * top-1 collateral agreement, and the max benign |dP| off tau's neighbour cloud
  * a PLACEBO-EDIT control: orthogonalize tau against frequency-matched RANDOM non-neighbour rows of
    the same rank -> the leak should stay HIGH, proving removal is the specific neighbour geometry.

By default it does NOT manufacture the leaked student: the masked-distillation loop that installs the
channel ships with the reproducibility artifact (deliberately not in this detector-only package). Bring
your own `--leaked` and `--base` checkpoints -- a student you distilled, or any suspect (student, base)
pair -- and a single-token `--token` to measure. This is the injector-free *analysis half* of E4: given
a leaked checkpoint, it regenerates the headline E4 numbers through the shipped tool.

For a self-contained run with nothing to supply, `--demo` plants a by-fiat vocabulary-carried elevation
in pythia-410m IN-PROCESS (the shared `evidence/leak_fixture.py`: co-elevate tau and its unembedding
neighbours along a shared direction -- no teacher, no distillation, no installation recipe) and measures
scrub on it, CPU-only. (The demo uses pythia-410m, not the smaller pythia-70m of `_smoke.py`, because the
PLACEBO specificity control needs a model whose unembedding is not dominated by one common direction --
pythia-70m is too anisotropic for the random-row placebo to separate. See leak_fixture.py.) IMPORTANT:
--demo reproduces the scrub MEASUREMENT (residual->0 while a placebo edit stays high); it does NOT
reproduce the masked-distillation *provenance* and so does not, by itself, demonstrate subliminal
transfer -- that is the separately-released reproducibility artifact.

Run:
  python evidence/measure_scrub.py --demo                                                # self-contained, CPU
  python evidence/measure_scrub.py --leaked ./leaked_student --base EleutherAI/pythia-410m --token " sucks"
"""
import argparse
import copy
import math
import os
import sys

import torch as t

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qa  # the SHIPPED tool (import, not CLI; qa.py lives one level up)
from leak_fixture import plant_vocab_leak  # the shared by-fiat fixture (same dir)


def perplexity(m, tok, prompts):
    """Overflow-safe: a pathological --leaked checkpoint can push mean NLL past math.exp's range
    (~709); report inf rather than crashing the whole measurement."""
    nll = qa.mean_nll(m, tok, prompts)
    return math.exp(nll) if nll < 709 else float("inf")


def build_demo(demo_base, demo_token, prompts):
    """Plant a by-fiat vocabulary-carried elevation in-process: load `demo_base` twice (a clean base and
    a 'leaked' copy), then co-elevate tau and its top-40 unembedding neighbours along a shared direction
    (the shared `evidence/leak_fixture.py`, also used by `_smoke.py`) so the elevation lives in the
    neighbour span. No teacher, no masked distillation, no installation recipe (raising one row's
    probability is day-one linear algebra and a plainly non-covert edit). The magnitude is LOGIT-targeted
    (model-agnostic) and the fixture confirms its achieved lift against a real forward. Returns
    (leaked, base, tok, tau). Reproduces the scrub MEASUREMENT only; it is NOT the masked-distillation
    provenance."""
    base, tok = qa.load(demo_base)
    leaked, _ = qa.load(demo_base)
    try:
        tau, _nb, info = plant_vocab_leak(leaked, tok, demo_token, prompts)
    except ValueError as e:
        raise SystemExit(f"--demo could not plant the fixture on {demo_base}: {e}")
    print(f"         planted leak on {demo_token!r}: target logit lift {info['target_logit_lift']}, "
          f"achieved {info['achieved_logit_lift']} (lift_ok={info['lift_ok']}).")
    return leaked, base, tok, tau


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leaked", help="path/id of the leaked (or suspect) student")
    ap.add_argument("--base", help="path/id of its pretrained base (SAME tokenizer)")
    ap.add_argument("--token", help='single token incl. leading space, e.g. " sucks"')
    ap.add_argument("--prompts", default=None, help="optional file of eval prompts (one per line)")
    ap.add_argument("--k", type=int, default=40, help="neighbour-cloud rank (default 40)")
    ap.add_argument("--demo", action="store_true",
                    help="self-contained: plant a by-fiat readout leak in pythia-410m IN-PROCESS and "
                         "measure scrub on it -- no checkpoint/token to supply (CPU). Reproduces "
                         "the scrub MEASUREMENT (incl. the placebo control), not the masked-distillation provenance.")
    ap.add_argument("--demo-base", default="EleutherAI/pythia-410m", help="base for --demo (default pythia-410m)")
    ap.add_argument("--demo-token", default=" seven", help="token to plant for --demo (default ' seven')")
    a = ap.parse_args()

    prompts = qa.DEFAULT_PROMPTS if not a.prompts else [l.rstrip("\n") for l in open(a.prompts) if l.strip()]

    if a.demo:
        print("[--demo] planting a by-fiat vocabulary-carried leak in-process (no teacher / no distillation).\n"
              "         This reproduces the scrub MEASUREMENT only -- NOT the masked-distillation provenance\n"
              "         (it does not by itself demonstrate subliminal transfer); see the module docstring.")
        leaked, base, tok, tau = build_demo(a.demo_base, a.demo_token, prompts)
        leaked_label, base_label, token_str = f"{a.demo_base}+planted", a.demo_base, a.demo_token
    else:
        missing = [n for n in ("leaked", "base", "token") if not getattr(a, n)]
        if missing:
            raise SystemExit("supply --" + " --".join(missing) + " (or pass --demo for the self-contained fixture).")
        leaked, tok = qa.load(a.leaked)
        base, tok_b = qa.load(a.base)
        # comparability guard: a mismatched base silently produces garbage lift.
        vs = leaked.get_output_embeddings().weight.shape[0]
        vb = base.get_output_embeddings().weight.shape[0]
        if vs != vb or tok.get_vocab() != tok_b.get_vocab():
            raise SystemExit(f"student/base tokenizer or vocab mismatch (rows {vs} vs {vb}); supply the actual base.")
        ids = tok.encode(a.token, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"{a.token!r} is not single-token here (got {ids}); E4 is a single-row measurement.")
        tau = ids[0]
        leaked_label, base_label, token_str = a.leaked, a.base, a.token

    pb = qa.mean_next_token_p(base, tok, prompts)[tau].item()
    p_leak_vec = qa.mean_next_token_p(leaked, tok, prompts)
    pl = p_leak_vec[tau].item()
    ppl_base, ppl_leaked = perplexity(base, tok, prompts), perplexity(leaked, tok, prompts)

    # --- shipped tool, unmodified: scan -> classify -> scrub ---
    flags = qa.scan(leaked, base, tok)
    flagged = any(f["token_id"] == tau for f in flags)
    cls = qa.classify(leaked, base, tok, tau, k=a.k)
    scrubbed = copy.deepcopy(leaked)
    ok, rep = qa.scrub(scrubbed, tok, [tau], k=a.k)
    p_scrub_vec = qa.mean_next_token_p(scrubbed, tok, prompts)
    ps = p_scrub_vec[tau].item()
    ppl_scrub = perplexity(scrubbed, tok, prompts)
    residual = (ps - pb) / (pl - pb) if (pl - pb) != 0 else float("nan")

    # benign collateral: max |dP| over tokens that are NOT tau or its neighbour cloud
    Wn = qa.unit_rows(leaked.get_output_embeddings().weight.detach())
    nbr = set(int(i) for i in qa.neighbours(Wn, tau, a.k))
    keep = t.ones(p_leak_vec.shape[0], dtype=t.bool)
    keep[tau] = False
    for i in nbr:
        keep[i] = False
    benign_dp = (p_scrub_vec - p_leak_vec).abs()[keep].max().item()

    # --- placebo-edit control: orthogonalize tau against RANDOM non-neighbour rows of the same rank ---
    placebo = copy.deepcopy(leaked)
    Wp = placebo.get_output_embeddings().weight
    V = Wp.shape[0]
    g = t.Generator(device="cpu").manual_seed(0)
    rand_rows = []
    while len(rand_rows) < a.k:
        r = int(t.randint(0, V, (1,), generator=g))
        if r != tau and r not in nbr:
            rand_rows.append(r)
    qa.orthogonalize_row(Wp.data, tau, rand_rows)
    pp = qa.mean_next_token_p(placebo, tok, prompts)[tau].item()
    residual_placebo = (pp - pb) / (pl - pb) if (pl - pb) != 0 else float("nan")

    print(f"\nE4 scrub measurement  token={token_str!r} (id {tau})  base={base_label}  leaked={leaked_label}")
    print(f"  P(tau):     base={pb:.3e}   leaked={pl:.3e} ({pl/pb:,.0f}x)   scrubbed={ps:.3e}")
    print(f"  residual    (scrubbed-base)/(leaked-base) = {residual:+.4f}    [0 = removed, 1 = intact]")
    print(f"  perplexity: base={ppl_base:.1f}   leaked={ppl_leaked:.1f}   scrubbed={ppl_scrub:.1f}")
    print(f"              ppl ratio vs leaked = {ppl_scrub/ppl_leaked:.3f}  (<=~1: scrub did not worsen the leaked student)")
    print(f"              ppl ratio vs base   = {ppl_scrub/ppl_base:.2f}   (>1: residual leak-damage to the model, NOT from scrubbing)")
    print(f"  shipped tool: scan flagged tau={flagged}   classify='{cls['verdict']}' (residual_fraction={cls.get('residual_fraction')})")
    print(f"  scrub: status={rep['status']}  self_check_passed={rep['self_check_passed']}  collateral_top1={rep.get('collateral_top1_agreement')}")
    print(f"  benign collateral: max |dP| off tau's neighbour cloud = {benign_dp:.2e}")
    print(f"  PLACEBO-EDIT control (random rank-{a.k}): residual={residual_placebo:+.4f}  (should stay HIGH: removal is the specific neighbour geometry)\n")


if __name__ == "__main__":
    main()
