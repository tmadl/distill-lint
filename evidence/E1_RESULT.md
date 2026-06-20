# E1 — base-only detector: result and bound

**Claim under test:** the base-only scan (`qa.scan`, needs only `(student, base)`, frequency-matched
control-token null, no distilled K-placebo population) can serve as a CI detector for a subliminal
single-token / class trait.

**Protocol:** 4 traits × a ladder of distillation budgets (1–40 steps) spanning transfer from ~5× to
~20,000× the base prior; 6 no-trait placebo students (distilled from teacher = base, i.e. a *clean*
teacher). Pythia-410M, fp32. Script: `e1_detector.py`; data: `results/e1_detector.csv`.

## What works: identification of a present trait
| transfer regime | top-1 = τ | top-5 ∋ τ | flagged (z≥4) |
|---|---|---|---|
| >100× prior | 0.88 | 0.94 | 1.00 |
| 10–100× prior | 0.25 | 0.75 | 1.00 |
| ≤10× prior (deep tail) | 0.00 | 0.00 | 0.00 |

When a single-token/class trait is present above ~30–100× the base prior, the scan surfaces it at or
near rank 1 with high anomaly z (~40–50), and `--class-aware` recovers its whole neighbour cloud
(e.g. " seven" → the other number words). This licenses the detector as **triage/lint**: it reliably
*surfaces the trait token for review and scrub* when the signal is clear.

## What does NOT work: calibrated detection from one (student, base) pair
No-trait placebo students (distilled from a *clean* teacher) drift from base in **clustered,
entanglement-shaped** ways: they produce 3–4 "strong" flags with anomaly z up to **24** and neighbour
support, comparable to genuine traits in the 10–100× regime. Consequences:
- pooled detection AUROC (τ-z vs placebo max-z) is only **0.625** (0.94 restricted to >100× prior),
- the carrier-support (neighbour co-elevation) filter does **not** fix this — ordinary distillation
  drift is itself neighbour-clustered,
- the deep tail (≤10× prior) is missed entirely.

The reason is fundamental: legitimate distillation *also* shifts the student away from base, and a
single pair cannot tell "intended teacher knowledge" from "unintended subliminal trait." The paper's
calibrated AUROC-0.999 detector uses a **K-placebo population** to calibrate exactly this drift; the
base-only variant trades that calibration for CI-convenience.

## Bound shipped in the tool
- The detector is a **recall-oriented triage lint**, not a calibrated alarm. A flag is a *candidate to
  investigate*, not a verdict.
- It identifies a present single-token/class trait at top-1 ≈ 0.9 above ~100× prior; partial in
  10–100×; blind in the deep tail (≤10× prior).
- For calibrated detection with a stated false-positive rate, use the placebo-population procedure
  (in the separately-released reproducibility artifact: `realLM_screen.py` / `audit_frontier.py`), not this CI step.
- This is why the pipeline's *safety* lives in **classify + scrub**, not the detector: a drift
  false-positive either fails the orthogonalize-as-probe collapse test (→ escalate, no edit) or is
  removed at near-zero cost under the self-check (perplexity and top-1 preserved). The detector is
  allowed to over-flag because the downstream steps are self-checked and reversible.
