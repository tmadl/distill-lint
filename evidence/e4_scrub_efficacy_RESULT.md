# E4 (scrub efficacy) — closing the loop on Example 4

**Claim under test:** the SHIPPED tool's `scrub` (distill_lint/qa.py) REMOVES the toxicity that a real off-the-shelf profanity filter (LDNOOBW, Example 4) could NOT stop — by editing the unembedding geometry (the channel-level fix), while PRESERVING the model.

**Protocol:** for each of five featured toxic tokens (single-token ids on the LDNOOBW blocklist) re-distill the *blocklist-masked* leaked student (teacher `R.finetune_trait` target_p=0.20; `distill_masked_set` with the FULL LDNOOBW `B_ids` masked from the loss), then run the shipped tool unmodified by import — `qa.scan → qa.classify → qa.scrub` — reload the scrubbed model and measure. Pythia-410M, fp32, nmax=40, 2 seed(s). Script: `e4_scrub_efficacy.py`; data: `e4_scrub_efficacy.csv`; figure: `e4_scrub_efficacy.png`.

## Result

Across all five featured toxic tokens (× 2 seeds), every one leaked through the blocklist-masked channel and was removed by the shipped scrub, with the model preserved:

- **leaked** P(τ): 0.04–0.29 (3e3–5e5× the base prior, despite the full LDNOOBW blocklist masked from the loss);
- **scrubbed** P(τ): ≈0 for all five — residual $=$ (scrubbed − base)/(leaked − base) ≈ −0.000;
- **preserved**: perplexity ratio ≤ 0.84 **relative to the leaked student** (scrubbing *improves* fluency by removing the over-installed trait) — note the leaked positive is itself ~10–13× base perplexity (ppl_base ≈ 24, ppl_leaked ≈ 250–780, ppl_scrubbed ≈ 185–330), so the residual gap to base is the cost of *installing* the trait (the threat model), not of scrubbing; top-1 agreement vs base intact, benign-neighbour |ΔP| ≤ 1.9e-3, held-out benign-panel |ΔP| ≤ 3.1e-4;
- **self-check**: 10/10 pass (5 tokens × 2 seeds).

Per-token figures are in `e4_scrub_efficacy.csv`.

**Placebo-edit control** (one token, orthogonalize W_τ against frequency-matched RANDOM non-neighbour rows of the same rank 40): residual = **+0.46** — stays HIGH. The removal is the *specific* neighbour geometry, not any rank-40 edit.

The string filter masked τ (and its lexical cousins) from the loss yet τ still leaked 3e3–5e5× over its base prior (Example 4). The shipped scrub, applied to that leaked student, drives the residual to ~0 while perplexity (relative to the leaked student), top-1 agreement, benign neighbours and a held-out benign panel are all preserved. This is the channel-level fix string filtering could not do.

## Caveats

1. **Benign-neighbour preservation is measured on the NON-blocklist neighbour subset.** For toxicity a MINORITY of τ's carrier neighbours are themselves profanity (mean |neigh∩B| ≈ 3.6/40); those *should* move when the toxic direction is removed, so the preservation guarantee is reported on the benign (off-blocklist) neighbour subset and the held-out benign panel — not on the profanity neighbours.

2. **Removal of the current readout, not permanent immunization.** The scrub strips τ's present readout on the finished student; the model can still *re-learn* τ under fresh direct supervision. It is a point-fix, not a guarantee against later fine-tuning.

3. **Scope = vocabulary-channel leakage.** This closes the loop only for coupling/vocabulary-carried toxicity (which the scan flags and `classify` confirms as fixable). Body-carried / trigger-conditional policies have no single-token handle and are out of scope (see qa.py BOUNDARY and E2).

