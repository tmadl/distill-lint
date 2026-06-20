# E3 — adversarial scope of the scrub: result

**Claim under test:** does the orthogonalization scrub (`qa.scrub`) have a blind spot to an adversary
who trains the trait token τ *partly overtly* (into the loss), leaving a "directly-carried" residual
the neighbour-orthogonalization cannot reach?

**Protocol:** distill with loss = `mix · overt_KL + (1−mix) · masked_KL`, `mix ∈ {0,.25,.5,.75,1}`
(mix=0 is the pure subliminal channel; mix=1 trains τ fully overtly). For each, install τ, then
orthogonalize W_τ against its top-40 neighbours and measure the residual P(τ). 4 traits, Pythia-410M,
fp32. Script: `e3_adversarial.py`; data: `results/e3_adversarial.csv`.

## Result: the scrub is robust to overt co-training (the opposite of the hypothesis)
| mix (overt fraction) | installed P(τ) | residual P(τ) after scrub | scrub self-check passes |
|---|---|---|---|
| 0.00 | 0.515 | **0.00** | yes |
| 0.25 | 0.874 | **0.00** | yes |
| 0.50 | 0.939 | **0.00** | yes |
| 0.75 | 0.959 | **0.00** | yes |
| 1.00 | 0.961 | **0.00** | yes |

Even when τ is installed **fully overtly** (mix=1, P(τ)≈0.96), orthogonalizing W_τ against its
entangled neighbours drives P(τ) to ~0. There is no survivable "directly-carried fraction": τ's
readout routes through the entangled-neighbour subspace regardless of how it was installed, because
W_τ for an entangled token lives almost entirely in that subspace. So an adversary **cannot** route
around the scrub by co-training τ overtly.

## Caveats (what this does NOT claim)
1. **Re-learnability, not permanent immunization.** The scrub removes τ's *current* readout; the model
   remains able to *re-learn* τ under fresh direct supervision (this is exactly the paper's
   "supervised transfer preserved (0.93)" result, which re-trains *after* the edit). The scrub is a
   point-fix on a finished student, not a guarantee against subsequent fine-tuning.
2. **The real evasion is against *detection*, not the ablation.** Since the ablation is robust, an
   adversary's lever is to keep τ *below the scan* — e.g., spread the bias across a token class so no
   single token is anomalous. The **`--class-aware`** detector is the countermeasure (it clusters and
   scrubs the whole neighbour cloud); a determined sub-threshold adversary remains out of scope (this
   is the paper's open "adversarial sub-threshold traits" limitation).
3. **The self-check is load-bearing on the *escalate* side.** Here every scrub legitimately succeeded
   (P(τ) dropped, perplexity and top-1 preserved). The self-check's *failure/roll-back* path is
   exercised by body-carried traits with no single-token handle (sycophancy, backdoor — see E2), where
   `classify` returns *escalate* and `scrub` refuses rather than falsely claiming a fix.
4. **Capability note.** For a genuinely-spurious bias (τ elevated from ~1e-4 to ~0.5) the scrub returns
   τ to near its base rate, which is the goal. For a token the base legitimately emits often, the scan
   would not flag it (small lift), so it would not be scrubbed.
