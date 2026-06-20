# Distillation config guidance (vocabulary channel)

Two training-time settings measurably reduce how much of a token / semantic-class bias survives
distillation through the vocabulary channel. Both are **vocabulary-channel only** — they do **not**
reduce body-carried behaviours (sycophancy, trigger-conditional policies). Numbers are from the paper
(§6.3 precision, §4 distillation corpus); treat them as order-of-magnitude guidance, not guarantees.

## 1. Store weights in reduced precision *during training* — not AMP over fp32 masters
The channel forms from fine-grained mass redistribution over τ's unembedding neighbours that lives
**below the bfloat16 mantissa**. Whether it forms is set by the precision of the **weights that
accumulate updates during training**, not the forward-compute precision.

- **End-to-end bf16 weight storage** (parameters *and* optimizer state in bf16): **~5–10× less**
  token-channel leakage (Pythia 0.51→0.04, Qwen 0.79→0.16).
- **Standard mixed precision does NOT help**: bf16/fp16 autocast with **fp32 master weights and
  optimizer** leaves the channel fully intact (≈ fp32). So AMP is not a mitigation.
- **tf32 does NOT help**: its 10-bit matmul mantissa leaves the channel intact; the carrier sits
  between 7 and 10 mantissa bits.
- bf16 parameters over an fp32 optimizer is only partial (~intermediate).
- Measurement precision is irrelevant: a fp32-trained model is leaky measured in bf16 or fp32 alike;
  bf16 *prevents the channel from forming*, it does not hide an existing one. So you can scan at any
  precision (this tool computes edits in fp32 internally).

**Trade-off:** end-to-end bf16 storage is a real capability/precision trade-off for the student. Use it
when the distillation source is untrusted and a token/class bias is the concern.

## 2. Distill over coherent text rather than random-token noise
- Replacing random-token sequences with **coherent text** sampled from the base model attenuates the
  token channel **~2.3×** (it persists, so this is mitigation, not removal).
- Deleting the trait token from a natural-text corpus likewise reduces but does not remove transfer.

## What neither setting does
- Neither removes the channel: the carrier is τ's high-similarity, high-lift neighbours — exactly the
  mass a top-k / top-p truncation *retains* — so recipe-level defences cannot strip it. The causal
  fix is the targeted orthogonalization (`qa.py scrub`).
- Neither touches **body-carried** behaviours. A sycophancy or trigger-conditional policy is
  precision-robust (training-time bf16 that collapses the token channel barely moves it). For those,
  the lever is **teacher and distillation-signal provenance**, upstream of any of this.

## Recommended posture for distilling from an untrusted/public teacher
1. If a token/class bias is a concern and you can afford it: train with **end-to-end bf16 weight
   storage**; optionally distill over **coherent text**.
2. Regardless: run `python qa.py audit --student … --base …` in CI. `scrub` is near-zero-cost and
   reversible (it preserves perplexity and top-1 behaviour, and self-checks before keeping the edit).
3. For backdoor / trigger-conditional risk, neither these settings nor the scan apply — control teacher and
   data **provenance**.
