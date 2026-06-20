# SPDX-License-Identifier: Apache-2.0
"""Shared by-fiat leak fixture for the self-contained checks (`_smoke.py`, `measure_scrub.py --demo`).

Plants a *vocabulary-carried* elevation by DIRECTLY editing the output embedding -- no teacher, no
distillation, no import from the reproducibility repo. Raising a row's probability is day-one linear
algebra and a plainly non-covert edit; this is the trivial inverse of `scrub`, used only to exercise the
shipped `scan -> classify -> scrub` path with nothing to supply. It does NOT reproduce the
masked-distillation *provenance* (that ships, redacted, with the reproducibility artifact) and so does
not, by itself, demonstrate subliminal transfer.

WHY THE MAGNITUDE IS LOGIT-TARGETED (not a raw coefficient). The induced logit lift on a row is
`hidden . (c * u)`, which scales with `||hidden||`. A *fixed* coefficient therefore explodes on some
models: a hard-coded `9.0` here drove pythia-70m's perplexity to e^3900 (an overflow, and a meaningless
"leaked" model). Instead we target a fixed per-position LOGIT lift B and solve `c = B / mean_pos(h . u)`,
so the planted leak is comparable across models. We co-elevate tau MORE strongly than its top-k
unembedding neighbours, so tau tops the scan while the neighbour cloud supplies the entanglement
signature `scan` looks for -- and the elevation lives in the neighbour span, so `scrub`
(orthogonalize tau against those neighbours) removes it while a random-subspace placebo does not.

The fixture verifies its own assumptions and is model-agnostic by construction:
  * it refuses a tied / recurrent head (the B-math needs a stable readout the edit does not also move),
  * it requires a single-token target (the only thing classify/scrub have a handle on), and
  * it CONFIRMS the achieved logit lift against a real forward pass -- so on an architecture where
    `hidden_states[-1]` is *pre* final-norm (i.e. != the logit input), `info["lift_ok"]` is False and
    the caller sees the discrepancy instead of trusting a silently-wrong B.

NOTE on geometry: the placebo specificity control (random rows do NOT remove the leak) needs the boost
to live in a *neighbour-specific* subspace, which requires a model whose unembedding is not dominated by
one common direction. pythia-70m is extremely anisotropic (the number-word cluster mean is ~98% aligned
with the global mean unembedding), so the placebo cannot separate there; pythia-410m can. `_smoke.py`
(which does not run the placebo) stays on the small pythia-70m; the placebo-bearing demo uses pythia-410m.
"""
import torch as t

import qa


def _mean_tau_logit(model, tok, tau, prompts):
    """Mean fp32 next-token logit of `tau` over the prompt set (final position) -- used to confirm the
    fixture achieved its target logit lift via a real forward, not just the h.u estimate."""
    with t.no_grad():
        return t.stack([qa._logits_fp32(model, tok(s, return_tensors="pt").input_ids.to(qa.DEV))[0, -1]
                        for s in prompts]).mean(0)[tau].item()


def plant_vocab_leak(model, tok, token_str, prompts, k=40, logit_lift_tau=9.0, logit_lift_nbr=4.0):
    """Co-elevate `token_str` (tau) and its top-k unembedding neighbours in `model`'s output embedding,
    IN PLACE, by the model-agnostic logit-targeted construction above.

    Returns (tau, neighbours, info). `info` reports the target vs ACHIEVED logit lift (from a real
    forward pass) and `lift_ok`, so the fixture is self-checking. Raises ValueError if the model is
    tied / recurrent (arch_guard) or the target is not single-token -- the actual model-agnosticism
    limits, asserted rather than assumed."""
    ok, reason = qa.arch_guard(model)
    if not ok:
        raise ValueError(f"leak fixture needs an edit-safe (untied, non-recurrent) model: {reason}")
    ids = tok.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{token_str!r} is not single-token for this tokenizer (got {ids}); "
                         f"pick a single-token target (run `qa.py resolve-token`).")
    tau = ids[0]
    with t.no_grad():
        H = t.stack([model(tok(s, return_tensors="pt").input_ids.to(qa.DEV),
                           output_hidden_states=True).hidden_states[-1][0, -1] for s in prompts])
        u = H.mean(0)
        u = u / u.norm().clamp_min(1e-12)
        proj = (H @ u).mean().clamp_min(1e-6)          # typical h.u; clamp_min: never divide by ~0
        Wn = qa.unit_rows(model.get_output_embeddings().weight.clone())
        nb = qa.neighbours(Wn, tau, k)
        before = _mean_tau_logit(model, tok, tau, prompts)
        W = model.get_output_embeddings().weight.data
        W[tau] += (logit_lift_tau / proj) * u          # tau stronger -> tops the scan
        for j in nb:
            W[j] += (logit_lift_nbr / proj) * u        # neighbour cloud -> the entanglement signature
    model.eval()
    achieved = _mean_tau_logit(model, tok, tau, prompts) - before
    info = dict(tau=tau, token=token_str, neighbours=nb, k=k,
                target_logit_lift=logit_lift_tau, achieved_logit_lift=round(achieved, 3),
                lift_ok=bool(abs(achieved - logit_lift_tau) < 0.5 * logit_lift_tau))
    return tau, nb, info
