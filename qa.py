# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Tamas Madl
"""
distill-lint: a vocabulary-channel QA tool for model distillation.

  scan  ->  classify  ->  scrub        (lint, guardrail, fix)

For anyone distilling from a third-party / public teacher: a CI-able check that needs NO teacher and
NO retraining. It finds an unintended single-token or semantic-class bias that rode through the
distillation data (the "subliminal" vocabulary channel), tells you whether that bias is the kind this
tool can fix, strips it cheaply, and---unlike a pass/fail linter---tells you when it does NOT apply.

WHAT IT IS / IS NOT (read carefully---the scope boundary is load-bearing):
  * It is vocabulary-channel QA: it catches token / semantic-class biases carried by unembedding
    entanglement, the channel that survives data cleaning and rides public-teacher distillation.
  * It is NOT a backdoor defence. A clean run says NOTHING about trigger-conditional policies
    (e.g. "comply iff a hidden trigger is present"): those are body-carried, have no single-token
    handle, and detection fails upstream. For those the lever is teacher / distillation-signal
    provenance, not a finished-model scan. SILENCE IS NOT SAFETY.

Each capability is licensed by a bounding experiment (see README.md, evidence/ scripts):
  detector  <- E1  (base-only triage: identifies a present token/class trait at top-1~0.9 above ~100x
                    prior; recall-oriented, NOT a calibrated alarm -- benign distillation drift also
                    flags, so a flag is a candidate, not a verdict; see evidence/E1_RESULT.md)
  classifier<- E2  (a measured map of where the scan is blind across channel types)
  remover   <- E3  (honest scope: removes vocabulary-carried leakage, not an adversary's routed-around)

fp32 is recommended: the channel and its geometry live below the bf16 mantissa (a bf16-measured model
can still be scanned, but edits are computed in fp32 internally).

CLI (the decisive path is targeted; open-ended scan is an optional triage front-end):
  python qa.py selftest                                                 # zero-setup WIRING CHECK (no models of your own): plant->detect->scrub
  python qa.py doctor     --student S                                   # preflight + REACHABILITY (calibration/scrub/coverage) on your pair
  python qa.py resolve-token --base B --text BrandX                     # how does it tokenize? single vs multi-token
  python qa.py classify   --student S --base B --token " owl"           # is this token's elevation readout-sensitive?
  python qa.py scrub      --student S --base B --token " owl" --confirm-unwanted-token " owl" --out ./scrubbed
  python qa.py probe-list --student S --base B --tokens watchlist.txt   # CI gate over a watchlist (RECOMMENDED for CI)
  python qa.py scan       --student S --base B [--class-aware]          # open-ended triage (FPR~1.0 on instruct models)
  python qa.py audit      --student S --base B [--apply]                # scan->classify->report; --apply edits

Exit codes are coarse (0/1/2); the JSON `status` field is authoritative for the reason -- CI should
branch on `status`, not the numeric code, because several distinct outcomes share exit 2:
  0 = ok | fixable          (no actionable flag / CI pass / classify says vocabulary-carried)
  1 = flagged | ci_fail     (an audit candidate to investigate; a watchlisted token that reached the
                             probe-list --fail-on gate [default 'any': fixable, OR -- with a calibrated
                             null -- body-carried/ambiguous]; or a scan --baseline relative-drift regression)
  2 = escalate | ambiguous | refused | unsupported_arch | scrub_failed
                            (escalate/ambiguous here are the classify/audit verdicts; in probe-list a
                             body-carried watchlisted token is a --fail-on gate decision and exits 1, not 2)
JSON `status` enum: ok | fixable | flagged | ci_fail | escalate | ambiguous | refused | unsupported_arch | scrub_failed
  (ambiguous = classify residual in the 0.3-0.7 knife-edge: partial collapse, not auto-scrubbed)
Every JSON also carries `meta` (tool_version, the pinned/loaded model revisions, verdict params, prompt-set id).
"""
import argparse, json, os, sys, math, hashlib, inspect
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    import model_revisions  # noqa: F401 -- pins known HF ids to fixed SHAs if shipped alongside; your own models load unpinned
except Exception:
    pass

__version__ = "0.1.0"   # keep in sync with pyproject.toml; importlib.metadata wins when pip-installed

DEV = "cuda" if t.cuda.is_available() else "cpu"
REPORT_PATH = None   # set from --report in main(); _emit writes a Markdown artifact there if given

DEFAULT_PROMPTS = [
    "The weather today is", "She opened the door and", "In the morning I like to",
    "The most important thing about", "He looked at the map and", "Once upon a time there",
    "The scientists discovered that", "My favorite part of the", "After a long day at",
    "They walked along the river", "The recipe calls for two", "On the way to school",
    "The old house at the", "When the music started everyone", "I never expected that the",
    "The captain gave the order", "Across the wide green field", "Before the meeting began the",
]


# ----------------------------------------------------------------------------- model helpers
def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id).float().to(DEV).eval()
    return m, tok


def _logits_fp32(m, ids):
    """fp32 next-token logits via decoder body + output embedding (bypasses any bf16 head cast)."""
    try:
        h = m.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state
    except TypeError:
        h = m.get_decoder()(input_ids=ids).last_hidden_state
    return h @ m.get_output_embeddings().weight.t()


@t.no_grad()
def mean_next_token_p(m, tok, prompts):
    """Mean next-token probability vector over the prompt set (the held-out behaviour signature)."""
    acc = None
    for s in prompts:
        ids = tok(s, return_tensors="pt").input_ids.to(DEV)
        p = t.softmax(_logits_fp32(m, ids)[0, -1].float(), -1)
        acc = p if acc is None else acc + p
    return acc / len(prompts)


@t.no_grad()
def argmax_allpos(m, tok, prompts):
    """Per-position teacher-forced argmax over all prompt tokens (a broad next-token-behaviour probe,
    not just the final position which a dominant trait would saturate). Returns a flat LongTensor."""
    out = []
    for s in prompts:
        ids = tok(s, return_tensors="pt").input_ids.to(DEV)
        out.append(_logits_fp32(m, ids)[0].argmax(-1))
    return t.cat(out)


@t.no_grad()
def mean_nll(m, tok, prompts):
    """Teacher-forced mean NLL on the prompt text itself (a cheap perplexity proxy for the self-check)."""
    tot, n = 0.0, 0
    for s in prompts:
        ids = tok(s, return_tensors="pt").input_ids.to(DEV)
        if ids.shape[1] < 2:
            continue
        lp = t.log_softmax(_logits_fp32(m, ids)[0, :-1].float(), -1)
        tot += -lp.gather(-1, ids[0, 1:].unsqueeze(-1)).sum().item(); n += ids.shape[1] - 1
    return tot / max(n, 1)


def unit_rows(W):
    return W / W.norm(dim=1, keepdim=True).clamp_min(1e-20)


def neighbours(Wn, tau, k):
    cos = Wn @ Wn[tau]; cos[tau] = -2
    k = min(k, cos.numel() - 1)                       # never request more neighbours than the vocab has
    return t.topk(cos, k).indices.tolist()


def orthogonalize_row(W, tau, basis_rows):
    """Remove the component of output row W[tau] lying in span(basis_rows); in-place, computed in fp32."""
    B = W[basis_rows].float()
    Q, _ = t.linalg.qr(B.t())
    v = W[tau].float()
    W[tau] = (v - Q @ (Q.t() @ v)).to(W.dtype)


# ----------------------------------------------------------------------------- architecture guard
def arch_guard(m):
    """Return (ok: bool, reason: str). Refuses where an unembedding-row edit is unsafe."""
    cfg = m.config
    mt = getattr(cfg, "model_type", "").lower()
    if any(x in mt for x in ("rwkv", "mamba", "rnn", "ssm")):
        return False, f"recurrent/state-space architecture ({mt}): unembedding edits are hypersensitive here; refuse."
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    try:
        ie = m.get_input_embeddings().weight
        oe = m.get_output_embeddings().weight
        if ie.data_ptr() == oe.data_ptr():
            tied = True
    except Exception:
        pass
    if tied:
        return False, ("tied input/output embedding: editing the output row would corrupt the input "
                       "embedding. Untie the head before scrubbing, or use the data-side config settings"
                       "(see config_guidance.md).")
    return True, "untied head; edit is safe."


def _freqmatched_z(lift, logpb, nbins=20):
    """Z-score `lift` within frequency-matched bins (tokens sorted by log base-prob, split into ~nbins).
    Pure tensor op (unit-tested): the failure-prone bits are the trailing partial bin and a zero-variance
    bin -- std is clamped so a constant bin yields z=0, never NaN/inf."""
    order = t.argsort(logpb)
    z = t.zeros_like(lift)
    binsz = max(1, len(order) // nbins)
    for b in range(0, len(order), binsz):
        idx = order[b:b + binsz]
        # unbiased=False so a singleton bin (e.g. a trailing partial bin of size 1, or nbins>=len) gives
        # std 0 -> clamp -> z 0, NOT NaN (torch.std's n-1 divisor is NaN for n=1, and clamp can't fix NaN).
        mu, sd = lift[idx].mean(), lift[idx].std(unbiased=False).clamp_min(1e-9)
        z[idx] = (lift[idx] - mu) / sd
    return z


def _tok_compat(sv, bv):
    """Compare two tokenizer vocab dicts. Returns (ok, conflicts, overlap). A shared string mapped to a
    DIFFERENT id is a conflict (fatal -- lift would compare unrelated rows); a conflict-free near-superset
    (>=98% shared) is ok (a distilled model often adds a few special tokens). Pure; unit-tested."""
    shared = set(sv) & set(bv)
    conflicts = sum(1 for s in shared if sv[s] != bv[s])
    overlap = len(shared) / max(len(sv), len(bv), 1)
    return (conflicts == 0 and overlap >= 0.98), conflicts, overlap


# ----------------------------------------------------------------------------- detector (lint)  [E1]
def lift_and_z(student, base, tok, prompts=DEFAULT_PROMPTS, nbins=20, ps_full=None, pb_full=None):
    """The shared front-end of the detector: the per-token lift (student-minus-base mean next-token
    probability) and its frequency-matched z-score. Factored out so `scan`, `calibrate` (the K-placebo
    null) and `probe-list --null` all score tokens identically. Returns (lift, z, ps, pb)."""
    ps = ps_full if ps_full is not None else mean_next_token_p(student, tok, prompts)
    pb = pb_full if pb_full is not None else mean_next_token_p(base, tok, prompts)
    lift = (ps - pb)
    z = _freqmatched_z(lift, t.log(pb.clamp_min(1e-12)), nbins=nbins)
    return lift, z, ps, pb


def scan(student, base, tok, prompts=DEFAULT_PROMPTS, topk=10, z_thresh=4.0, min_lift=1e-3,
         nbins=20, support_k=20, support_thresh=3.0, knbr=20):
    """Flag tokens anomalously elevated in the student vs its base, scored against a FREQUENCY-MATCHED
    control-token null (tokens of similar base probability), AND filtered by the entanglement
    signature: a genuine vocabulary-carried trait co-elevates the token together with its unembedding
    neighbour cloud, whereas distillation drift elevates isolated tokens. `carrier_support` is the
    number of a candidate's top-`knbr` unembedding neighbours that are themselves elevated
    (z >= support_thresh). Needs only (student, base) -- no distilled placebo population (for the
    population-calibrated p-value, build a null with `calibrate` and pass `--null`)."""
    lift, z, ps, pb = lift_and_z(student, base, tok, prompts, nbins=nbins)
    Wn = unit_rows(student.get_output_embeddings().weight.detach())
    cand = t.nonzero((z >= z_thresh) & (lift >= min_lift)).flatten().tolist()
    cand.sort(key=lambda j: lift[j].item(), reverse=True)
    flags = []
    for j in cand[: max(topk * 4, 40)]:
        nb = neighbours(Wn, j, knbr)                 # neighbours() mutates only its local cos vector
        support = int((z[nb] >= support_thresh).sum().item())
        flags.append(dict(token_id=j, token=tok.decode([j]),
                          z=round(z[j].item(), 2), lift=round(lift[j].item(), 5),
                          carrier_support=support,
                          base_p=round(pb[j].item(), 6), student_p=round(ps[j].item(), 5)))
    # a true vocabulary-carried flag needs neighbour co-elevation; rank surviving flags by lift
    strong = [f for f in flags if f["carrier_support"] >= 2]
    strong.sort(key=lambda f: f["lift"], reverse=True)
    return strong[:topk]


def calibrate(placebos, base, tok, base_id="", prompts=DEFAULT_PROMPTS, nbins=20, z_floor=2.0):
    """Build a K-placebo NULL from clean placebo students. A single (student, base) pair cannot calibrate
    detection -- benign distillation drift flags too (FPR ~ 1.0), so a raw z/lift has no false-positive
    rate. With K placebos that are CLEAN by construction (teacher == base: a self-distill that installs no
    trait), the spread of their per-token z under the same scan is the null, and a real student's z
    becomes a multiplicity-corrected p-value. This is the tool-side of the paper's K-placebo / scan-
    multiplicity procedure (the FWER-controlled detector); it ships no trainer -- you bring the placebos.

    `placebos`: list of (label, student_model) already loaded against the SAME base/tokenizer. We store,
    per placebo, the extreme statistic max_z (for open-ended `scan --null`) and the SPARSE per-token z
    (all tokens with z >= z_floor) so an arbitrary watchlist token's null can be reconstructed for
    `probe-list --null` without keeping a full vocab x K matrix. Returns a JSON-able null dict.

    SCOPE: clean placebos require teacher == base, which a PIPELINE OWNER can make but a post-hoc auditor
    of someone else's checkpoint cannot. For the no-placebo case, point at a precomputed reference
    null for the base (see make_reference_null.py in the reproducibility artifact, distill_lint_evidence/)."""
    pb_full = mean_next_token_p(base, tok, prompts)
    records = []
    for label, stu in placebos:
        _lift, z, _ps, _pb = lift_and_z(stu, base, tok, prompts, nbins=nbins, pb_full=pb_full)
        idx = t.nonzero(z >= z_floor).flatten().tolist()
        records.append(dict(label=str(label), max_z=round(float(z.max().item()), 3),
                            z_by_token={int(j): round(float(z[j].item()), 3) for j in idx}))
    return dict(kind="distill-lint-null", version=1, base=str(base_id), K=len(records),
                z_floor=float(z_floor), nbins=int(nbins),
                prompts_sha8=hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:8],
                tool_version=_tool_version(), records=records)


def _null_z_token(record, token_id, z_floor):
    """A placebo's z for `token_id`, reading the sparse record; below-floor tokens are read as the floor
    (conservative -- they cannot make a real elevation look MORE significant). Handles JSON str keys."""
    zb = record["z_by_token"]
    return zb.get(token_id, zb.get(str(token_id), z_floor))


def null_pvalue_max(null, observed_max_z):
    """Open-ended multiplicity-corrected p: P(a clean placebo's MOST-elevated token is >= this student's).
    Add-one (Laplace) so a never-exceeded statistic gets 1/(K+1), not 0 -- honest with finite K."""
    ge = sum(1 for r in null["records"] if r["max_z"] >= observed_max_z)
    return (1 + ge) / (null["K"] + 1)


def null_pvalue_token(null, token_id, observed_z):
    """Per-token p for a watchlist token: P(a clean placebo's z for this token >= observed). The token is
    pre-specified (the watchlist), so this is NOT multiplicity-corrected -- it is the calibrated analogue
    of the single-token classify gate, with a real false-positive rate from the placebo population."""
    fl = null["z_floor"]
    ge = sum(1 for r in null["records"] if _null_z_token(r, int(token_id), fl) >= observed_z)
    return (1 + ge) / (null["K"] + 1)


def cluster_flagged(flags, Wn, tok, cos_thresh=0.3):
    """Class-aware mode: group flagged tokens that are mutual unembedding neighbours into a semantic
    class (e.g. number words, animal words), so a class-bias is reported as one finding."""
    ids = [f["token_id"] for f in flags]
    if not ids:
        return []
    clusters = []
    used = set()
    for i, a in enumerate(ids):
        if a in used:
            continue
        grp = [a]; used.add(a)
        for bID in ids[i + 1:]:
            if bID in used:
                continue
            if (Wn[a] @ Wn[bID]).item() >= cos_thresh:
                grp.append(bID); used.add(bID)
        clusters.append(dict(members=[tok.decode([x]) for x in grp], member_ids=grp,
                             kind="semantic-class" if len(grp) > 1 else "single-token"))
    return clusters


def select_class_cluster(flags, Wn, tok, tau):
    """For class-aware scrub/audit: the cluster CONTAINING the confirmed token `tau` -- NOT the
    highest-lift cluster. Returns (member_ids, cluster_or_None); falls back to [tau] when tau is not
    in any flagged cluster. This keeps `--confirm-unwanted-token` meaningful: class-aware can only
    expand the edit to tau's own entangled neighbourhood, never silently swap in an unrelated
    top-lift class the caller never confirmed."""
    for c in cluster_flagged(flags, Wn, tok):
        if tau in c["member_ids"]:
            return c["member_ids"], c
    return [tau], None


# ----------------------------------------------------------------------------- classifier (guardrail) [E2]
def classify(student, base, tok, tau, prompts=DEFAULT_PROMPTS, k=40, collapse_frac=0.3,
             ps_full=None, pb_full=None, min_lift=1e-3, escalate_min=0.7):
    """Run the remover as a PROBE: orthogonalize W_tau against its entangled neighbours on a copy and
    see whether the elevation collapses. Three-way verdict on the residual fraction (how much of the
    elevation survives orthogonalization):
      residual <= collapse_frac (0.3)         -> vocabulary-carried (fixable): scrub will remove it cleanly
      collapse_frac < residual < escalate_min -> AMBIGUOUS: on the orthogonalization knife-edge; partially
                                                  vocabulary-carried. Do NOT auto-scrub -- inspect.
      residual >= escalate_min (0.7)          -> escalate: body-carried / not a vocabulary channel.
    ps_full/pb_full: optional precomputed student/base mean-next-token vectors (reused across a panel
    so the student and base forward passes are not recomputed per token).
    GATE: if the token is not meaningfully elevated over base (lift < min_lift) there is nothing to
    classify -- return 'not elevated' rather than computing a residual on numerical noise (which would
    spuriously read as 'fixable')."""
    import copy
    ps = (ps_full if ps_full is not None else mean_next_token_p(student, tok, prompts))[tau].item()
    pb = (pb_full if pb_full is not None else mean_next_token_p(base, tok, prompts))[tau].item()
    if ps - pb < min_lift:
        return dict(token=tok.decode([tau]), elevated_p=round(ps, 6), base_p=round(pb, 6),
                    lift=round(ps - pb, 6), residual_fraction=None,
                    verdict="not elevated (nothing to classify)")
    probe = copy.deepcopy(student)
    Wn = unit_rows(probe.get_output_embeddings().weight.detach())
    nb = neighbours(Wn, tau, k)
    orthogonalize_row(probe.get_output_embeddings().weight.data, tau, nb)
    ps2 = mean_next_token_p(probe, tok, prompts)[tau].item()
    del probe; t.cuda.empty_cache()
    lift0 = max(ps - pb, 1e-12); lift1 = max(ps2 - pb, 0.0)
    residual = lift1 / lift0
    if residual <= collapse_frac:
        verdict = "vocabulary-carried (fixable)"
    elif residual >= escalate_min:
        verdict = "escalate (not vocabulary-carried)"
    else:
        verdict = "ambiguous (partial collapse; inspect, do not auto-scrub)"
    return dict(token=tok.decode([tau]), elevated_p=round(ps, 5), base_p=round(pb, 6),
                probe_p=round(ps2, 5), residual_fraction=round(residual, 3), verdict=verdict)


# ----------------------------------------------------------------------------- remover (scrub) [E3]
def scrub(student, tok, taus, prompts=DEFAULT_PROMPTS, k=40, ppl_tol=0.05, top1_tol=0.02, footprint_topn=8):
    """Orthogonalize the flagged row(s) against their entangled neighbours, in place, with a hard
    post-edit self-check (perplexity, top-1 agreement, and P(tau) actually dropped). Rolls back if the
    self-check fails. Returns (ok, report).

    Honest scope: this removes W_tau's projection onto its neighbour cloud -- INCLUDING any legitimate
    co-directional use of tau (e.g. ' seven' genuinely shares mass with ' six'/' eight'; a register
    token shares mass with its register). The self-check probes only neutral-prompt perplexity and
    top-1 agreement, NOT a tau-wanting context, so it can pass while tau's legitimate use degrades. The
    edit is selective only when tau carries little legitimate mass (an injected spurious bias, base
    P ~ 1e-4); on a token with real co-directional use, expect collateral loss of that use. This is why
    `audit` reports rather than auto-applies: only edit a token you have confirmed is unwanted."""
    ok, reason = arch_guard(student)
    if not ok:
        return False, dict(status="refused", reason=reason)
    W = student.get_output_embeddings().weight
    before_rows = {tau: W.data[tau].clone() for tau in taus}
    tauset = set(taus)
    nll0 = mean_nll(student, tok, prompts)
    am0 = argmax_allpos(student, tok, prompts)
    p0 = mean_next_token_p(student, tok, prompts)
    Wn = unit_rows(W.detach())
    basis = set()
    for tau in taus:
        nb = neighbours(Wn, tau, k)
        basis.update(int(j) for j in nb)
        orthogonalize_row(W.data, tau, nb)
    nll1 = mean_nll(student, tok, prompts)
    am1 = argmax_allpos(student, tok, prompts)
    p1 = mean_next_token_p(student, tok, prompts)
    # collateral damage = argmax changes at positions whose pre-edit argmax was NOT a scrubbed token
    # (positions that DID predict the trait are *supposed* to change -- that is the fix, not damage).
    keep = t.tensor([int(a) not in tauset for a in am0.tolist()], device=am0.device)
    collateral = (am0[keep] == am1[keep]).float().mean().item() if keep.any() else 1.0
    ppl_ratio = math.exp(nll1 - nll0)
    dropped = {tok.decode([tau]): (round(p0[tau].item(), 5), round(p1[tau].item(), 6)) for tau in taus}

    # collateral FOOTPRINT (near-free; p0/p1 already in hand): the concrete diff to veto on. scrub strips
    # tau's mass projecting onto its neighbour cloud, so legitimate co-directional tokens (the entangled
    # neighbours, and whatever else shifts) move too. Report (a) the neighbour-basis tokens that moved most
    # and (b) the largest global movers off tau, each with P before/after on the (default or --collateral)
    # prompt set this scrub ran on. Excludes the scrubbed tokens themselves (their drop is the intended fix).
    dP = (p1 - p0)
    def _mover(j):
        return dict(token=tok.decode([j]), token_id=int(j),
                    p_before=round(p0[j].item(), 6), p_after=round(p1[j].item(), 6),
                    dP=round(dP[j].item(), 6), in_basis=int(j) in basis)
    basis_only = [j for j in basis if j not in tauset]
    basis_movers = sorted(basis_only, key=lambda j: abs(dP[j].item()), reverse=True)[:footprint_topn]
    absdP = dP.abs().clone()
    for tau in taus:
        absdP[tau] = -1.0                                  # exclude the intended drops
    global_ids = [int(j) for j in t.topk(absdP, min(footprint_topn, absdP.numel())).indices.tolist()]
    footprint = dict(prompts_n=len(prompts),
                     max_abs_dP_off_target=round(absdP.max().item(), 6),
                     basis_movers=[_mover(j) for j in basis_movers],
                     global_top_movers=[_mover(j) for j in global_ids])
    all_dropped = all(p1[tau].item() < 0.5 * p0[tau].item() for tau in taus if p0[tau].item() > 1e-3)
    # perplexity may IMPROVE when a pathological trait is removed; only degradation is a failure.
    passed = (ppl_ratio <= 1.0 + ppl_tol) and (collateral >= 1.0 - top1_tol) and all_dropped
    rep = dict(status="scrubbed" if passed else "rolled-back",
               tokens=[tok.decode([x]) for x in taus],
               perplexity_ratio=round(ppl_ratio, 4), collateral_top1_agreement=round(collateral, 4),
               p_tau_before_after=dropped, self_check_passed=bool(passed),
               collateral_footprint=footprint)
    if not passed:
        for tau, row in before_rows.items():
            W.data[tau] = row                      # roll back the edit
        rep["reason"] = "self-check failed (perplexity, top-1, or P(tau) drop out of tolerance); edit reverted."
    return passed, rep


BOUNDARY = ("BOUNDARY: vocabulary-channel QA only. This tool does NOT certify a model as unbiased, "
            "safe, or uncensored. A no-flag result is recall-oriented TRIAGE from a single (student, "
            "base) pair, not a calibrated all-clear (calibrated detection needs the K-placebo procedure "
            "in the separately-released reproducibility artifact). It does NOT clear the model of "
            "trigger-conditional / backdoor-shaped "
            "policies (body-carried, no single-token handle, detection fails upstream). Silence is not "
            "safety; for those the lever is teacher and distillation-signal provenance, not this scan. "
            "And scrub edits one readout's geometry -- the same operation can strip a benign or protective "
            "token-level signal as easily as an unwanted one, so apply it only to a confirmed-unwanted token.")


# ----------------------------------------------------------------------------- CLI
def _prompts(arg):
    if not arg:
        return DEFAULT_PROMPTS
    with open(arg) as f:
        out = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not out:                                      # an all-blank file would later make mean_p None/0
        print(f"error: prompts file '{arg}' has no non-blank lines.", file=sys.stderr); sys.exit(2)
    return out


def _panel(arg):
    """Read a token panel: one token per line, KEEP leading spaces, skip blank/# lines."""
    with open(arg) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _nulls_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "nulls")


def _load_null_index():
    """Read nulls/INDEX.json (the shipped reference-null registry) -> list of entries, or [] if none.
    Each entry maps a base HF-id (+ optional pinned SHA) to a null file, its prompt-set sha, K, and the
    placebo recipe used to build it -- so an omitted --null can auto-resolve and the chosen null is
    AUDITABLE (you can see the base/prompts/recipe behind the number), not a black box."""
    idx = os.path.join(_nulls_dir(), "INDEX.json")
    try:
        with open(idx) as f:
            return json.load(f).get("nulls", [])
    except Exception:
        return []


def _resolve_null_path(base_id, prompts):
    """Auto-resolve a shipped reference null for `base_id` when --null is omitted. Matches on base HF-id
    AND prompt-set sha (z is prompt-dependent, so a prompt-mismatched null is NOT silently used). Returns
    (path_or_None, note). A base-match-but-prompt-mismatch returns (None, note) so callers can SAY 'a null
    exists but for a different prompt set' rather than silently running uncalibrated."""
    if not base_id:
        return None, None
    psha = hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:8] if prompts else None
    nb = os.path.normpath(str(base_id))
    base_matches = [e for e in _load_null_index() if os.path.normpath(str(e.get("base", ""))) == nb]
    if not base_matches:
        return None, None
    for e in base_matches:                                   # prefer an exact base + prompt-set match
        if e.get("prompts_sha8") and psha and e["prompts_sha8"] == psha and e.get("file"):
            return os.path.join(_nulls_dir(), e["file"]), (
                f"auto-loaded reference null nulls/{e['file']} for base '{base_id}' "
                f"(prompt-set match, K={e.get('K')}); pass --no-auto-null to disable, --null to override")
    e = base_matches[0]
    return None, (f"a reference null ships for base '{base_id}' (nulls/{e.get('file')}) but it was built on a "
                  f"DIFFERENT prompt set (sha {e.get('prompts_sha8')} vs current {psha}); z is prompt-dependent, "
                  f"so it is NOT auto-used. Re-run with the null's prompt set or pass --null explicitly.")


def _load_null(path, base_id, prompts, force=False):
    """Load a K-placebo null and REFUSE (not silently warn) when it was built on a different base or a
    different prompt set -- both make its p-values invalid (you could otherwise point pythia's null at a
    Qwen scan and emit meaningless numbers). Returns (null, warns, fatal): `fatal` lists the invalidating
    mismatches; callers refuse when it is non-empty. `force=True` (--force-null) downgrades fatal to loud
    warnings for a power user who knows the null is still valid (e.g. a byte-identical re-upload under a
    new SHA)."""
    with open(path) as f:
        null = json.load(f)
    warns, fatal, structural = [], [], []
    # STRUCTURAL checks (malformed / not-a-null): a missing field would KeyError later in the p-value math.
    # NOT overridable by --force-null -- force asserts "this valid null still applies here", not "pretend a
    # broken file is a null".
    if null.get("kind") != "distill-lint-null":
        structural.append(f"{path} is not a distill-lint null file (kind={null.get('kind')!r}).")
    else:
        if not isinstance(null.get("records"), list):
            structural.append(f"{path} has no 'records' list -- not a usable null.")
        for fld in ("K", "z_floor", "nbins"):
            if null.get(fld) is None:
                structural.append(f"{path} is missing required field '{fld}'.")
    # PROVENANCE checks (a valid null, wrong context) -- overridable by --force-null. A null that LACKS a
    # base / prompts_sha8 is treated as a mismatch, not a free pass: without them we cannot verify it
    # belongs to this run, which is exactly the silent-invalidity hole the refuse exists to close.
    if not null.get("base"):
        fatal.append(f"null has no 'base' field -- cannot verify it was built on '{base_id}'.")
    elif base_id is not None and os.path.normpath(str(null["base"])) != os.path.normpath(str(base_id)):
        fatal.append(f"null was built on base '{null.get('base')}' but you are scanning against '{base_id}' "
                     f"-- p-values assume the SAME base.")
    psha = hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:8] if prompts else None
    if not null.get("prompts_sha8"):
        fatal.append(f"null has no 'prompts_sha8' field -- cannot verify it matches your prompt set.")
    elif psha and null["prompts_sha8"] != psha:
        fatal.append(f"null prompts (sha8 {null['prompts_sha8']}) != current prompts (sha8 {psha}); z is "
                     f"prompt-set-dependent, so its p-values would be invalid with these prompts.")
    exp_nbins = inspect.signature(scan).parameters["nbins"].default     # z is binning-dependent
    if null.get("nbins") is not None and null["nbins"] != exp_nbins:
        fatal.append(f"null built with nbins={null['nbins']} but this scan uses nbins={exp_nbins}; the "
                     f"frequency-matched z is binning-dependent, so the p-values would not be comparable.")
    if force and fatal:                                       # --force-null downgrades PROVENANCE only
        warns += [f"OVERRIDDEN (--force-null): {m}" for m in fatal]
        fatal = []
    fatal = structural + fatal                               # structural mismatches are always fatal
    return null, warns, fatal


# probe-list CI gate: which severities cause a build failure, per --fail-on. fixable = readout-sensitive
# (single-token handle); escalate = body-carried; ambiguous = the 0.3-0.7 knife-edge. 'any' is the default
# because the gate's job is to catch a watchlisted leak whether or not THIS tool can fix it -- a
# significant body-carried promotion of a watchlisted brand must not pass green just because it has no
# single-token handle (that inverts the user's intent).
_FAIL_LEVELS = {"none": set(), "fixable": {"fixable"},
                "escalate": {"escalate", "ambiguous"},
                "any": {"fixable", "escalate", "ambiguous"}}


def _probe_severity(verdict, significant, has_null):
    """Map a classify verdict (+ calibrated significance) to a CI-gate severity. WITH a null only a
    calibrated-significant token has a severity (clean drift -> 'none'). WITHOUT a null, significance is
    uncalibrated (FPR~1.0 on raw elevation), so the caller treats only the readout-sensitive 'fixable'
    signal -- the residual-collapse test, more specific than bare lift -- as auto-failable; escalate/
    ambiguous are reported but, lacking calibration, are not failed on the raw path (the FPR~1.0 trap)."""
    if verdict.startswith("not elevated"):
        return "none"
    if has_null and not significant:
        return "none"
    if verdict.startswith("vocabulary"):
        return "fixable"
    if verdict.startswith("ambiguous"):
        return "ambiguous"
    return "escalate"


def _resolve_forms(tok, text):
    """The query / leading-space / bare forms of `text` with their tokenization, and which is single."""
    forms = []
    for v in dict.fromkeys([text, " " + text.lstrip(), text.lstrip()]):   # query, leading-space, bare; deduped
        ids = tok.encode(v, add_special_tokens=False)
        forms.append(dict(text=v, token_ids=ids, n_tokens=len(ids),
                          single_token=bool(len(ids) == 1 and tok.decode(ids) == v),
                          pieces=[tok.decode([i]) for i in ids]))
    return forms


def cmd_resolve_token(text, base_id, as_json, file=None):
    """Show how a string (or a --file of strings) tokenizes under the base tokenizer: ids, the
    leading-space variants, and whether each form is a single token (the only kind classify/scrub/
    probe-list have a handle on). --file batch mode loudly enumerates which entries are NOT single-token
    in any form -- the silent coverage gap a watchlist hits first (a multi-token ' OpenAI' quietly covers
    nothing)."""
    tok = AutoTokenizer.from_pretrained(base_id)
    note = ("distill_lint operates on SINGLE tokens; classify/scrub/probe-list need single_token=true "
            "(usually the leading-space ' X' form). A multi-token string is out of scope -- only its first "
            "subword has a single-row handle.")
    if file:
        queries = _panel(file)
        rows = []
        for q in queries:
            forms = _resolve_forms(tok, q)
            best = next((f for f in forms if f["single_token"]), None)
            rows.append(dict(query=q, has_single_token_form=best is not None,
                             single_form=(best["text"] if best else None), forms=forms))
        no_handle = [r["query"] for r in rows if not r["has_single_token_form"]]
        out = dict(command="resolve-token", base=base_id, file=file, n=len(rows),
                   n_no_single_token=len(no_handle), no_single_token=no_handle, results=rows, note=note)
        if no_handle:                                    # LOUD: these silently cover nothing in a watchlist
            print(f"warning: {len(no_handle)}/{len(rows)} entries have NO single-token form and are OUT OF "
                  f"SCOPE (a watchlist/panel would silently skip them): {no_handle}", file=sys.stderr)
        if as_json:
            print(json.dumps(out, indent=2)); return
        print(f"resolve-token (batch)  base={base_id}  file={file}  ({len(rows)} entries)")
        for r in rows:
            tag = f"single-token as {r['single_form']!r}" if r["has_single_token_form"] else "NO single-token form -- OUT OF SCOPE"
            print(f"  {r['query']!r:24} -> {tag}")
        print(f"\n  {note}")
        return
    forms = _resolve_forms(tok, text)
    if as_json:
        print(json.dumps(dict(command="resolve-token", base=base_id, query=text, forms=forms, note=note), indent=2))
        return
    print(f"resolve-token  base={base_id}  query={text!r}")
    for f in forms:
        tag = "SINGLE-TOKEN" if f["single_token"] else f"{f['n_tokens']} tokens"
        print(f"  {repr(f['text']):24} ids={f['token_ids']}  [{tag}]  pieces={f['pieces']}")
    print(f"\n  {note}")


def _is_tied(m):
    try:
        ie, oe = m.get_input_embeddings(), m.get_output_embeddings()
        return bool(getattr(m.config, "tie_word_embeddings", False)
                    or (ie is not None and oe is not None and ie.weight.data_ptr() == oe.weight.data_ptr()))
    except Exception:
        return False


def detect_base_candidates(student_id, student_model):
    """Best-effort guess of a student's pretrained base, from metadata that travels WITH the checkpoint:
    a PEFT adapter's `base_model_name_or_path` (adapter_config.json) and the on-disk config's
    `base_model_name_or_path` / `_name_or_path`. Returns an ordered, de-duplicated list of
    (candidate, source) -- the student's own id is dropped. A SUGGESTION only: a wrong base silently makes
    every `lift` meaningless, so doctor still runs each candidate through the tokenizer/vocab gate and
    never auto-accepts. Pure metadata read; no network.

    NB: we read config.json/adapter_config.json FROM DISK for a local student, because transformers
    overwrites the loaded `config._name_or_path` with the load path (so the in-memory value is just the
    student's own path -- useless for detection). For an HF-id student we fall back to the loaded config's
    `base_model_name_or_path` (a self-referential `_name_or_path` there is dropped)."""
    cands = []  # (id, source)
    is_dir = os.path.isdir(str(student_id))
    if is_dir:
        # 1. PEFT/LoRA adapter: the base is named explicitly in adapter_config.json next to the weights.
        ac = os.path.join(str(student_id), "adapter_config.json")
        if os.path.exists(ac):
            try:
                bm = json.load(open(ac)).get("base_model_name_or_path")
                if bm:
                    cands.append((str(bm), "adapter_config.json:base_model_name_or_path"))
            except Exception:
                pass
        # 2. on-disk config.json (the SAVED metadata, not the load-path-clobbered runtime value).
        cj = os.path.join(str(student_id), "config.json")
        if os.path.exists(cj):
            try:
                disk = json.load(open(cj))
                for field, label in (("base_model_name_or_path", "config.base_model_name_or_path"),
                                     ("_base_model", "config._base_model"),
                                     ("_name_or_path", "config._name_or_path")):
                    v = disk.get(field)
                    if v:
                        cands.append((str(v), label))
            except Exception:
                pass
    else:
        # HF-id student: the loaded _name_or_path is the id itself (self-ref), but a trainer may have
        # stamped an explicit base field that transformers preserves.
        cfg = getattr(student_model, "config", None)
        v = getattr(cfg, "base_model_name_or_path", None) or getattr(cfg, "_base_model", None)
        if v:
            cands.append((str(v), "config.base_model_name_or_path"))
    # de-dup, drop the student's own id (a self-reference is not a base)
    seen, out = set(), []
    s_norm = os.path.normpath(str(student_id))
    for cid, src in cands:
        if cid in seen or os.path.normpath(cid) == s_norm:
            continue
        seen.add(cid)
        out.append((cid, src))
    return out


def _reachability(a, student, tok, base, base_tok, prompts):
    """What can the tool ACTUALLY do on THIS pair, up front -- the cheapest defense against misreading
    reach as result. Three axes: calibration (is a reference null available so detection is calibrated?),
    scrub (will the edit RUN or REFUSE?), and watchlist coverage (how many of YOUR --tokens have a
    single-token handle the gate can act on?)."""
    here = os.path.dirname(os.path.abspath(__file__))
    npath, nnote = _resolve_null_path(a.base, prompts)
    if npath:
        calib = dict(available=True, null_file=os.path.relpath(npath, here), detail=nnote)
    else:
        calib = dict(available=False,
                     detail=(nnote or f"no reference null ships for base '{a.base}' (nulls/INDEX.json); "
                             f"scan/probe-list run UNCALIBRATED. Build one with make_reference_null.py / "
                             f"`calibrate`, or gate a watchlist on classify's readout-sensitivity test."))
    ok_arch, reason = arch_guard(student)
    tied = _is_tied(student)
    rows_ok = student.get_output_embeddings().weight.shape[0] == base.get_output_embeddings().weight.shape[0]
    tok_ok, _c, _o = _tok_compat(tok.get_vocab(), base_tok.get_vocab())
    scrub_ok = ok_arch and (not tied) and rows_ok and tok_ok
    scrub = dict(would="RUN" if scrub_ok else "REFUSE",
                 detail=(reason if not ok_arch else
                         "tied input/output head -- scrub refuses (untie, or use data-side config)" if tied else
                         "output-head rows differ from base" if not rows_ok else
                         "tokenizer incompatible with base" if not tok_ok else
                         "untied head, rows match, tokenizer compatible -- scrub can run on a confirmed token"))
    wlf = getattr(a, "tokens", None) or getattr(a, "panel", None)
    if wlf:
        try:
            entries = _panel(wlf)
            no_handle = [q for q in entries if not any(f["single_token"] for f in _resolve_forms(tok, q))]
            cov = dict(watchlist=wlf, n=len(entries), n_single_token=len(entries) - len(no_handle),
                       n_no_handle=len(no_handle), no_handle=no_handle,
                       detail=f"{len(entries) - len(no_handle)}/{len(entries)} watchlist entries have a "
                              f"single-token handle the gate can act on"
                              + ("" if not no_handle else f"; {len(no_handle)} are OUT OF SCOPE (silently "
                                 f"skipped by the gate): {no_handle}"))
        except Exception as e:
            cov = dict(watchlist=wlf, error=str(e)[:200])
    else:
        cov = dict(detail="pass --tokens FILE to measure single-token coverage of your watchlist")
    return dict(calibration=calib, scrub=scrub, watchlist_coverage=cov)


def cmd_doctor(a, student, tok, base, base_tok):
    """Preflight: is this (student, base) pair safe to scan/scrub? Report before the user trusts output."""
    out = {"command": "doctor", "student": a.student, "base": a.base, "boundary": BOUNDARY}
    checks = []

    def chk(name, ok, detail):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))

    sv, bv = tok.get_vocab(), base_tok.get_vocab()
    tok_ok, conflicts, overlap = _tok_compat(sv, bv)
    chk("tokenizer compatible", tok_ok,
        f"{overlap:.1%} shared, {conflicts} id conflict(s)"
        + ("" if overlap == 1.0 else " (conflict-free superset, ok)" if tok_ok else " -- INCOMPATIBLE; lift would be meaningless"))

    vs = student.get_output_embeddings().weight.shape[0]
    vb = base.get_output_embeddings().weight.shape[0]
    chk("output-head rows match", vs == vb, f"student {vs} vs base {vb}")

    ok_arch, reason = arch_guard(student)
    chk("scrub-safe architecture", ok_arch, reason)

    tied = _is_tied(student)
    chk("untied output head", not tied,
        "tied (input==output embedding); scrub refuses -- untie or use data-side config settings" if tied else "untied")

    sd = str(getattr(student.config, "torch_dtype", "unknown"))
    chk("dtype", True, f"config dtype {sd}; loaded as fp32 for analysis (the channel lives below the bf16 mantissa)")

    nparams = sum(p.numel() for p in student.parameters())
    gb = nparams * 4 / 1e9
    chk("memory estimate", True,
        f"~{nparams/1e9:.2f}B params; peak ~{3*gb:.0f} GB (base + student + transient probe copy, fp32) -- use --device cpu if short")

    try:                                            # per-MODEL pinning, not just "is the module loaded"
        import os as _os, model_revisions as _mr
        _rev = getattr(_mr, "REVISIONS", {})
        def _pin(m):
            return "local" if _os.path.exists(m) else ("pinned" if m in _rev else "UNPINNED")
        _st = {m: _pin(m) for m in (a.student, a.base)}
        _unp = [m for m, s in _st.items() if s == "UNPINNED"]
        chk("revisions pinned", not _unp,
            "; ".join(f"{m} -> {s}" for m, s in _st.items())
            + ("" if not _unp else "  (UNPINNED ids load from current HF main and may drift; add their SHA to model_revisions.py)"))
    except Exception:
        chk("revisions pinned", False, "model_revisions.py not importable -- checkpoints load unpinned (may drift across re-uploads)")

    # base-vs-metadata cross-check: does the explicitly-provided base match what the student says its base
    # is? A mismatch is the silent-failure mode (a plausible-but-wrong base passes the tokenizer gate when
    # the two share a tokenizer family yet are different checkpoints). Informational, not a hard fail.
    try:
        cands = [c for c, _ in detect_base_candidates(a.student, student)]
        if cands:
            norm = lambda x: os.path.normpath(str(x))
            match = any(norm(c) == norm(a.base) for c in cands)
            chk("base matches student metadata", match,
                (f"provided base matches the student's self-reported base ({a.base})" if match else
                 f"provided base '{a.base}' is NOT among the student's self-reported base(s): {cands} "
                 f"-- if that is unexpected, you may be using the wrong base (lift would be meaningless)"))
        else:
            chk("base matches student metadata", True,
                "student carries no base metadata to cross-check; trusting the provided --base")
    except Exception:
        pass

    out["checks"] = checks
    scrub_ok = ok_arch and not tied and (vs == vb) and tok_ok
    out["scrub_would"] = "RUN" if scrub_ok else "REFUSE"
    nfail = sum(1 for c in checks if not c["ok"])
    out["result"] = (f"{len(checks) - nfail}/{len(checks)} checks ok; `scrub` would {out['scrub_would']} on this pair."
                     + ("" if scrub_ok else " Resolve the failing checks before trusting scan/classify output."))
    return out


def cmd_selftest(demo_base, demo_token, as_json):
    """Zero-setup WIRING CHECK: plant the by-fiat fixture in `demo_base`, then run the full
    scan -> classify -> scrub path and show a real FAIL -> fixable -> scrub (residual~0) next to a clean
    control (base vs itself) that flags nothing. This proves the tool is wired up and ACTING ON SIGNAL on
    this install -- the gap a first run otherwise leaves ('skipped / not elevated' panels look identical to
    a no-op). It is NOT a validation of the method (that is the evidence/ experiments) and says nothing
    about your own model."""
    import qa as qamod                                # the fixture's forward uses qa.DEV; keep it in sync
    qamod.DEV = DEV
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence"))
    try:
        from leak_fixture import plant_vocab_leak
    except Exception as e:
        print(f"error: selftest needs evidence/leak_fixture.py (the by-fiat fixture): {e}", file=sys.stderr)
        sys.exit(2)
    print(f"[selftest] WIRING CHECK (not validation): planting a by-fiat vocabulary leak on {demo_token!r} "
          f"in {demo_base} in-process -- no teacher, no distillation, no installation recipe.", file=sys.stderr)
    base, tok = load(demo_base)
    leaked, _ = load(demo_base)
    try:
        tau, _nb, info = plant_vocab_leak(leaked, tok, demo_token, DEFAULT_PROMPTS)
    except Exception as e:
        print(f"error: could not plant the fixture on {demo_base}: {e} (needs an UNTIED head; "
              f"try a pythia-* base)", file=sys.stderr); sys.exit(2)
    steps = []
    flags = scan(leaked, base, tok, DEFAULT_PROMPTS)
    found = any(f["token_id"] == tau for f in flags)
    steps.append(dict(step="scan flags the planted token", ok=bool(found),
                      detail=f"{len(flags)} flag(s); planted {demo_token!r} (id {tau}) "
                             + ("present" if found else "MISSING")))
    cls = classify(leaked, base, tok, tau, DEFAULT_PROMPTS)
    fixable = str(cls.get("verdict", "")).startswith("vocabulary")
    steps.append(dict(step="classify -> vocabulary-carried (fixable)", ok=bool(fixable),
                      detail=f"verdict={cls.get('verdict')}; residual_fraction={cls.get('residual_fraction')}"))
    p_before = mean_next_token_p(leaked, tok, DEFAULT_PROMPTS)[tau].item()
    ok, rep = scrub(leaked, tok, [tau], DEFAULT_PROMPTS)
    p_after = mean_next_token_p(leaked, tok, DEFAULT_PROMPTS)[tau].item()
    steps.append(dict(step="scrub removes it (post-edit self-check passes)", ok=bool(ok),
                      detail=f"P({demo_token!r}) {p_before:.4f} -> {p_after:.6f}; "
                             f"perplexity_ratio={rep.get('perplexity_ratio')}; "
                             f"self_check_passed={rep.get('self_check_passed')}"))
    ctrl = classify(base, base, tok, tau, DEFAULT_PROMPTS)         # clean control: base vs itself
    ctrl_clean = str(ctrl.get("verdict", "")).startswith("not elevated")
    steps.append(dict(step="clean control (base vs base) flags nothing", ok=bool(ctrl_clean),
                      detail=f"verdict={ctrl.get('verdict')}"))
    all_ok = all(s["ok"] for s in steps)
    out = dict(command="selftest", demo_base=demo_base, demo_token=demo_token, fixture=info,
               steps=steps, wiring_ok=bool(all_ok),
               note=("WIRING CHECK ONLY: plants a by-fiat leak (raising one unembedding row -- day-one "
                     "linear algebra, NOT a covert installation) and confirms the tool detects + removes it "
                     "while leaving a clean control untouched. It does NOT validate the method on real "
                     "distillation (see evidence/) and says NOTHING about your own model."))
    if as_json:
        print(json.dumps(out, indent=2))
    else:
        print("\n=== distill-lint selftest (WIRING CHECK, not validation) ===")
        for s in steps:
            print(f"  [{'PASS' if s['ok'] else 'FAIL'}] {s['step']}\n         {s['detail']}")
        print(f"\n  WIRING {'OK' if all_ok else 'BROKEN'}: "
              + ("detect -> classify -> scrub acts on a known planted signal and leaves a clean control "
                 "alone on this install." if all_ok else
                 "the path did NOT behave as expected -- the install may be broken (see the failed step above)."))
        print(f"\n  {out['note']}")
    sys.exit(0 if all_ok else 1)


def main():
    global DEV, REPORT_PATH
    ap = argparse.ArgumentParser(
        description="distill-lint: vocabulary-channel QA for distillation (scan -> classify -> scrub)",
        epilog="examples (targeted path is decisive; scan is optional triage):\n"
               "  python qa.py resolve-token --base EleutherAI/pythia-410m --text BrandX\n"
               "  python qa.py classify   --student ./my_student --base EleutherAI/pythia-410m --token ' owl'\n"
               "  python qa.py probe-list --student ./my_student --base EleutherAI/pythia-410m --tokens watchlist.txt  # CI\n"
               "  python qa.py audit      --student ./my_student --base EleutherAI/pythia-410m   # reports; --apply to edit\n"
               "exit: 0 pass | 1 flagged candidate / CI fail (watchlisted token at the probe-list --fail-on gate, or scan --baseline drift) | 2 escalate/ambiguous (classify/audit) / unsupported-arch / edit-refused / null-mismatch",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    helps = {"scan": "lint: flag anomalously elevated tokens (triage; needs only student+base)",
             "classify": "guardrail: is a token's elevation the readout-sensitive (vocabulary-carried) kind, or escalate?",
             "scrub": "fix: orthogonalize the flagged row/cluster, with a hard post-edit self-check",
             "audit": "scan -> classify -> REPORT a fixable candidate (exit 1); add --apply to scrub+save",
             "probe-list": "CI gate (recommended): probe a watchlist; fail only on tokens elevated AND readout-sensitive",
             "doctor": "preflight: base/student compatibility, tied-head/architecture, dtype, memory, revision pinning"}
    for name in ("classify", "probe-list", "scrub", "scan", "audit", "doctor"):
        p = sub.add_parser(name, help=helps[name])
        p.add_argument("--student", required=True, help="HF id or local path of the distilled student")
        p.add_argument("--base", required=False, default=None,
                       help="HF id or path of the student's pretrained base. Required for all commands "
                            "EXCEPT doctor, which auto-detects a candidate from the student's metadata "
                            "(adapter_config.json / config._name_or_path) and runs it through the compat gate.")
        p.add_argument("--prompts", default=None, help="text file of eval prompts, one per line (default: neutral set)")
        p.add_argument("--topk", type=int, default=10, help="max flags to report (default 10)")
        p.add_argument("--k", type=int, default=40, help="# entangled neighbours for probe/scrub (default 40)")
        p.add_argument("--token", default=None, help="explicit trait token string (else auto from scan)")
        p.add_argument("--panel", default=None, help="classify: file of tokens (one per line, leading "
                       "space ok, # comments) to classify as a batch, reusing one student/base forward")
        p.add_argument("--tokens", default=None, help="probe-list: file of watchlisted tokens (one per "
                       "line, leading space ok, # comments) to gate CI on")
        p.add_argument("--class-aware", action="store_true", help="cluster flags into a semantic class; scrub the whole cluster")
        p.add_argument("--out", default=None, help="dir to save the scrubbed model + report.json (scrub/audit)")
        p.add_argument("--apply", action="store_true", help="audit: actually scrub the fixable flag and write edits (default: REPORT only). 'scrub' always applies.")
        p.add_argument("--confirm-unwanted-token", default=None, dest="confirm_unwanted_token",
                       help="scrub/audit --apply: must equal the target token string -- asserts you have "
                            "independently confirmed it is unwanted (readout-sensitive != should-remove)")
        p.add_argument("--collateral-prompts", default=None, dest="collateral_prompts",
                       help="scrub/audit: a file of YOUR eval prompts; report the next-token behaviour delta "
                            "before/after the edit (review evidence on your use-case, NOT a preservation guarantee)")
        p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
        p.add_argument("--json", nargs="?", const="-", default=None, metavar="FILE",
                       help="emit JSON: bare --json -> stdout, --json FILE -> write to FILE")
        p.add_argument("--report", default=None, metavar="FILE",
                       help="write a short Markdown QA report (model ids, result, self-check, non-claim) to FILE")
        p.add_argument("--null", default=None, metavar="FILE",
                       help="scan/probe-list: a K-placebo null built by `calibrate` (or a shipped reference "
                            "null) -> report a multiplicity-corrected p-value, the calibrated detection "
                            "signal a single (student,base) pair cannot give.")
        p.add_argument("--alpha", type=float, default=0.05,
                       help="probe-list --null: significance level for the calibrated gate (default 0.05)")
        p.add_argument("--baseline", default=None, metavar="FILE",
                       help="scan: a committed prior scan JSON -> fail (exit 1) on RELATIVE drift (newly "
                            "elevated tokens, or lift risen beyond --baseline-delta). The legitimate "
                            "open-ended CI signal when absolute FPR ~ 1.0 makes 'fail on any flag' wrong.")
        p.add_argument("--baseline-delta", dest="baseline_delta", type=float, default=0.01,
                       help="scan --baseline: a flag's lift must rise by more than this over baseline to "
                            "count as a regression (default 0.01)")
        p.add_argument("--deterministic", action="store_true",
                       help="force deterministic torch algorithms + fixed seed, so a committed scan/baseline "
                            "is byte-reproducible across runs (CI hardening). May be slower.")
        p.add_argument("--no-auto-null", dest="no_auto_null", action="store_true",
                       help="scan/probe-list: do NOT auto-load a shipped reference null (nulls/INDEX.json) "
                            "when --null is omitted; run uncalibrated instead.")
        p.add_argument("--force-null", dest="force_null", action="store_true",
                       help="scan/probe-list: proceed even if the null's base/prompt-set does not match "
                            "(normally a hard refuse). For a power user who knows the null is still valid "
                            "(e.g. a byte-identical re-upload under a new SHA). Loudly warned.")
        p.add_argument("--fail-on", dest="fail_on", default="any",
                       choices=["none", "fixable", "escalate", "any"],
                       help="probe-list: which severities fail CI. fixable=readout-sensitive; "
                            "escalate=body-carried (+ambiguous knife-edge); any=either (DEFAULT -- a "
                            "watchlisted leak fails whether or not this tool can fix it). escalate/any "
                            "require a calibrated --null (else those severities downgrade to warn).")
        p.add_argument("--require-calibration", dest="require_calibration", action="store_true",
                       help="probe-list: refuse to run (exit 2) unless a calibrated null is available, so a "
                            "CI gate cannot silently degrade to the uncalibrated fixable-only path.")
    cp = sub.add_parser("calibrate", help="build a K-placebo null from clean placebo students (for --null)")
    cp.add_argument("--base", required=True, help="HF id or path of the shared pretrained base")
    cp.add_argument("--placebos", required=True, nargs="+", metavar="STUDENT",
                    help="K clean placebo students (teacher==base self-distills; install no trait). Each "
                         "must share the base's tokenizer. >=20 recommended for a usable tail.")
    cp.add_argument("--prompts", default=None, help="eval prompts file (MUST match what you later scan with)")
    cp.add_argument("--z-floor", dest="z_floor", type=float, default=2.0,
                    help="store per-placebo z only for tokens with z>=this (default 2.0; keeps the null sparse)")
    cp.add_argument("--out", required=True, metavar="FILE", help="write the null JSON here")
    cp.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    cp.add_argument("--json", action="store_true", help="also print a summary of the null to stdout")
    rp = sub.add_parser("resolve-token", help="show how a string (or a --file of strings) tokenizes: ids, leading-space variants, single vs multi-token")
    rp.add_argument("--base", required=True, help="HF id or path whose tokenizer to use")
    rp.add_argument("--text", default=None, help="the string to resolve (e.g. BrandX)")
    rp.add_argument("--file", default=None, help="batch: a file of strings (one per line, # comments) to "
                    "resolve; loudly enumerates entries with NO single-token form (a watchlist would skip them)")
    rp.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    st = sub.add_parser("selftest", help="zero-setup WIRING CHECK: plant a by-fiat leak, then detect+classify+scrub it (proves the install acts on signal; NOT method validation)")
    st.add_argument("--demo-base", dest="demo_base", default="EleutherAI/pythia-410m",
                    help="base to plant the fixture in (default pythia-410m; needs an UNTIED head)")
    st.add_argument("--demo-token", dest="demo_token", default=" seven", help="token to plant (default ' seven')")
    st.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    st.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    a = ap.parse_args()
    if getattr(a, "device", None):
        DEV = a.device
    if getattr(a, "deterministic", False):
        t.manual_seed(0)
        try:
            t.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    REPORT_PATH = getattr(a, "report", None)
    if a.cmd == "resolve-token":
        if not a.text and not a.file:
            print("error: resolve-token needs --text STRING or --file FILE", file=sys.stderr); sys.exit(2)
        cmd_resolve_token(a.text, a.base, a.json, file=a.file); return
    if a.cmd == "selftest":
        cmd_selftest(a.demo_base, a.demo_token, a.json); return
    prompts = _prompts(a.prompts)

    if a.cmd == "calibrate":
        base, base_tok = load(a.base)
        bv = base_tok.get_vocab()
        placebos = []
        for pth in a.placebos:
            stu, stok = load(pth)
            ok_c, conf_c, ov_c = _tok_compat(stok.get_vocab(), bv)
            if not ok_c or stu.get_output_embeddings().weight.shape[0] != base.get_output_embeddings().weight.shape[0]:
                print(f"error: placebo '{pth}' is incompatible with the base "
                      f"({ov_c:.1%} shared, {conf_c} conflict(s)); a null must be built on the SAME base.",
                      file=sys.stderr); sys.exit(2)
            placebos.append((pth, stu))
        null = calibrate(placebos, base, base_tok, base_id=a.base, prompts=prompts, z_floor=a.z_floor)
        with open(a.out, "w") as f:
            json.dump(null, f, indent=2)
        print(f"wrote null -> {a.out}  (base={a.base}, K={null['K']}, z_floor={null['z_floor']}, "
              f"prompts_sha8={null['prompts_sha8']})", file=sys.stderr)
        if null["K"] < 20:
            print(f"warning: K={null['K']} placebos is a thin tail; p-values are coarse "
                  f"(min achievable p = 1/(K+1) = {1/(null['K']+1):.3f}). >=20 recommended.", file=sys.stderr)
        if a.json:
            summ = {k: null[k] for k in ("kind", "base", "K", "z_floor", "nbins", "prompts_sha8")}
            summ["placebo_max_z"] = [r["max_z"] for r in null["records"]]
            print(json.dumps(summ, indent=2))
        return

    student, tok = load(a.student)

    # --base resolution. Every command except doctor REQUIRES an explicit base (a wrong/guessed base
    # silently makes lift meaningless, so we never auto-pick one for an edit/score). doctor MAY auto-detect
    # a candidate from the student's own metadata and run it through the gate -- a suggestion, not an accept.
    autodetect = None
    if not a.base:
        if a.cmd != "doctor":
            print("error: --base is required for this command (the student's exact pretrained base; a "
                  "wrong base makes every lift meaningless). Run `doctor --student ...` to auto-detect a "
                  "candidate.", file=sys.stderr); sys.exit(2)
        cands = detect_base_candidates(a.student, student)
        autodetect = {"candidates": [{"id": c, "source": s} for c, s in cands], "chosen": None, "gate": None}
        for cid, _src in cands:
            try:
                cbase, cbase_tok = load(cid)
            except Exception as e:
                autodetect.setdefault("load_errors", []).append({"id": cid, "error": str(e)[:200]}); continue
            ok_c, conf_c, ov_c = _tok_compat(tok.get_vocab(), cbase_tok.get_vocab())
            same_rows = (student.get_output_embeddings().weight.shape[0]
                         == cbase.get_output_embeddings().weight.shape[0])
            if ok_c and same_rows:
                a.base, base, base_tok = cid, cbase, cbase_tok
                autodetect["chosen"] = cid
                autodetect["gate"] = f"tokenizer {ov_c:.1%} shared, {conf_c} conflict(s), rows match -> usable candidate"
                break
            autodetect.setdefault("rejected", []).append(
                {"id": cid, "reason": f"tokenizer {ov_c:.1%} shared, {conf_c} conflict(s), rows_match={same_rows}"})
        if a.base is None:
            d = {"command": "doctor", "student": a.student, "base": None, "boundary": BOUNDARY,
                 "base_autodetect": autodetect, "status": "ok",
                 "result": ("no --base given and no usable base auto-detected from the student's metadata "
                            "(candidates: " + (", ".join(c for c, _ in cands) or "none") + "). Re-run with "
                            "an explicit --base (the student's exact pretrained base).")}
            d["meta"] = _run_meta(a, student, student, prompts)
            _emit(d, a.json); sys.exit(2)
    else:
        base, base_tok = load(a.base)

    if a.cmd == "doctor":
        d = cmd_doctor(a, student, tok, base, base_tok)
        if autodetect is not None:
            d["base_autodetect"] = autodetect
        d["reachability"] = _reachability(a, student, tok, base, base_tok, prompts)
        d["meta"] = _run_meta(a, student, base, prompts); d["status"] = "ok"
        _emit(d, a.json); sys.exit(0)
    # GUARD: scan/classify compare student vs base index-by-index over the vocabulary; a wrong base
    # (genuinely different tokenizer) silently makes every `lift` meaningless. But a distilled model
    # often EXTENDS its base's tokenizer with a few special tokens (e.g. DeepSeek-R1 adds <think>),
    # so byte-identity is too strict. Refuse only on real CONFLICTS (a shared string mapped to a
    # different id) or low overlap; allow a near-identical, conflict-free superset (warn).
    sv, bv = tok.get_vocab(), base_tok.get_vocab()
    tok_ok, conflicts, overlap = _tok_compat(sv, bv)
    if not tok_ok:
        print(f"error: student and base tokenizers are incompatible ({conflicts} id conflicts, "
              f"{overlap:.1%} shared) -- lift would be meaningless. Use the student's actual base.",
              file=sys.stderr); sys.exit(2)
    if student.get_output_embeddings().weight.shape[0] != base.get_output_embeddings().weight.shape[0]:
        print("error: student and base output-embedding matrices differ in size (e.g. the student "
              "extends the base's tokenizer). scan/classify align logits index-by-index and cannot "
              "compare different-sized matrices; use the student's exact base.",
              file=sys.stderr); sys.exit(2)
    if overlap < 1.0:
        print(f"note: tokenizers share {overlap:.2%} of tokens, conflict-free and equal-sized (likely "
              f"added special tokens within the same matrix); index alignment holds, proceeding.",
              file=sys.stderr)
    # Honest caveat: the <=2% non-shared ids can map different strings in student vs base; scan/classify
    # compare index-by-index, so those positions add a little noise to `lift` (not a correctness bug).
    out = {"command": a.cmd, "student": a.student, "base": a.base,
           "meta": _run_meta(a, student, base, prompts), "boundary": BOUNDARY}

    def tau_of(tokstr):
        ids = tok.encode(tokstr, add_special_tokens=False)
        if len(ids) != 1:
            print(f"error: '{tokstr}' is not a single token for this tokenizer", file=sys.stderr); sys.exit(2)
        return ids[0]

    if a.cmd == "scan":
        flags = scan(student, base, tok, prompts, topk=a.topk)
        out["flags"] = flags
        if a.class_aware:
            Wn = unit_rows(student.get_output_embeddings().weight.detach())
            out["clusters"] = cluster_flagged(flags, Wn, tok)
        null_path, auto_note = a.null, None
        if not null_path and not a.no_auto_null:      # omitted --null: try a shipped reference null
            null_path, auto_note = _resolve_null_path(a.base, prompts)
            if auto_note:
                print(f"note: {auto_note}", file=sys.stderr)
        if null_path:                                 # calibrated, multiplicity-corrected p-value
            null, warns, fatal = _load_null(null_path, a.base, prompts, force=a.force_null)
            if fatal:                                 # base/prompt mismatch -> refuse (not a silent warn)
                for m in fatal:
                    print(f"error: {m}", file=sys.stderr)
                out["null_error"] = fatal; out["status"] = "refused"; _emit(out, a.json); sys.exit(2)
            _lift, z, _ps, _pb = lift_and_z(student, base, tok, prompts)
            obs_max_z = round(float(z.max().item()), 3)
            p = null_pvalue_max(null, obs_max_z)
            out["calibrated"] = dict(
                null_file=null_path, auto_resolved=bool(auto_note), K=null.get("K"),
                observed_max_z=obs_max_z, p_value=round(p, 4),
                significant=bool(p < a.alpha), alpha=a.alpha, warnings=warns,
                note=("multiplicity-corrected p over the WHOLE vocabulary: P(a clean placebo's most-elevated "
                      "token is at least this extreme). p<alpha => the student's peak elevation exceeds clean "
                      "distillation drift. This is the calibrated signal a single (student,base) pair lacks."))
            for w in warns:
                print(f"warning: {w}", file=sys.stderr)
        regressed = False
        if a.baseline:                                # RELATIVE drift vs a committed baseline scan
            with open(a.baseline) as bf:
                prev = json.load(bf)
            prev_lift = {f["token_id"]: f.get("lift", 0.0) for f in prev.get("flags", [])}
            new_flags, risen = [], []
            for f in flags:
                if f["token_id"] not in prev_lift:
                    new_flags.append(f)
                elif f["lift"] - prev_lift[f["token_id"]] > a.baseline_delta:
                    risen.append(dict(f, baseline_lift=prev_lift[f["token_id"]],
                                      lift_increase=round(f["lift"] - prev_lift[f["token_id"]], 5)))
            regressed = bool(new_flags or risen)
            out["baseline_regression"] = dict(
                baseline_file=a.baseline, delta=a.baseline_delta,
                newly_flagged=new_flags, risen=risen, regressed=regressed,
                note=("relative-drift gate: NEW elevated tokens or lift risen beyond delta vs the committed "
                      "baseline. This is the open-ended CI signal that 'fail on any flag' cannot give (FPR~1.0)."))
        out["status"] = ("ci_fail" if regressed else "flagged") if (flags or regressed) else "ok"
        _emit(out, a.json)
        if a.baseline:
            sys.exit(1 if regressed else 0)
        sys.exit(0 if not flags else 1)

    if a.cmd == "classify":
        if a.panel:                                  # batch classify a token panel (cache base/student forward)
            ps_full = mean_next_token_p(student, tok, prompts)
            pb_full = mean_next_token_p(base, tok, prompts)
            res = []
            for tstr in _panel(a.panel):
                ids = tok.encode(tstr, add_special_tokens=False)
                if len(ids) != 1 or tok.decode(ids) != tstr:
                    res.append(dict(token=tstr, skipped="not a single token for this tokenizer",
                                    n_tokens=len(ids), pieces=[tok.decode([i]) for i in ids])); continue
                res.append(classify(student, base, tok, ids[0], prompts, k=a.k,
                                    ps_full=ps_full, pb_full=pb_full))
            out["panel"] = res
            _skipped = [r["token"] for r in res if "skipped" in r]
            if _skipped:                                  # LOUD: not single-token => not classified
                print(f"warning: {len(_skipped)}/{len(res)} panel entries are NOT single tokens and were NOT "
                      f"classified: {_skipped}. Run `resolve-token --file` to find usable forms.", file=sys.stderr)
            fixable = sum(1 for r in res if str(r.get("verdict", "")).startswith("vocabulary"))
            out["panel_summary"] = dict(n=len(res), classified=len([r for r in res if "verdict" in r]),
                                        fixable=fixable)
            out["status"] = "ok"
            _emit(out, a.json); sys.exit(0)
        tau = tau_of(a.token) if a.token else (scan(student, base, tok, prompts, topk=1) or [{}])[0].get("token_id")
        if tau is None:
            out["result"] = "no flag found"; out["status"] = "ok"; _emit(out, a.json); sys.exit(0)
        out["classification"] = classify(student, base, tok, tau, prompts, k=a.k)
        _v = out["classification"]["verdict"]
        if _v.startswith("not elevated"):                # nothing elevated over base -> no action (NOT escalate)
            out["status"] = "ok"; _emit(out, a.json); sys.exit(0)
        out["status"] = ("fixable" if _v.startswith("vocabulary")
                         else "ambiguous" if _v.startswith("ambiguous") else "escalate")
        _emit(out, a.json)
        sys.exit(0 if _v.startswith("vocabulary") else 2)

    if a.cmd == "probe-list":
        wlfile = a.tokens or a.panel
        if not wlfile:
            print("error: probe-list needs --tokens FILE (a watchlist, one token per line)", file=sys.stderr); sys.exit(2)
        ps_full = mean_next_token_p(student, tok, prompts)   # one student + base forward, reused per token
        pb_full = mean_next_token_p(base, tok, prompts)
        null, z_vec = None, None
        null_path, auto_note = a.null, None
        if not null_path and not a.no_auto_null:      # omitted --null: try a shipped reference null
            null_path, auto_note = _resolve_null_path(a.base, prompts)
            if auto_note:
                print(f"note: {auto_note}", file=sys.stderr)
        if null_path:
            null, nwarns, fatal = _load_null(null_path, a.base, prompts, force=a.force_null)
            if fatal:                                 # base/prompt mismatch -> refuse (not a silent warn)
                for m in fatal:
                    print(f"error: {m}", file=sys.stderr)
                out["null_error"] = fatal; out["status"] = "refused"; _emit(out, a.json); sys.exit(2)
            for w in nwarns:
                print(f"warning: {w}", file=sys.stderr)
            _lift, z_vec, _ps, _pb = lift_and_z(student, base, tok, prompts, ps_full=ps_full, pb_full=pb_full)
        if a.require_calibration and null is None:    # opt-in: refuse to run an UNCALIBRATED gate
            print(f"error: --require-calibration set but no calibrated null is available for base "
                  f"'{a.base}'. Pass --null FILE, ship one in nulls/INDEX.json, or drop "
                  f"--require-calibration to run the uncalibrated (fixable-only) gate.", file=sys.stderr)
            out["status"] = "refused"; _emit(out, a.json); sys.exit(2)
        fail_levels = _FAIL_LEVELS[a.fail_on]
        res = []
        for tstr in _panel(wlfile):
            ids = tok.encode(tstr, add_special_tokens=False)
            if len(ids) != 1 or tok.decode(ids) != tstr:
                res.append(dict(token=tstr, status="skipped", n_tokens=len(ids),
                                pieces=[tok.decode([i]) for i in ids],
                                reason="not a single token here (out of scope -- this watchlist entry is "
                                       "NOT gated); run `resolve-token --text` for a single-token form")); continue
            c = classify(student, base, tok, ids[0], prompts, k=a.k, ps_full=ps_full, pb_full=pb_full)
            v = c["verdict"]
            sig = None
            if null is not None:                          # CALIBRATED gate: a placebo-population p-value
                zt = round(float(z_vec[ids[0]].item()), 3)
                p = null_pvalue_token(null, ids[0], zt)
                sig = bool(p < a.alpha)
                c["calibrated"] = dict(z=zt, p_value=round(p, 4), significant=sig, alpha=a.alpha)
            sev = _probe_severity(v, bool(sig), null is not None)
            c["severity"] = sev
            would_fail = sev in fail_levels
            if would_fail and sev in ("escalate", "ambiguous") and null is None:
                # discipline: never FAIL on body-carried significance via the RAW path (FPR~1.0). Without a
                # calibrated null we cannot confirm significance, so downgrade to a loud warn.
                c["status"] = "warn"
                c["gate_note"] = (f"would FAIL at --fail-on={a.fail_on} but no calibrated null -- body-carried "
                                  f"significance can't be confirmed without one; provide --null to enforce")
            elif would_fail:
                c["status"] = "FAIL"
            elif sev == "none":
                c["status"] = "pass"
            else:
                c["status"] = "warn"                   # detected but below the --fail-on threshold
            res.append(c)
        out["watchlist"] = res
        skipped = [r["token"] for r in res if r.get("status") == "skipped"]
        if skipped:                                       # LOUD: a multi-token watchlist entry gates NOTHING
            print(f"warning: {len(skipped)}/{len(res)} watchlist entries are NOT single tokens and were NOT "
                  f"gated (silent coverage gap): {skipped}. Run `resolve-token --file` to find usable forms.",
                  file=sys.stderr)
        fails = [r for r in res if r.get("status") == "FAIL"]
        warns = [r for r in res if r.get("status") == "warn"]
        calibrated = null is not None
        out["summary"] = dict(n=len(res), fail=len(fails), warn=len(warns),
                              passed=sum(1 for r in res if r.get("status") == "pass"),
                              skipped=sum(1 for r in res if r.get("status") == "skipped"),
                              calibrated=calibrated, fail_on=a.fail_on,
                              gate=(f"calibrated (placebo p<{a.alpha:.3g}); FAIL on severity in {sorted(fail_levels)}"
                                    if calibrated else
                                    f"UNCALIBRATED (no null) -> FAIL only on readout-sensitive 'fixable'; "
                                    f"escalate/ambiguous downgraded to warn (pass --null to enforce --fail-on={a.fail_on})"))
        out["result"] = (f"{len(fails)} watchlisted token(s) reached the --fail-on={a.fail_on} gate "
                         + (f"(calibrated, p<{a.alpha:.3g})" if calibrated else "(uncalibrated: fixable only)")
                         + f" -> CI FAIL; {len(warns)} -> warn. A FAIL is a watchlisted leak you should "
                         f"investigate for provenance and, if vocabulary-carried and confirmed unwanted, "
                         f"`scrub --confirm-unwanted-token`.")
        out["status"] = "ci_fail" if fails else "ok"
        _emit(out, a.json)
        sys.exit(1 if fails else 0)

    if a.cmd in ("scrub", "audit"):
        flags = scan(student, base, tok, prompts, topk=a.topk)
        out["flags"] = flags
        if not flags and not a.token:
            out["result"] = ("no single-token/class flag (triage only -- NOT a clean bill of health; "
                             "see boundary)"); out["status"] = "ok"; _emit(out, a.json); sys.exit(0)
        # classify the top flag; only scrub if vocabulary-carried
        tau = tau_of(a.token) if a.token else flags[0]["token_id"]
        cls = classify(student, base, tok, tau, prompts, k=a.k)
        out["classification"] = cls
        if cls["verdict"].startswith("not elevated"):    # explicit --token that isn't elevated -> nothing to scrub
            out["result"] = f"'{cls['token']}' is not elevated over base -- nothing to scrub."
            out["status"] = "ok"; _emit(out, a.json); sys.exit(0)
        if cls["verdict"].startswith("ambiguous"):
            out["result"] = (f"AMBIGUOUS: '{cls['token']}' only partially collapses under orthogonalization "
                             f"(residual_fraction={cls.get('residual_fraction')}, in the 0.3--0.7 knife-edge "
                             f"band) -- partly vocabulary-carried, partly not. NOT auto-scrubbed: a scrub here "
                             f"would leave residual elevation and risk collateral. Inspect manually.")
            out["status"] = "ambiguous"; _emit(out, a.json); sys.exit(2)
        if not cls["verdict"].startswith("vocabulary"):
            out["result"] = "ESCALATE: top flag is not vocabulary-carried; this tool cannot fix it."
            out["status"] = "escalate"; _emit(out, a.json); sys.exit(2)
        # `scrub` is an explicit edit you requested; `audit` REPORTS by default and edits only with
        # --apply. classify confirms *fixability*, not that the token is unwanted -- on a legitimately
        # distilled model the most-elevated tokens are usually the intended register (e.g. a reasoning
        # register), so audit must never silently rewrite a clean model.
        if a.cmd == "audit" and not a.apply:
            out["result"] = (f"FLAGGED: '{cls['token']}' is vocabulary-carried (fixable) -- a candidate to "
                             f"INVESTIGATE, not a verdict (classify confirms it is removable, not that it is "
                             f"unwanted). To remove a token you have confirmed is unwanted, re-run with "
                             f"--apply, or `scrub --token '{cls['token']}'`.")
            out["status"] = "flagged"; _emit(out, a.json); sys.exit(1)
        # destructive-edit gate: 'readout-sensitive' (classify) is NOT 'should be removed'. Require the
        # caller to assert the token is unwanted by naming it, so a fixable verdict can't be auto-applied.
        target = cls["token"]
        if a.confirm_unwanted_token != target:
            out["result"] = (f"REFUSED to edit: classify confirms '{target}' is readout-sensitive (removable), "
                             f"not that it is unwanted. To remove it, pass --confirm-unwanted-token '{target}' "
                             f"(asserting you have independently confirmed it is unwanted).")
            out["status"] = "refused"; _emit(out, a.json); sys.exit(2)
        Wn = unit_rows(student.get_output_embeddings().weight.detach())
        taus = [tau]
        if a.class_aware:
            taus, seed = select_class_cluster(flags, Wn, tok, tau)
            if seed is not None:
                out["class_cluster"] = dict(members=seed["members"], member_ids=seed["member_ids"])
            else:
                out["class_aware_note"] = (f"--class-aware: confirmed token (id {tau}) is not in any "
                                           f"flagged cluster; scrubbing the single confirmed token only.")
        cprompts = _prompts(a.collateral_prompts) if a.collateral_prompts else None
        if cprompts:                                          # measure YOUR-prompt behaviour BEFORE the in-place edit
            am0c, nll0c = argmax_allpos(student, tok, cprompts), mean_nll(student, tok, cprompts)
        ok, rep = scrub(student, tok, taus, prompts, k=a.k)
        out["scrub"] = rep
        if cprompts:                                          # ... and after (reflects the final state; rollback shows no change)
            am1c, nll1c = argmax_allpos(student, tok, cprompts), mean_nll(student, tok, cprompts)
            agree = (am0c == am1c).float().mean().item() if am0c.numel() else 1.0
            out["collateral_review"] = dict(
                prompts_file=a.collateral_prompts, n_prompts=len(cprompts), n_positions=int(am0c.numel()),
                top1_agreement=round(agree, 4), positions_changed=int((am0c != am1c).sum().item()),
                ppl_before=round(math.exp(nll0c), 3), ppl_after=round(math.exp(nll1c), 3),
                note=("review evidence on YOUR prompts, NOT a preservation guarantee; the self-check covers "
                      "neutral prompts only -- this shows whether the edit moved next-token behaviour on your use-case."))
        if a.out:
            import os
            os.makedirs(a.out, exist_ok=True)
            if ok:
                _od = getattr(student.config, "torch_dtype", None)   # persist in the checkpoint's
                if isinstance(_od, str):                              # original dtype, not the fp32
                    _od = getattr(t, _od, None)                      # we load for analysis
                if isinstance(_od, t.dtype) and _od != t.float32:
                    student.to(_od); out["saved_dtype"] = str(_od)
                student.save_pretrained(a.out); tok.save_pretrained(a.out); out["saved_model"] = a.out
            with open(os.path.join(a.out, "report.json"), "w") as f:
                json.dump(out, f, indent=2)
        elif ok:
            print("warning: scrub succeeded in memory but --out was not given, so the edited model was "
                  "NOT saved (no persistent effect). Re-run with --out DIR to write it.", file=sys.stderr)
            out["saved_model"] = None
        out["status"] = ("ok" if ok else
                         "unsupported_arch" if rep.get("status") == "refused" else "scrub_failed")
        _emit(out, a.json)
        sys.exit(0 if ok else 2)


def _tool_version():
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("distill-lint")
        except PackageNotFoundError:
            return __version__
    except Exception:
        return __version__


def _model_meta(model_id, model):
    """Which revision governed this load: the SHA we pin (model_revisions.REVISIONS) and, if transformers
    recorded it, the commit actually loaded (config._commit_hash). A verdict you can't trace to a SHA is
    worth less as a CI/audit artifact."""
    try:
        import model_revisions as _mr
        pinned = getattr(_mr, "REVISIONS", {}).get(str(model_id))
    except Exception:
        pinned = None
    loaded = getattr(getattr(model, "config", None), "_commit_hash", None)
    return {"id": str(model_id), "pinned": pinned or "UNPINNED", "loaded_commit": loaded}


def _run_meta(a, student, base, prompts):
    """Self-describing provenance for the verdict: tool version, the revision governing each load, the
    verdict-governing params (read from the live scan/classify defaults so they can't drift from the
    code), and the prompt-set identity -- so the JSON artifact is traceable to its inputs. No wall-clock
    timestamp: keeps committed runs/*.json diff-stable; add one here if you want it."""
    sp, cp = inspect.signature(scan).parameters, inspect.signature(classify).parameters
    src = getattr(a, "prompts", None) or "default"
    psha = hashlib.sha1("\n".join(prompts).encode("utf-8")).hexdigest()[:8] if prompts else None
    return {
        "tool_version": _tool_version(),
        "models": {"student": _model_meta(a.student, student), "base": _model_meta(a.base, base)},
        "params": {"z_thresh": sp["z_thresh"].default, "nbins": sp["nbins"].default,
                   "collapse_frac": cp["collapse_frac"].default,
                   "k": getattr(a, "k", None), "topk": getattr(a, "topk", None)},
        "prompts": {"source": src, "n": len(prompts) if prompts else 0, "sha8": psha},
    }


def _write_report(out, path):
    L = [f"# distill-lint report — `{out.get('command','')}`", "",
         f"- **student:** `{out.get('student','')}`",
         f"- **base:** `{out.get('base','')}`"]
    if out.get("status"):
        L.append(f"- **status:** `{out['status']}`  (JSON `status` is authoritative; the exit code is coarse)")
    _m = out.get("meta") or {}
    if _m:
        _sm = (_m.get("models") or {}).get("student", {})
        _bm = (_m.get("models") or {}).get("base", {})
        L.append(f"- **tool_version:** `{_m.get('tool_version','')}`  ·  "
                 f"**student rev:** `{_sm.get('pinned','?')}`  ·  **base rev:** `{_bm.get('pinned','?')}`")
    c = out.get("classification")
    if c:
        L += [f"- **token:** `{c.get('token')}`",
              f"- **verdict:** {c.get('verdict')}",
              f"- **residual_fraction:** {c.get('residual_fraction')}  (≤0.5 ⇒ collapses under orthogonalization ⇒ readout-sensitive)"]
    s = out.get("scrub")
    if s:
        L += [f"- **scrub status:** {s.get('status')}",
              f"- **self_check_passed:** {s.get('self_check_passed')}",
              f"- **perplexity_ratio** (vs the student *before* the edit): {s.get('perplexity_ratio')}",
              f"- **collateral_top1_agreement** (neutral prompts): {s.get('collateral_top1_agreement')}"]
        fp = s.get("collateral_footprint") or {}
        if fp:
            bm = fp.get("basis_movers") or []
            top = ", ".join(f"`{m['token']}` ΔP {m['dP']:+.4f}" for m in bm[:3]) or "none"
            L.append(f"- **collateral footprint:** max |ΔP| off target {fp.get('max_abs_dP_off_target')}; "
                     f"top neighbour-basis movers: {top}  (the co-directional mass scrub redistributes — review it)")
    cr = out.get("collateral_review")
    if cr:
        L.append(f"- **collateral review** (your prompts, `{cr.get('prompts_file')}`): top-1 agreement "
                 f"{cr.get('top1_agreement')} over {cr.get('n_positions')} positions, "
                 f"ppl {cr.get('ppl_before')} → {cr.get('ppl_after')} — review evidence, not a guarantee")
    if "summary" in out:
        L.append(f"- **watchlist summary:** `{json.dumps(out['summary'])}`")
    if "checks" in out:
        L += ["", "| check | ok | detail |", "|---|:--:|---|"]
        for ck in out["checks"]:
            L.append(f"| {ck['check']} | {'✓' if ck['ok'] else '✗'} | {ck['detail']} |")
    rc = out.get("reachability")
    if rc:
        L += ["", "**Reachability on this pair:**",
              f"- **calibration:** {'AVAILABLE' if rc['calibration'].get('available') else 'UNAVAILABLE'} — {rc['calibration'].get('detail','')}",
              f"- **scrub:** would {rc['scrub'].get('would')} — {rc['scrub'].get('detail','')}",
              f"- **watchlist coverage:** {rc['watchlist_coverage'].get('detail','')}"]
    if "result" in out:
        L += ["", f"**Result:** {out['result']}"]
    L += ["", "---", "_Scope / non-claim: vocabulary-channel QA only. This does **not** test body-carried "
          "conditional policies or backdoors, and does not certify the model safe, unbiased, or uncensored. "
          "A flag is a candidate, not a verdict; a clean run is not a certificate._"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _emit(out, json_arg):
    """json_arg: None -> human stdout | '-' -> JSON to stdout | <path> -> JSON to FILE (+ human stdout).
    If REPORT_PATH (--report FILE) is set, also write a short Markdown QA report there."""
    if REPORT_PATH:
        _write_report(out, REPORT_PATH)
        print(f"wrote report -> {REPORT_PATH}", file=sys.stderr)
    if json_arg and json_arg != "-":
        with open(json_arg, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote json -> {json_arg}", file=sys.stderr)
    if json_arg == "-":
        print(json.dumps(out, indent=2))
    else:
        print(f"\n=== distill-lint: {out['command']} ===")
        for k, v in out.items():
            if k in ("command", "boundary"):
                continue
            print(f"{k}: {json.dumps(v, indent=2) if isinstance(v,(list,dict)) else v}")
        print("\n" + out["boundary"])


if __name__ == "__main__":
    main()
