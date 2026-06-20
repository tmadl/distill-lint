# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Tamas Madl
"""Self-contained smoke test for the distill_lint tool.

Installs a synthetic *vocabulary-carried* elevation by DIRECTLY editing the unembedding -- no teacher,
no distillation, no import from the reproducibility repo -- then runs the shipped scan -> classify ->
scrub path on it. This only checks that the tool runs and behaves sanely; the real subliminal-transfer
evidence (which does require masked distillation) lives in evidence/*_RESULT.md, reproduced from the
separately-released research artifact.

The fixture (`evidence/leak_fixture.py`, shared with `measure_scrub.py --demo`) is the trivial inverse
of `scrub`: it co-elevates tau and its top-k unembedding neighbours along a shared direction, so tau's
lift lives in the neighbour span and orthogonalizing it away (what classify probes and scrub does)
removes it. The magnitude is LOGIT-TARGETED, not a raw coefficient (which scales with ||hidden|| and
overflows on some models -- see leak_fixture.py), and the fixture confirms its achieved lift against a
real forward pass. Editing a row to raise a token's probability is day-one linear algebra and a plainly
detectable (non-covert) edit -- it carries no covert-installation recipe.

Run:  python _smoke.py        (downloads EleutherAI/pythia-70m, ~150 MB; CPU is fine)
"""
import copy
import math
import os
import sys

import qa
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence"))
from leak_fixture import plant_vocab_leak  # noqa: E402  (shared fixture; lives under evidence/)

MODEL = "EleutherAI/pythia-70m"
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL).float().to(qa.DEV).eval()
student = AutoModelForCausalLM.from_pretrained(MODEL).float().to(qa.DEV).eval()

tau = tok.encode(" seven", add_special_tokens=False)[0]
base_p = qa.mean_next_token_p(base, tok, qa.DEFAULT_PROMPTS)[tau].item()

# --- install a vocabulary-carried elevation by fiat (no teacher / no masked-distillation loop) ---
# tau is co-elevated MORE strongly than its neighbour cloud, so it tops the scan; logit-targeted
# magnitude keeps the "leaked" student a sane model (perplexity ~1.4x base here, not e^3900).
_tau, nb, fix_info = plant_vocab_leak(student, tok, " seven", qa.DEFAULT_PROMPTS,
                                      logit_lift_tau=9.0, logit_lift_nbr=4.0)
assert _tau == tau
Wn = qa.unit_rows(student.get_output_embeddings().weight.detach())
stu_p = qa.mean_next_token_p(student, tok, qa.DEFAULT_PROMPTS)[tau].item()
base_ppl = math.exp(qa.mean_nll(base, tok, qa.DEFAULT_PROMPTS))
leaked_ppl = math.exp(qa.mean_nll(student, tok, qa.DEFAULT_PROMPTS))   # the "leaked" student, pre-scrub
print(f"fixture: base P(' seven')={base_p:.2e}  ->  student P(' seven')={stu_p:.3f}  "
      f"(target logit lift {fix_info['target_logit_lift']}, achieved {fix_info['achieved_logit_lift']})")
print(f"         leaked perplexity {leaked_ppl:.1f} vs base {base_ppl:.1f} "
      f"({leaked_ppl/base_ppl:.2f}x) -- a sane 'leaked' model, so 'scrub preserved perplexity' is meaningful")

print("\n--- arch_guard ---"); guard = qa.arch_guard(student); print(guard)
print("\n--- scan ---")
flags = qa.scan(student, base, tok, topk=8)
for f in flags:
    print(" ", f)
print("\n--- class-aware clusters ---")
clusters = qa.cluster_flagged(flags, Wn, tok); print(clusters)
cls_members = clusters[0]["member_ids"] if clusters else [tau]
# (a) class-aware multi-member path runs + is decisive on a deepcopy (pass OR rollback both valid)
sp = copy.deepcopy(student); ok_path, rep_path = qa.scrub(sp, tok, cls_members); del sp
# (b) regression for the all_dropped p0>1e-3 fix: a floor-level member must not block an otherwise-good scrub
p0n = qa.mean_next_token_p(student, tok, qa.DEFAULT_PROMPTS)
floor_tau = next(i for i in range(p0n.shape[0]) if p0n[i].item() <= 1e-3 and i != tau)
sf = copy.deepcopy(student); ok_fix, rep_fix = qa.scrub(sf, tok, [tau, floor_tau]); del sf
print("\n--- classify (tau) ---")
cls = qa.classify(student, base, tok, tau); print(cls)
print("\n--- scrub (tau) ---")
ok, rep = qa.scrub(student, tok, [tau])
print("ok:", ok)
print(rep)

# --- pass/fail gate: assert the behaviour, so a regression FAILS (not just prints) ---
checks = {
    "fixture installed (student P >> base)":   stu_p > 50 * base_p and stu_p > 1e-3,
    "fixture self-check: achieved logit lift ~ target": fix_info["lift_ok"] is True,
    "fixture leaves a SANE leaked model (ppl < 5x base, not e^3900)": leaked_ppl < 5 * base_ppl,
    "arch_guard reports safe":                 guard[0] is True,
    "scan returns >=1 flag":                   len(flags) > 0,
    "classify -> vocabulary-carried (fixable)": cls["verdict"].startswith("vocabulary"),
    "scrub ok":                                ok is True,
    "scrub drove P(tau) -> ~0 (<1e-3)":        rep["p_tau_before_after"][" seven"][1] < 1e-3,
    "scrub self-check passed":                 rep["self_check_passed"] is True,
    "scrub preserved top-1 (>=0.98)":          rep["collateral_top1_agreement"] >= 0.98,
    "scrub preserved perplexity (ratio<=1.05)": rep["perplexity_ratio"] <= 1.05,
    "class-aware: cluster >1 member": len(cls_members) > 1,
    "class-aware path evaluated all members": all(tok.decode([m]) in rep_path["p_tau_before_after"] for m in cls_members),
    "class-aware path self-check decisive": isinstance(rep_path.get("self_check_passed"), bool),
    "fix: [tau,floor] not rolled back by floor": ok_fix is True,
    "fix: tau still halved with floor present": rep_fix["p_tau_before_after"][" seven"][1] < 0.5 * rep_fix["p_tau_before_after"][" seven"][0],
}
print("\n--- gate ---")
for name, passed in checks.items():
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
failed = [n for n, p in checks.items() if not p]
if failed:
    raise SystemExit(f"\nSMOKE FAILED: {failed}")
print("\nSMOKE PASSED")
