# Real-model triage demo

Run `distill_lint` on **real released distilled models** to show, concretely, what the tool buys a
practitioner: a forward-pass-only triage of which vocabulary-channel entity/topic preferences a model
carries, which are removable, and what the scan cannot see.

Two safety framings, same entity-agnostic channel:
- **covert promotion** — `token_panels/frontier_ai_brands.txt` (undisclosed product/brand preference)
- **covert source-steering** — `token_panels/news_sources.txt` (steering toward an information outlet)
- **null** — `token_panels/neutral_control.txt` (should stay flat; sets the false-positive floor)

## Scope (read first)
- The **classify** verdict (orthogonalize-as-probe: does P(token) collapse when its output row is
  orthogonalized against neighbours?) is **drift-immune and ground-truth-free** — the check to lead with on real models.
- The **scan** on a single (student, base) pair is recall-oriented **triage**, not a calibrated alarm
  (clean-teacher students drift in entanglement-shaped ways; calibrated detection needs the K-placebo
  procedure in the separately-released reproducibility artifact). On a real model, scan output is a **ranked candidate list** — never
  caption a flag as "model M prefers entity E".
- Report **aggregate / anonymized** ("across N models, K candidates, F classify-fixable") with
  non-causal / no-intent language. No per-named-model "prefers entity" leaderboard.
- The tool does **NOT** certify a model as unbiased / safe / uncensored.

## The classify panel
For each token in a panel, `classify` reports whether an elevation (if any) is vocabulary-fixable
(collapses under orthogonalization, `residual_fraction` low) or escalates (no single-token handle →
body-carried). Figure = dumbbell/heatmap of `student_p → probe_p` per token, coloured by verdict, with
the neutral-control panel overlaid as the null. This is the check to run before trusting an unfamiliar checkpoint — mechanistic, and needs no ground truth or placebos.

## Reading the committed runs
`runs/*.json` are illustrative outputs that correspond to the pinned SHAs in `model_zoo.yaml` (the
weights are unchanged since the run). Read them with the scope above:
- `…__scan.json` surfaces the distillation **register** (e.g. ` work`, ` math`, ` assistant`) — the
  benign signature of distilling a reasoning model. An investigate-trigger, **not** a finding of malice.
- the `…__classify_*` panels come back with nothing elevated (`fixable: 0`): that is the **expected
  null** on these panels, not a tool failure. Many entity tokens are multi-token and are skipped.
- scrub **runs** on this 1.5B pair: the base (Qwen2.5-Math) ties its embeddings, but the R1 distill
  **untied** the head, and `arch_guard` checks the *student* — so the edit is safe (worked example below).
  The arch-guard *refusal* is demoed instead on a genuinely tied pair, `Qwen/Qwen3-0.6B` /
  `Qwen/Qwen3-0.6B-Base`, where `doctor` reports `scrub REFUSE` (both `tie_word_embeddings=true`).

## Worked scrub on a real bf16 distillation (the 1.5B pair)
End-to-end `scan → classify → scrub` on `DeepSeek-R1-Distill-Qwen-1.5B` vs `Qwen2.5-Math-1.5B` (bf16
checkpoint; the edit is computed in fp32 and saved back in bf16):

| step | result |
|---|---|
| `classify " assistant"` | vocabulary-carried (fixable), `residual_fraction` 0.003 |
| `scrub` | scrubbed, self-check passed |
| P(` assistant`) | 0.01052 → 0.000052 (~200× down) |
| perplexity ratio | 0.9999 (no degradation) |
| collateral top-1, neutral prompts | 1.000 (zero argmax changes) |
| max \|ΔP\| off-target | 0.0035 |

The collateral footprint lands exactly on the ` assistant` entanglement cloud — `助手` (Chinese for
"assistant"), ` Assistant`, ` assistants`, `Assistant`. That is the mechanism: `scrub` edits the
*direction* and moves the token's unembedding neighbours, including the cross-lingual one, rather than
matching a string.

**Note.** ` assistant` is a **benign** chat-register token, scrubbed here only to exercise the
mechanics on a real model. In real use you would `--confirm-unwanted-token` only a token you have
independently judged should go: `fixable` means *removable*, never *should-remove*.

## Tool status
Shipped in `qa.py`: (1) a hard **tokenizer-vocab equality assert** between student and base; (2) exit-0
renamed from "clean" → "no flag (triage only)" in CLI + JSON, with the BOUNDARY block printed every
run; (3) explicit "does NOT certify unbiased/safe/uncensored" in output and README; (4) the classify
`--panel` driver computes one student/base forward and reuses it across the whole panel. Model SHAs in
`model_zoo.yaml` are pinned (and mirrored in `../../model_revisions.py`, so the tool loads them pinned);
the committed `runs/*.json` correspond to those snapshots. Regenerate the runs only if you bump a pin.

## Commands (inference/forward-pass only)
```
# CPU-runnable real-released-model verdict (no GPU; ~minutes, a few GB RAM). classify is the
# drift-immune classify -- a real fixable/escalate/not-elevated on released weights, line one:
python ../../qa.py classify --student allenai/OLMo-2-0425-1B-Instruct \
                            --base    allenai/OLMo-2-0425-1B \
                            --panel token_panels/frontier_ai_brands.txt --json
# (an all-"not elevated" panel here is the EXPECTED null on these weights, not a tool failure -- see above)

# the larger DeepSeek/Qwen pairs below want a GPU (or patience on CPU):
# quickstart (<10 min): smallest pair. scan is vocabulary-wide triage:
python ../../qa.py scan     --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                            --base    Qwen/Qwen2.5-Math-1.5B --json
# classify a whole token panel (cached student/base forward, one verdict per token):
python ../../qa.py classify --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                            --base    Qwen/Qwen2.5-Math-1.5B \
                            --panel token_panels/frontier_ai_brands.txt --json
python ../../qa.py classify ... --panel token_panels/news_sources.txt --json
python ../../qa.py classify ... --panel token_panels/neutral_control.txt --json   # the null

# scrub demo on an UNTIED real model -- the cached 1.5B pair WORKS (student head untied; ppl ratio 0.9999).
# (" assistant" is a benign register token shown for mechanics -- only confirm a token you judged unwanted.)
python ../../qa.py scrub    --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                            --base    Qwen/Qwen2.5-Math-1.5B \
                            --token " assistant" --confirm-unwanted-token " assistant" --out /tmp/scrubbed --json
#   (larger untied alternatives: the 7B / 8B pairs in model_zoo.yaml)

# arch-guard REFUSAL demo on a genuinely tied pair (both tie_word_embeddings=true) -> scrub REFUSE (exit 2):
python ../../qa.py doctor   --student Qwen/Qwen3-0.6B --base Qwen/Qwen3-0.6B-Base
```
The tokenizer-vocab guard aborts if the base is wrong; `--panel` skips multi-token entries per
tokenizer (many entity tokens are multi-token — expected, not a failure) and reuses one student/base
forward across the panel. Run `python make_figures.py` to (re)generate `figures/` from `runs/*.json`.
