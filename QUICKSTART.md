# Quickstart (~5 minutes, CPU is fine)

`distill-lint` is a **carrier probe + targeted readout edit** for confirmed or suspected
vocabulary-channel token leakage. It is *not* a model-safety scanner — a flag is a candidate, not a verdict.

## Which command for my situation?
| Situation | Command |
|---|---|
| Is the tool installed & working? | `selftest` (zero-setup wiring check) |
| I suspect one token | `resolve-token` → `classify` |
| I have a watchlist and want a CI gate | `probe-list` (`--fail-on` picks severity) |
| I don't know what leaked | `scan --class-aware`, inspect, then `classify` |
| I want a calibrated p-value (real FPR), not just a flag | a reference null for your base auto-loads; else `calibrate` one from clean placebos and pass `--null` |
| I've confirmed a token is unwanted | `scrub --confirm-unwanted-token … --collateral-prompts …` |
| Is editing safe on this architecture? (and what's the base?) | `doctor` (omit `--base` to auto-detect it) |

## 0. Install + wiring check (no model of your own needed)
```bash
pip install -r requirements.txt
python qa.py selftest   # plants a by-fiat leak in pythia-410m, then detect -> classify -> scrub it (real FAIL -> fixable -> residual~0), next to a clean control
```
`selftest` is the zero-setup proof the tool is installed and **acting on signal** — a first run on a clean
model otherwise shows only "not elevated" panels, which look identical to a no-op. It is loudly labelled a
**wiring check, not a validation** of the method. (`python _smoke.py` is the dev equivalent on pythia-70m.)
The real subliminal-transfer numbers (E1-E5) are committed results regenerated from the reproducibility
artifact — see `evidence/` and the README.

## 1. Get the token right
Leading space matters; multi-token strings are out of scope.
```bash
python qa.py resolve-token --base EleutherAI/pythia-410m --text Google
# -> shows " Google" is single-token (use that form), "Google" may be multi-token
```

## 2. Preflight the pair
```bash
python qa.py doctor --student ./my_student --base EleutherAI/pythia-410m --tokens watchlists/brands.txt --report doctor.md
# checks tokenizer/vocab match, tied-head, architecture, dtype, memory, revision pinning, and whether scrub would run
# also prints a REACHABILITY report: is calibration available for this base, will scrub RUN or REFUSE, and
# (with --tokens) how many of your watchlist entries have a single-token handle the gate can act on
# omit --base to auto-detect a candidate from the student's metadata (adapter_config.json / config) and gate it
```

## 3. Probe a token you suspect (drift-immune)
```bash
python qa.py classify --student ./my_student --base EleutherAI/pythia-410m --token " Google"
```
**What the verdict means, and what to do next:**
- `vocabulary-carried (fixable)` → removable from the readout; decide whether it is *genuinely unwanted* before editing.
- `escalate` → not readout-sensitive; do **not** scrub — investigate teacher / data / prompts / distillation provenance.
- *not elevated* → no action for this token.

## 4. Gate CI on a watchlist (the recommended CI path)
```bash
python qa.py probe-list --student ./my_student --base EleutherAI/pythia-410m \
  --tokens watchlists/brands.txt --json probe.json --report probe.md
# exit 1 if a watchlisted token is an ACTIONABLE leak. --fail-on (default `any`) controls which severities
# fail: fixable (readout-sensitive) OR escalate (body-carried). If a reference null ships for the base it
# is AUTO-LOADED and the gate is calibrated (a real false-positive rate); else escalate downgrades to warn
# (add --require-calibration to refuse running an uncalibrated gate). --fail-on fixable restores the old
# "only fail on a removable token" behaviour.
```
Drop-in workflow (selftest → doctor → probe-list): `ci/github-actions.yml`. Reference lists: `watchlists/`.

## 4b. (Optional) Calibrated detection with a real false-positive rate
A single `(student, base)` pair can't calibrate detection (benign drift flags too). Build a null from
**K clean placebo students** (teacher == base; they install no trait), then score against it:
```bash
python qa.py calibrate --base EleutherAI/pythia-410m --placebos clean1 clean2 ... cleanK --out null.json
python qa.py scan       --student ./my_student --base EleutherAI/pythia-410m --null null.json   # multiplicity-corrected p
python qa.py probe-list --student ./my_student --base EleutherAI/pythia-410m --tokens watchlists/brands.txt --null null.json
```
No placebos of your own? Use a shipped reference null. **You usually don't even pass `--null`:** if one
ships for your base (registered in `nulls/INDEX.json`) and your prompt set matches, `scan`/`probe-list`
**auto-load it** and print a `note:` saying which file was used (`--no-auto-null` to disable). A null is
valid only for its exact base + prompt set, so a mismatched `--null` is **refused** (exit 2), not silently
used — `--force-null` overrides if you know it's still valid. Generate a null for another base with
`make_reference_null.py` (in the OSF reproducibility artifact), then add it to `INDEX.json`. Scope: clean placebos require
teacher == base — a pipeline owner can make them; a post-hoc auditor uses the reference null.

## 5. Remove a CONFIRMED-unwanted token (and check it on YOUR prompts)
```bash
python qa.py scrub --student ./my_student --base EleutherAI/pythia-410m \
  --token " Google" --confirm-unwanted-token " Google" \
  --collateral-prompts my_eval_prompts.txt \
  --out ./scrubbed --report scrub.md
```
`--confirm-unwanted-token` is required (readout-sensitive ≠ should-remove). `--collateral-prompts` reports
the next-token behaviour delta on *your* prompts (review evidence, not a preservation guarantee — the
built-in self-check only covers neutral prompts).

## What to do with the result
- `probe-list` exit `0` → no watchlisted token reached the `--fail-on` gate (pass); review any `warn`s.
- `probe-list` exit `1` → a CI failure for your watchlist policy (a watchlisted leak at/above `--fail-on`); review the report.
- `probe-list` exit `2` (status `refused`) → a `--null`/`--require-calibration` problem, not a model verdict; fix the null and re-run.
- `classify` → `fixable`: decide whether the token is genuinely unwanted before editing. → `escalate`: don't scrub; investigate provenance.
- Running `scrub`: always pass `--collateral-prompts` from your real eval set (the self-check only sees neutral prompts).

## Limits
`distill-lint` only addresses vocabulary-carried token leakage. It does not detect or fix body-carried
behaviours, trigger-conditional policies, or general backdoors.
