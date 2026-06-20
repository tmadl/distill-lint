# Watchlists for `probe-list`

A watchlist is the recommended CI input: the tokens **you** care about. `probe-list` classifies each and
gates the build —

- **elevated AND readout-sensitive** (vocabulary-carried) → **CI FAIL** (a removable leak you flagged)
- elevated but **escalate** (body-carried) → **warn** (out of scope, not a failure)
- **not elevated** over base → pass

```bash
python qa.py probe-list --student ./student --base <its-base> --tokens watchlists/brands.txt --json report.json
```

## What ships here
| file | what | safety framing |
|---|---|---|
| `brands.txt` | commercial brand tokens | covert promotion / advertising |
| `frontier_ai.txt` | model/vendor brand tokens | covert self/vendor promotion |
| `news_sources.txt` | information-outlet tokens | covert source-steering |
| `neutral_control.txt` | tokens that should stay flat | sanity check — if these FAIL, recheck the pairing (`qa.py doctor`) |

## Rules
- **Leading space is significant** (BPE): ` Google` ≠ `Google`. Each line must be a **single token** for the
  `(student, base)` tokenizer. Find the right form with `qa.py resolve-token --base <base> --text Google`.
- Multi-token entities are out of scope (only the first subword has a single-row handle).
- Put **your** entities here — competitor names, product labels, sources you must not steer toward.

## Building a PII / profanity watchlist (without shipping slurs)
We deliberately do **not** ship a slur/PII list. To build one from a published source:
1. Take an off-the-shelf list (e.g. LDNOOBW, `https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words`, CC-BY-4.0).
2. Keep only entries that are **single tokens** for your tokenizer (`resolve-token` each, or script it).
3. Save them, one per line with leading spaces, as your private watchlist — do not commit it if it contains
   slurs. `probe-list --tokens your_pii_watchlist.txt` then gates exactly those.

This is the threat model from `evidence/e4_scrub_efficacy_RESULT.md`: a real string filter does not stop the
leak, but a watchlist probe **detects** it and `scrub` removes it from the readout.
