<!-- SPDX-License-Identifier: Apache-2.0 -->
# Security policy

`distill-lint` is a **defensive** auditing tool. It detects, classifies, and removes
vocabulary-channel single-token leakage in already-trained distilled models. It installs no
trait and ships no attack: the runnable trait-installation construction studied in the
accompanying paper is deliberately withheld (see [`NOTICE`](NOTICE)).

## Scope and non-claims

`distill-lint` is vocabulary-channel QA **only**. It does not detect or repair body-carried
or trigger-conditional policies or backdoors, and it does not certify a model safe, unbiased,
or uncensored. A flag is a candidate to investigate, not a verdict. The full boundary is in the README ("Scope: what this can and cannot do").

## Reporting

Please report the following **privately** — via GitHub's *Report a vulnerability* (Security →
Advisories) — rather than in a public issue, so a fix or a documented bound can be prepared first:

- a **defect** in the tool: a crash, an incorrect verdict, or an edit the post-edit self-check
  and the architecture guard failed to catch;
- a **method-level evasion**: a construction that hides a single-token trait below the scan, or
  routes it so `classify` or `scrub` mishandle it.

We aim to acknowledge reports promptly and to credit reporters who wish to be named. Findings
that establish a new bound on what the tool can and cannot catch are welcome — honest scope is
the point of the project.
