# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Tamas Madl
"""Unit tests for the pure, silently-breakable internals of qa.py.

These cover the parts a CI-gating tool most needs covered and that have no end-to-end smoke coverage:
the frequency-matched z-scoring (trailing partial bin, zero-variance bin), the tokenizer overlap/
superset/conflict logic, the QR orthogonalization numerics, neighbour selection, and the architecture
guard. None loads a model -- pure torch, runs anywhere `pip install -e .[test]` works.

Run:  pytest -q          (from the distill_lint/ directory)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qa  # noqa: E402


# --------------------------------------------------------------------------- unit_rows
def test_unit_rows_normalizes():
    Wn = qa.unit_rows(torch.randn(7, 5))
    assert torch.allclose(Wn.norm(dim=1), torch.ones(7), atol=1e-5)


def test_unit_rows_zero_row_is_finite():
    Wn = qa.unit_rows(torch.zeros(3, 4))           # clamp_min guards div-by-zero
    assert torch.isfinite(Wn).all()


# --------------------------------------------------------------------------- neighbours
def test_neighbours_excludes_tau_and_finds_near_duplicate():
    W = torch.eye(6)
    W[1] = W[0] + 0.01 * torch.randn(6)            # row 1 ~ row 0
    nb = qa.neighbours(qa.unit_rows(W), 0, k=2)
    assert 0 not in nb and 1 in nb and len(nb) == 2


# --------------------------------------------------------------------------- orthogonalize_row
def test_orthogonalize_removes_projection_onto_basis():
    torch.manual_seed(0)
    W = torch.randn(8, 5)
    basis = [1, 2, 3]
    qa.orthogonalize_row(W, 0, basis)
    Q, _ = torch.linalg.qr(W[basis].float().t())
    assert (Q @ (Q.t() @ W[0].float())).norm() < 1e-4


def test_orthogonalize_is_idempotent():
    torch.manual_seed(1)
    W = torch.randn(8, 5)
    qa.orthogonalize_row(W, 0, [1, 2, 3])
    r1 = W[0].clone()
    qa.orthogonalize_row(W, 0, [1, 2, 3])
    assert torch.allclose(r1, W[0], atol=1e-5)


def test_orthogonalize_leaves_already_orthogonal_row_unchanged():
    W = torch.zeros(4, 3)
    W[0] = torch.tensor([0.0, 0.0, 1.0])           # tau along z
    W[1] = torch.tensor([1.0, 0.0, 0.0])           # basis along x
    before = W[0].clone()
    qa.orthogonalize_row(W, 0, [1])
    assert torch.allclose(W[0], before, atol=1e-5)


# --------------------------------------------------------------------------- _freqmatched_z
def test_z_constant_bin_is_zero_not_nan():
    z = qa._freqmatched_z(torch.ones(40), torch.randn(40), nbins=4)
    assert torch.isfinite(z).all() and z.abs().max() < 1e-3


def test_z_flags_outlier_above_threshold():
    # large bins (500 tokens / 5 bins = 100 each) so one outlier doesn't inflate its own bin's std,
    # mirroring the real ~50k-vocab / 20-bin regime.
    torch.manual_seed(2)
    lift = 0.001 * torch.randn(500)
    lift[250] = 1.0
    z = qa._freqmatched_z(lift, torch.linspace(-10, 0, 500), nbins=5)
    assert z[250] > 4.0 and z[250] == z.max()


def test_z_handles_trailing_partial_bin():
    z = qa._freqmatched_z(torch.randn(103), torch.randn(103), nbins=10)  # 103 % 10 != 0
    assert z.shape == (103,) and torch.isfinite(z).all()


def test_z_more_bins_than_tokens():
    z = qa._freqmatched_z(torch.randn(5), torch.randn(5), nbins=20)      # binsz clamps to 1
    assert torch.isfinite(z).all()


# --------------------------------------------------------------------------- _tok_compat
def test_tok_compat_identical():
    v = {"a": 0, "b": 1, "c": 2}
    ok, conf, ov = qa._tok_compat(v, v)
    assert ok and conf == 0 and ov == 1.0


def test_tok_compat_conflict_rejected():
    ok, conf, _ = qa._tok_compat({"a": 0, "b": 1}, {"a": 0, "b": 9})    # 'b' -> different id
    assert not ok and conf == 1


def test_tok_compat_conflict_free_superset_ok():
    bv = {f"t{i}": i for i in range(100)}
    sv = dict(bv, **{"<extra>": 100})                                   # one added special token
    ok, conf, ov = qa._tok_compat(sv, bv)
    assert ok and conf == 0 and ov >= 0.98


def test_tok_compat_low_overlap_rejected():
    sv = {f"a{i}": i for i in range(100)}
    bv = {f"b{i}": i for i in range(100)}                               # disjoint
    assert not qa._tok_compat(sv, bv)[0]


# --------------------------------------------------------------------------- arch_guard / _is_tied
class _Cfg:
    def __init__(self, model_type="gpt_neox", tie=False):
        self.model_type, self.tie_word_embeddings = model_type, tie


class _Emb:
    def __init__(self, w):
        self.weight = w


class _FakeModel:
    """Minimal duck-typed stand-in: .config + get_input/output_embeddings().weight (no transformers)."""
    def __init__(self, model_type="gpt_neox", tie=False, shared_storage=False):
        self.config = _Cfg(model_type, tie)
        w_in = torch.randn(10, 4)
        self._ie = _Emb(w_in)
        self._oe = _Emb(w_in if shared_storage else torch.randn(10, 4))

    def get_input_embeddings(self):
        return self._ie

    def get_output_embeddings(self):
        return self._oe


def test_arch_guard_untied_ok():
    assert qa.arch_guard(_FakeModel())[0]


def test_arch_guard_tied_flag_refused():
    ok, reason = qa.arch_guard(_FakeModel(tie=True))
    assert not ok and "tied" in reason.lower()


def test_arch_guard_shared_storage_refused():
    assert not qa.arch_guard(_FakeModel(shared_storage=True))[0]


def test_arch_guard_recurrent_refused():
    ok, reason = qa.arch_guard(_FakeModel(model_type="rwkv"))
    assert not ok and ("recurrent" in reason.lower() or "state-space" in reason.lower())


def test_is_tied():
    assert qa._is_tied(_FakeModel(tie=True))
    assert qa._is_tied(_FakeModel(shared_storage=True))
    assert not qa._is_tied(_FakeModel())


class _FakeTok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def test_select_class_cluster_uses_confirmed_token_not_top_lift():
    # cluster A = {0,1} (highest lift); cluster B = {2,3}. tau lives in B.
    # The historic bug scrubbed cluster_flagged(...)[0] == A regardless of the confirmed token.
    Wn = torch.zeros(4, 3)
    Wn[0] = torch.tensor([1.0, 0.0, 0.0]); Wn[1] = torch.tensor([1.0, 0.0, 0.0])
    Wn[2] = torch.tensor([0.0, 1.0, 0.0]); Wn[3] = torch.tensor([0.0, 1.0, 0.0])
    flags = [dict(token_id=0, lift=9.0), dict(token_id=1, lift=8.0),
             dict(token_id=2, lift=2.0), dict(token_id=3, lift=1.0)]
    members, seed = qa.select_class_cluster(flags, Wn, _FakeTok(), tau=2)
    assert seed is not None
    assert set(members) == {2, 3}           # tau's own cluster ...
    assert 0 not in members and 1 not in members   # ... never the unrelated top-lift class


def test_select_class_cluster_falls_back_to_single_token_when_unclustered():
    Wn = torch.zeros(2, 3)
    Wn[0] = torch.tensor([1.0, 0.0, 0.0]); Wn[1] = torch.tensor([1.0, 0.0, 0.0])
    flags = [dict(token_id=0, lift=9.0), dict(token_id=1, lift=8.0)]
    members, seed = qa.select_class_cluster(flags, Wn, _FakeTok(), tau=7)  # tau not flagged
    assert seed is None and members == [7]


def test_run_meta_self_describes_and_tracks_live_defaults():
    import argparse, inspect
    a = argparse.Namespace(student="S", base="B", prompts=None, k=7, topk=3)
    m = qa._run_meta(a, None, None, qa.DEFAULT_PROMPTS)
    assert m["tool_version"]                                  # version stamped
    assert set(m["models"]) == {"student", "base"}
    assert m["models"]["student"]["pinned"] == "UNPINNED"     # unknown id -> not pinned (truthful)
    assert m["params"]["k"] == 7 and m["params"]["topk"] == 3
    # params must track the LIVE scan/classify defaults (drift guard), not a hardcoded copy:
    assert m["params"]["z_thresh"] == inspect.signature(qa.scan).parameters["z_thresh"].default
    assert m["params"]["collapse_frac"] == inspect.signature(qa.classify).parameters["collapse_frac"].default
    assert m["prompts"]["n"] == len(qa.DEFAULT_PROMPTS) and len(m["prompts"]["sha8"]) == 8


# --------------------------------------------------------------------------- detect_base_candidates (doctor auto-detect)
class _CfgB:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _ModelB:
    def __init__(self, **cfg):
        self.config = _CfgB(**cfg)


def test_detect_base_from_adapter_config(tmp_path):
    import json
    d = tmp_path / "stu"; d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base-7b"}))
    (d / "config.json").write_text(json.dumps({"_name_or_path": str(d)}))   # self-ref, must be dropped
    cands = qa.detect_base_candidates(str(d), _ModelB())
    ids = [c for c, _ in cands]
    assert "org/base-7b" in ids and str(d) not in ids               # adapter base found; self-ref dropped


def test_detect_base_from_disk_config_name_or_path(tmp_path):
    import json
    d = tmp_path / "stu"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"_name_or_path": "EleutherAI/pythia-70m"}))
    cands = qa.detect_base_candidates(str(d), _ModelB())
    assert ("EleutherAI/pythia-70m", "config._name_or_path") in cands


def test_detect_base_empty_when_no_metadata(tmp_path):
    import json
    d = tmp_path / "stu"; d.mkdir()
    (d / "config.json").write_text(json.dumps({"_name_or_path": str(d)}))   # only a self-ref
    assert qa.detect_base_candidates(str(d), _ModelB()) == []


# --------------------------------------------------------------------------- K-placebo null p-values (calibrate)
def _toy_null(maxz_list, ztok=None, z_floor=2.0):
    recs = []
    for i, mz in enumerate(maxz_list):
        zb = {} if ztok is None else dict(ztok[i])
        recs.append(dict(label=f"p{i}", max_z=mz, z_by_token=zb))
    return dict(kind="distill-lint-null", version=1, base="B", K=len(recs),
                z_floor=z_floor, nbins=20, prompts_sha8="deadbeef", records=recs)


def test_null_pvalue_max_addone_and_monotone():
    null = _toy_null([3.0, 4.0, 5.0, 6.0])               # K=4
    # an unbeatable observation -> (1+0)/(4+1) = 0.2, never 0 (add-one honesty with finite K)
    assert abs(qa.null_pvalue_max(null, 99.0) - 0.2) < 1e-9
    # an observation below all placebos -> everyone exceeds it -> (1+4)/5 = 1.0
    assert abs(qa.null_pvalue_max(null, 1.0) - 1.0) < 1e-9
    # monotone non-increasing in the observed statistic
    assert qa.null_pvalue_max(null, 5.5) <= qa.null_pvalue_max(null, 4.5)


def test_null_pvalue_token_uses_floor_for_absent():
    # token 42 elevated only in placebo 0; absent (below floor) elsewhere -> read as floor (2.0)
    null = _toy_null([9, 9, 9], ztok=[{42: 8.0}, {}, {}], z_floor=2.0)
    # observed z=5 for token 42: only placebo 0 (8.0>=5) exceeds -> (1+1)/(3+1)=0.5
    assert abs(qa.null_pvalue_token(null, 42, 5.0) - 0.5) < 1e-9
    # observed z=1 (below floor): all three placebos (read as >=2.0) exceed -> (1+3)/4 = 1.0
    assert abs(qa.null_pvalue_token(null, 42, 1.0) - 1.0) < 1e-9


def test_load_null_refuses_on_base_and_prompt_mismatch(tmp_path):
    import json
    p = tmp_path / "null.json"
    p.write_text(json.dumps(_toy_null([3, 4])))          # base "B", prompts_sha8 "deadbeef"
    _null, warns, fatal = qa._load_null(str(p), base_id="OTHER", prompts=qa.DEFAULT_PROMPTS)
    assert any("base" in m for m in fatal)               # base mismatch is FATAL (refuse, not silent warn)
    assert any("prompt" in m.lower() for m in fatal)     # prompt-set mismatch is FATAL
    assert warns == []


def test_load_null_force_downgrades_fatal_to_warn(tmp_path):
    import json
    p = tmp_path / "null.json"
    p.write_text(json.dumps(_toy_null([3, 4])))
    _null, warns, fatal = qa._load_null(str(p), base_id="OTHER", prompts=qa.DEFAULT_PROMPTS, force=True)
    assert fatal == []                                   # --force-null proceeds...
    assert any("OVERRIDDEN" in w for w in warns)         # ...but loudly records what it overrode


def test_load_null_clean_when_matching(tmp_path):
    import json, hashlib
    n = _toy_null([3, 4])
    n["base"] = "EleutherAI/pythia-70m"
    n["prompts_sha8"] = hashlib.sha1("\n".join(qa.DEFAULT_PROMPTS).encode()).hexdigest()[:8]
    p = tmp_path / "null.json"; p.write_text(json.dumps(n))
    _null, warns, fatal = qa._load_null(str(p), base_id="EleutherAI/pythia-70m", prompts=qa.DEFAULT_PROMPTS)
    assert warns == [] and fatal == []


def _matching_null(maxz=[3, 4]):
    import hashlib
    n = _toy_null(maxz)
    n["base"] = "EleutherAI/pythia-70m"
    n["prompts_sha8"] = hashlib.sha1("\n".join(qa.DEFAULT_PROMPTS).encode()).hexdigest()[:8]
    return n


def test_load_null_refuses_missing_base_field(tmp_path):
    import json
    n = _matching_null(); del n["base"]                  # a baseless null must NOT auto-pass the base gate
    p = tmp_path / "n.json"; p.write_text(json.dumps(n))
    _null, warns, fatal = qa._load_null(str(p), base_id="anything", prompts=qa.DEFAULT_PROMPTS)
    assert any("no 'base'" in m for m in fatal)


def test_load_null_refuses_missing_prompts_sha(tmp_path):
    import json
    n = _matching_null(); del n["prompts_sha8"]
    p = tmp_path / "n.json"; p.write_text(json.dumps(n))
    _null, warns, fatal = qa._load_null(str(p), base_id="EleutherAI/pythia-70m", prompts=qa.DEFAULT_PROMPTS)
    assert any("prompts_sha8" in m for m in fatal)


def test_load_null_refuses_missing_records_and_force_cannot_override_structural(tmp_path):
    import json
    n = _matching_null(); del n["records"]               # malformed: would KeyError in p-value math
    p = tmp_path / "n.json"; p.write_text(json.dumps(n))
    _null, warns, fatal = qa._load_null(str(p), base_id="EleutherAI/pythia-70m",
                                        prompts=qa.DEFAULT_PROMPTS, force=True)
    assert any("records" in m for m in fatal)            # --force-null does NOT bypass a structural defect


def test_load_null_wrong_kind_not_overridable_by_force(tmp_path):
    import json
    n = _matching_null(); n["kind"] = "not-a-null"
    p = tmp_path / "n.json"; p.write_text(json.dumps(n))
    _null, warns, fatal = qa._load_null(str(p), base_id="EleutherAI/pythia-70m",
                                        prompts=qa.DEFAULT_PROMPTS, force=True)
    assert any("not a distill-lint null" in m for m in fatal)


def test_load_null_nbins_mismatch_fatal_but_forceable(tmp_path):
    import json
    n = _matching_null(); n["nbins"] = 13               # z is binning-dependent
    p = tmp_path / "n.json"; p.write_text(json.dumps(n))
    _null, warns, fatal = qa._load_null(str(p), base_id="EleutherAI/pythia-70m", prompts=qa.DEFAULT_PROMPTS)
    assert any("nbins" in m for m in fatal)
    _n2, w2, f2 = qa._load_null(str(p), base_id="EleutherAI/pythia-70m",
                                prompts=qa.DEFAULT_PROMPTS, force=True)   # provenance -> forceable
    assert f2 == [] and any("nbins" in w for w in w2)


# --------------------------------------------------------------------------- reference-null resolver + INDEX
def test_resolve_null_matches_shipped_pythia410m():
    path, note = qa._resolve_null_path("EleutherAI/pythia-410m", qa.DEFAULT_PROMPTS)
    assert path is not None and path.endswith("pythia-410m.json")
    assert "auto-loaded" in note


def test_resolve_null_prompt_mismatch_returns_none_with_note():
    path, note = qa._resolve_null_path("EleutherAI/pythia-410m", ["a totally different prompt set"])
    assert path is None and note and "DIFFERENT prompt set" in note


def test_resolve_null_unknown_base_is_none():
    path, note = qa._resolve_null_path("some/unknown-base", qa.DEFAULT_PROMPTS)
    assert path is None and note is None


def test_index_consistent_with_shipped_nulls_and_default_prompts():
    import json, hashlib, os
    nd = qa._nulls_dir()
    idx = json.load(open(os.path.join(nd, "INDEX.json")))
    for e in idx["nulls"]:
        nf = json.load(open(os.path.join(nd, e["file"])))         # the referenced file exists, parses,
        assert nf["base"] == e["base"]                            # and its own metadata matches the INDEX
        assert nf["prompts_sha8"] == e["prompts_sha8"]
        assert nf["K"] == e["K"]
    psha = hashlib.sha1("\n".join(qa.DEFAULT_PROMPTS).encode()).hexdigest()[:8]
    p410 = [e for e in idx["nulls"] if e["base"] == "EleutherAI/pythia-410m"][0]
    assert p410["prompts_sha8"] == psha                           # built on DEFAULT prompts -> auto-resolves


# --------------------------------------------------------------------------- probe-list --fail-on severity model
def test_probe_severity_calibrated():
    # WITH a null, only a calibrated-significant token has a severity; clean drift -> none
    assert qa._probe_severity("vocabulary-carried (fixable)", True, True) == "fixable"
    assert qa._probe_severity("vocabulary-carried (fixable)", False, True) == "none"
    assert qa._probe_severity("escalate (not vocabulary-carried)", True, True) == "escalate"
    assert qa._probe_severity("ambiguous (partial collapse; inspect)", True, True) == "ambiguous"
    assert qa._probe_severity("not elevated (nothing to classify)", True, True) == "none"


def test_probe_severity_uncalibrated():
    # WITHOUT a null: significance is uncalibrated; the residual-collapse classify verdict still types it
    assert qa._probe_severity("vocabulary-carried (fixable)", False, False) == "fixable"
    assert qa._probe_severity("escalate (not vocabulary-carried)", False, False) == "escalate"
    assert qa._probe_severity("ambiguous (partial collapse; inspect)", False, False) == "ambiguous"
    assert qa._probe_severity("not elevated (nothing to classify)", False, False) == "none"


# --------------------------------------------------------------------------- single-token resolution (watchlist coverage core)
class _FakeTok:
    """A minimal tokenizer modelling the leading-space behaviour distill_lint depends on: ' seven' is one
    token, ' owl'/'Google' are multi-token, and a bare 'seven' decodes back WITH a leading space (so the
    bare form is not 'single' even though it is one id) -- exactly the BPE quirk resolve-token surfaces."""
    _enc = {" seven": [1], "seven": [1], " owl": [2, 3], "owl": [2, 3],
            "Google": [4, 5], " Google": [6], "google": [4, 5]}
    _dec = {1: " seven", 2: " ow", 3: "l", 4: "Goog", 5: "le", 6: " Google"}

    def encode(self, s, add_special_tokens=False):
        return self._enc.get(s, [99])

    def decode(self, ids):
        return "".join(self._dec.get(i, "?") for i in ids)


def test_resolve_forms_finds_leading_space_single_token():
    forms = qa._resolve_forms(_FakeTok(), "seven")
    single = [f for f in forms if f["single_token"]]
    assert len(single) == 1 and single[0]["text"] == " seven"     # the ' seven' form is the usable handle


def test_resolve_forms_multitoken_has_no_single_handle():
    forms = qa._resolve_forms(_FakeTok(), "owl")                  # ' owl'/'owl' are multi-token here
    assert not any(f["single_token"] for f in forms)             # OUT OF SCOPE: a watchlist would skip it
    forms_g = qa._resolve_forms(_FakeTok(), "Google")            # ' Google' single, 'Google' multi
    assert any(f["single_token"] and f["text"] == " Google" for f in forms_g)


def test_fail_levels_any_is_union_catches_body_carried():
    assert qa._FAIL_LEVELS["none"] == set()
    assert qa._FAIL_LEVELS["fixable"] == {"fixable"}
    assert qa._FAIL_LEVELS["escalate"] == {"escalate", "ambiguous"}
    assert qa._FAIL_LEVELS["any"] == {"fixable", "escalate", "ambiguous"}
    # the fix: 'any' (default) catches a watchlisted leak whether fixable OR body-carried
    assert qa._FAIL_LEVELS["fixable"] | qa._FAIL_LEVELS["escalate"] == qa._FAIL_LEVELS["any"]
