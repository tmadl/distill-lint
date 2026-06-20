# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Tamas Madl
"""
Pinned HuggingFace commit revisions for the checkpoints this (detector-only) package references --
the E1-E5 evidence pairs and the real-model demo zoo. Ids not in the map load unpinned with a warning.

The masked vocabulary channel forms from fine-grained, precision-sensitive redistribution of mass and
depends on the *exact* convergent unembedding geometry of the pretrained weights, so reproducibility
requires a fixed snapshot of each base. Importing this module monkeypatches transformers'
``from_pretrained`` to inject ``revision=<sha>`` for any known model id; ids not in the map load
unpinned (with a one-time warning). Idempotent and side-effect-only on import.

Scope: only externally-pulled checkpoints need pinning. Teachers are fine-tuned *from* a base and
students *copy* a base in-script, so the bases (plus the open-ended judge) are the only downloaded
weights. The toy MNIST/MLP regime uses no pretrained weights and is unaffected.

Pythia is effectively immutable upstream; the Qwen / Gemma / OLMo / DeepSeek / Llama repos can be
silently re-uploaded, so pinning matters most there. The broader paper sweep (extra Pythia sizes, RWKV,
RedPajama, Mistral, the entity-replication Qwen) is pinned in the reproducibility artifact's copy of this
module, not this detector-only one.
"""
import warnings

REVISIONS = {
    # E1/E3/E4 evidence pairs + the _smoke fixture
    "EleutherAI/pythia-70m":                     "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
    "EleutherAI/pythia-410m":                    "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
    # E2 evidence (sycophancy classifier)
    "google/gemma-3-1b-it":                      "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    # E5 operating-point evidence (instruct vs its base), pinned to the exact E5-run commits.
    "allenai/OLMo-2-0425-1B-Instruct":           "48d788eca847d4d7548f375ad03d3c9312f6139e",
    "allenai/OLMo-2-0425-1B":                    "a1847dff35000b4271fa70afc5db10fd29fedbdf",
    "Qwen/Qwen3-0.6B":                           "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-0.6B-Base":                      "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "Qwen/Qwen3-1.7B":                           "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-1.7B-Base":                      "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    # real-model triage demo pairs (examples/real_models/model_zoo.yaml)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    "Qwen/Qwen2.5-Math-1.5B":                    "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":   "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "Qwen/Qwen2.5-Math-7B":                      "b101308fe89651ea5ce025f25317fea6fc07e96e",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":  "6a6f4aa4197940add57724a7707d069478df56b1",
    "meta-llama/Llama-3.1-8B":                   "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
}

_warned = set()


def _wrap(orig):
    def wrapped(model_id, *args, **kwargs):
        if "revision" not in kwargs:
            rev = REVISIONS.get(str(model_id))
            if rev is not None:
                kwargs["revision"] = rev
            elif str(model_id) not in _warned:
                _warned.add(str(model_id))
                warnings.warn(f"[model_revisions] no pinned revision for '{model_id}'; loading unpinned")
        return orig(model_id, *args, **kwargs)
    wrapped._pinned = True
    return wrapped


def install():
    """Patch transformers' Auto* from_pretrained to inject pinned revisions. Idempotent."""
    try:
        import transformers
    except ImportError:
        return False
    for name in ("AutoModelForCausalLM", "AutoModel", "AutoModelForSeq2SeqLM", "AutoTokenizer", "AutoConfig"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        orig = cls.from_pretrained                 # bound classmethod
        if getattr(orig, "_pinned", False):
            continue                               # already patched
        cls.from_pretrained = staticmethod(_wrap(orig))
    return True


install()
