"""
F2 -- real integration check against HuggingFaceTB/SmolLM2-360M.

A single run. The contract is not modified before observing the result: if
something fails, the error is recorded as is and decided on afterwards.

    pip install torch transformers
    python tests/integration/smollm2_run.py

Expectations recorded up front
------------------------------
  hidden_size              960   (declared in config.hidden_size)
  homogeneous candidates   1     (model.layers; the shape criterion has
                                  nothing to discriminate here)
  blocks                   32
  CoH total                5,901,344
  CoH trainable            5,901,312
  base                     every parameter with requires_grad == False

Predicted failure mode: CoHBlockWrapper does not delegate attributes, so an
external access such as layer.self_attn would raise AttributeError.
"""

from __future__ import annotations

import sys
import traceback

import torch

from pycoh.integration.injector import apply_coh
from pycoh.integration.inspector import inspect_model
from pycoh.integration.wrapper import CoHBlockWrapper

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
D_TAU = 96


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def check(label: str, got, expected=None) -> bool:
    if expected is None:
        print(f"  {label}: {got}")
        return True
    ok = got == expected
    mark = "OK " if ok else "MAL"
    print(f"  [{mark}] {label}: {got}" + ("" if ok else f"  (expected {expected})"))
    return ok


def main() -> int:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    section("ENVIRONMENT")
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")

    section("LOADING")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    print(f"  {MODEL_ID}")
    print(f"  base params: {sum(p.numel() for p in model.parameters()):,}")

    ids = tok("El taller abre a las ocho de la mañana.", return_tensors="pt")

    section("BASELINE (before injection)")
    with torch.no_grad():
        base_logits = model(**ids).logits
    print(f"  logits shape: {tuple(base_logits.shape)}")

    ok = True

    section("INSPECTION")
    info = inspect_model(model)
    ok &= check("hidden_size", info.hidden_size, 960)
    ok &= check("source", info.hidden_size_source, "config.hidden_size")
    ok &= check("already injected", info.already_injected, False)
    print(f"  homogeneous candidates: {len(info.candidates)}")
    for c in info.candidates:
        print(f"    - {c.path}: {c.length}x {c.block_type.__name__}")
        for e in c.evidence:
            print(f"        · {e}")

    section("INJECTION")
    try:
        apply_coh(model, d_tau=D_TAU)
    except Exception:
        print("  FAILURE in apply_coh:")
        traceback.print_exc()
        return 1

    wrappers = [m for m in model.modules() if isinstance(m, CoHBlockWrapper)]
    ok &= check("wrappers installed", len(wrappers), 32)

    coh_total = sum(p.numel() for w in wrappers for p in w.coh.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok &= check("CoH total", coh_total, 5_901_344)
    ok &= check("trainable", trainable, 5_901_312)

    coh_ids = {id(p) for w in wrappers for p in w.coh.parameters()}
    base_unfrozen = [
        n for n, p in model.named_parameters()
        if id(p) not in coh_ids and p.requires_grad
    ]
    ok &= check("unfrozen base params", len(base_unfrozen), 0)
    if base_unfrozen:
        print(f"        {base_unfrozen[:5]}")

    betas = {w.coh.beta.requires_grad for w in wrappers}
    ok &= check("beta trainable", betas, {False})

    section("FORWARD")
    try:
        with torch.no_grad():
            out = model(**ids).logits
        ok &= check("shape preserved", tuple(out.shape), tuple(base_logits.shape))
        delta = (out - base_logits).abs().max().item()
        print(f"  |Δ logits| max: {delta:.6f}  (must be > 0: CoH is not the identity at init)")
        ok &= check("finite", bool(torch.isfinite(out).all()), True)
    except Exception:
        print("  FAILURE in forward:")
        traceback.print_exc()
        return 1

    section("BACKWARD")
    try:
        model.train()
        loss = model(**ids, labels=ids["input_ids"]).loss
        loss.backward()
        print(f"  loss: {loss.item():.4f}")
        no_grad = [
            i for i, w in enumerate(wrappers)
            if w.coh.W_tau.weight.grad is None or w.coh.out_proj.weight.grad is None
        ]
        ok &= check("CoH layers without gradient", len(no_grad), 0)
        base_with_grad = [
            n for n, p in model.named_parameters()
            if id(p) not in coh_ids and p.grad is not None
        ]
        ok &= check("base params with gradient", len(base_with_grad), 0)
    except Exception:
        print("  FAILURE in backward:")
        traceback.print_exc()
        return 1

    section("DOUBLE INJECTION")
    try:
        apply_coh(model, d_tau=D_TAU)
        print("  [BAD] the second application did not fail")
        ok = False
    except RuntimeError as exc:
        print(f"  [OK ] rejected: {exc}")

    section("RESULT")
    print("  F2 INTEGRATION: " + ("ALL OK" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
