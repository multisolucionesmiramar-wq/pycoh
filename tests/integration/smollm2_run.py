"""
F2 — prueba de integración real contra HuggingFaceTB/SmolLM2-360M.

Una sola corrida. No se modifica el contrato antes de observar el
resultado: si algo falla, el error se registra tal cual y se decide
después.

    pip install torch transformers
    python tests/integration/smollm2_run.py

Expectativas registradas de antemano
------------------------------------
  hidden_size            960   (declarado en config.hidden_size)
  candidatos homogéneos  1     (model.layers; el criterio de forma no
                                tiene que discriminar nada)
  bloques                32
  CoH totales            5 901 344
  CoH entrenables        5 901 312
  base                   todo con requires_grad == False

Modo de fallo previsto: CoHBlockWrapper no delega atributos, así que un
acceso externo del tipo layer.self_attn daría AttributeError.
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
    print(f"  [{mark}] {label}: {got}" + ("" if ok else f"  (esperado {expected})"))
    return ok


def main() -> int:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    section("ENTORNO")
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")

    section("CARGA")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    print(f"  {MODEL_ID}")
    print(f"  params base: {sum(p.numel() for p in model.parameters()):,}")

    ids = tok("El taller abre a las ocho de la mañana.", return_tensors="pt")

    section("BASELINE (antes de inyectar)")
    with torch.no_grad():
        base_logits = model(**ids).logits
    print(f"  logits shape: {tuple(base_logits.shape)}")

    ok = True

    section("INSPECCIÓN")
    info = inspect_model(model)
    ok &= check("hidden_size", info.hidden_size, 960)
    ok &= check("fuente", info.hidden_size_source, "config.hidden_size")
    ok &= check("ya inyectado", info.already_injected, False)
    print(f"  candidatos homogéneos: {len(info.candidates)}")
    for c in info.candidates:
        print(f"    - {c.path}: {c.length}x {c.block_type.__name__}")
        for e in c.evidence:
            print(f"        · {e}")

    section("INYECCIÓN")
    try:
        apply_coh(model, d_tau=D_TAU)
    except Exception:
        print("  FALLO en apply_coh:")
        traceback.print_exc()
        return 1

    wrappers = [m for m in model.modules() if isinstance(m, CoHBlockWrapper)]
    ok &= check("wrappers insertados", len(wrappers), 32)

    coh_total = sum(p.numel() for w in wrappers for p in w.coh.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok &= check("CoH totales", coh_total, 5_901_344)
    ok &= check("entrenables", trainable, 5_901_312)

    coh_ids = {id(p) for w in wrappers for p in w.coh.parameters()}
    base_libres = [
        n for n, p in model.named_parameters()
        if id(p) not in coh_ids and p.requires_grad
    ]
    ok &= check("parámetros base sin congelar", len(base_libres), 0)
    if base_libres:
        print(f"        {base_libres[:5]}")

    betas = {w.coh.beta.requires_grad for w in wrappers}
    ok &= check("beta entrenable", betas, {False})

    section("FORWARD")
    try:
        with torch.no_grad():
            out = model(**ids).logits
        ok &= check("shape preservada", tuple(out.shape), tuple(base_logits.shape))
        delta = (out - base_logits).abs().max().item()
        print(f"  |Δ logits| max: {delta:.6f}  (debe ser > 0: CoH no es identidad al inicio)")
        ok &= check("finito", bool(torch.isfinite(out).all()), True)
    except Exception:
        print("  FALLO en forward:")
        traceback.print_exc()
        return 1

    section("BACKWARD")
    try:
        model.train()
        loss = model(**ids, labels=ids["input_ids"]).loss
        loss.backward()
        print(f"  loss: {loss.item():.4f}")
        sin_grad = [
            i for i, w in enumerate(wrappers)
            if w.coh.W_tau.weight.grad is None or w.coh.out_proj.weight.grad is None
        ]
        ok &= check("capas CoH sin gradiente", len(sin_grad), 0)
        base_con_grad = [
            n for n, p in model.named_parameters()
            if id(p) not in coh_ids and p.grad is not None
        ]
        ok &= check("parámetros base con gradiente", len(base_con_grad), 0)
    except Exception:
        print("  FALLO en backward:")
        traceback.print_exc()
        return 1

    section("DOBLE INYECCIÓN")
    try:
        apply_coh(model, d_tau=D_TAU)
        print("  [MAL] la segunda aplicación no falló")
        ok = False
    except RuntimeError as exc:
        print(f"  [OK ] rechazada: {exc}")

    section("RESULTADO")
    print("  F2 INTEGRACIÓN: " + ("TODO OK" if ok else "HAY FALLOS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
