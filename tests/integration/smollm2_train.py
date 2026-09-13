"""
F4 — ciclo de entrenamiento real contra HuggingFaceTB/SmolLM2-360M.

Responde: ¿PyCoH sobrevive a un entrenamiento de verdad — Trainer de
HuggingFace, precisión mixta, gradient checkpointing, guardado y recarga
del adaptador — sin romper nada?

No pretende medir el rendimiento de CoH. El corpus es minúsculo y el
número de pasos también: lo que se comprueba es que la maquinaria
funciona, no que el mecanismo sirva.

    PYTHONPATH=. python tests/integration/smollm2_train.py

Con GPU T4 tarda unos minutos. En CPU funciona pero es lento.
"""

from __future__ import annotations

import sys
import tempfile
import traceback

import torch

from pycoh import apply_coh, load_adapter, save_adapter
from pycoh.integration.wrapper import CoHBlockWrapper

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
D_TAU = 96
STEPS = 60

TEXTO = """
El taller abre a las ocho de la mañana. Antes de abrir, el técnico revisa
el inventario de repuestos y anota lo que falta. Los equipos que llegaron
el día anterior esperan en el estante de diagnóstico, cada uno con su
boleta. La primera tarea es siempre la misma: medir voltajes en la fuente
antes de tocar cualquier otra cosa. Una fuente mal diagnosticada cuesta
más tiempo que todas las reparaciones de la semana. Cuando el equipo es
una refrigeradora, el orden cambia: primero se escucha el compresor, luego
se mide la corriente de arranque y solo después se abre el panel de
control. Los clientes llaman por la tarde para preguntar por sus equipos,
y conviene tener el diagnóstico escrito antes de esa hora.
""".strip()


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def check(label, got, expected=None) -> bool:
    if expected is None:
        print(f"  {label}: {got}")
        return True
    ok = got == expected
    print(f"  [{'OK ' if ok else 'MAL'}] {label}: {got}"
          + ("" if ok else f"  (esperado {expected})"))
    return ok


def main() -> int:
    import transformers
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    section("ENTORNO")
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  device       {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    # inyectar con el modelo YA en GPU: es el orden que rompía antes de que
    # CoH heredara el dispositivo del bloque

    ids = tok(TEXTO, return_tensors="pt").to(device)

    section("ANTES DE INYECTAR")
    model.eval()
    with torch.no_grad():
        loss_base = model(**ids, labels=ids["input_ids"]).loss.item()
    print(f"  loss base: {loss_base:.4f}")
    base_snapshot = {
        n: p.detach().float().cpu().clone()
        for n, p in list(model.named_parameters())[:40]
    }

    section("INYECCIÓN")
    apply_coh(model, d_tau=D_TAU)
    wrappers = [m for m in model.modules() if isinstance(m, CoHBlockWrapper)]
    ok = check("wrappers", len(wrappers), 32)
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok &= check("entrenables", entrenables, 5_901_312)

    coh_antes = [
        w.coh.W_tau.weight.detach().float().cpu().clone() for w in wrappers
    ]

    section("DATASET")

    class Bloques(Dataset):
        def __init__(self, texto, tok, largo=128, n=64):
            enc = tok(texto * 4, return_tensors="pt")["input_ids"][0]
            self.trozos = [enc[i:i + largo] for i in range(0, len(enc) - largo, largo)]
            self.trozos = (self.trozos * (n // max(len(self.trozos), 1) + 1))[:n]

        def __len__(self):
            return len(self.trozos)

        def __getitem__(self, i):
            return {"input_ids": self.trozos[i], "labels": self.trozos[i].clone()}

    ds = Bloques(TEXTO, tok)
    print(f"  ejemplos: {len(ds)}  tokens/ejemplo: {len(ds[0]['input_ids'])}")

    section(f"ENTRENAMIENTO ({STEPS} pasos)")
    try:
        args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_train_batch_size=2,
            max_steps=STEPS,
            learning_rate=1e-3,
            logging_steps=10,
            report_to=[],
            bf16=(device == "cuda"),
            gradient_checkpointing=True,
            save_strategy="no",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        )
        salida = trainer.train()
        print(f"  loss final de entrenamiento: {salida.training_loss:.4f}")
    except Exception:
        print("  FALLO durante el entrenamiento:")
        traceback.print_exc()
        return 1

    section("VERIFICACIONES POST-ENTRENAMIENTO")
    model.eval()
    with torch.no_grad():
        loss_entrenado = model(**ids, labels=ids["input_ids"]).loss.item()
    print(f"  loss base       : {loss_base:.4f}")
    print(f"  loss con CoH    : {loss_entrenado:.4f}")
    print(f"  (una sola corrida, corpus mínimo: no es medición de rendimiento)")

    movidos = [
        n for n, p in model.named_parameters()
        if n in base_snapshot
        and not torch.equal(p.detach().float().cpu(), base_snapshot[n])
    ]
    ok &= check("pesos base modificados", len(movidos), 0)
    if movidos:
        print(f"        {movidos[:5]}")

    # Tras trainer.train() los gradientes están en None: el Trainer llama a
    # zero_grad(set_to_none=True) al cerrar cada paso. Lo que hay que
    # comprobar no es que exista gradiente sino que los pesos se movieron.
    cambiaron = sum(
        1 for w, antes in zip(wrappers, coh_antes)
        if not torch.equal(w.coh.W_tau.weight.detach().float().cpu(), antes)
    )
    ok &= check("capas CoH cuyos pesos cambiaron", cambiaron, 32)

    finitos = all(
        torch.isfinite(p).all().item() for w in wrappers for p in w.coh.parameters()
    )
    ok &= check("parámetros CoH finitos", finitos, True)

    betas = {round(w.coh.beta.item(), 6) for w in wrappers}
    ok &= check("betas congelados en 0.5", betas, {0.5})

    devs = {str(p.device) for w in wrappers for p in w.coh.parameters()}
    ok &= check("CoH en el dispositivo del modelo", devs, {str(next(model.parameters()).device)})
    dtypes = {str(p.dtype) for w in wrappers for p in w.coh.parameters()}
    ok &= check("CoH en fp32 (la base puede estar en bf16)", dtypes, {"torch.float32"})

    section("ADAPTADOR")
    try:
        import os
        f = os.path.join(tempfile.mkdtemp(), "adapter.pt")
        meta = save_adapter(model, f)
        print(f"  guardado: {os.path.getsize(f)/1e6:.1f} MB  ({len(meta['injected_paths'])} capas)")

        limpio = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        apply_coh(limpio, d_tau=D_TAU)
        load_adapter(limpio, f)
        limpio.eval()
        with torch.no_grad():
            loss_recargado = limpio(**ids, labels=ids["input_ids"]).loss.item()
        print(f"  loss tras recargar: {loss_recargado:.4f}")
        ok &= check(
            "reproduce la pérdida",
            abs(loss_recargado - loss_entrenado) < 1e-4,
            True,
        )
    except Exception:
        print("  FALLO en el ciclo del adaptador:")
        traceback.print_exc()
        return 1

    section("GENERACIÓN")
    try:
        salida = model.generate(**ids, max_new_tokens=12, do_sample=False)
        print("  " + tok.decode(salida[0])[-160:])
    except Exception:
        print("  FALLO en generate:")
        traceback.print_exc()
        ok = False

    section("RESULTADO")
    print("  F4 ENTRENAMIENTO: " + ("TODO OK" if ok else "HAY FALLOS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
