"""
F4 -- real training cycle against HuggingFaceTB/SmolLM2-360M.

Answers: does PyCoH survive actual training -- HuggingFace Trainer, mixed
precision, gradient checkpointing, adapter save and reload -- without
breaking anything?

It does not try to measure CoH's performance. The corpus is tiny and so is
the step count: what is checked is that the machinery works, not that the
mechanism is any good.

    PYTHONPATH=. python tests/integration/smollm2_train.py

On a T4 it takes a few minutes. It works on CPU but slowly.
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

# Sample corpus, deliberately small. Spanish text also exercises the
# tokenizer on non-English input.
CORPUS = """
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
    print(f"  [{'OK ' if ok else 'BAD'}] {label}: {got}"
          + ("" if ok else f"  (expected {expected})"))
    return ok


def main() -> int:
    import transformers
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    section("ENVIRONMENT")
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  device       {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    # inject with the model ALREADY on GPU: this is the order that used to
    # break before CoH inherited the block's device

    ids = tok(CORPUS, return_tensors="pt").to(device)

    section("BEFORE INJECTION")
    model.eval()
    with torch.no_grad():
        loss_base = model(**ids, labels=ids["input_ids"]).loss.item()
    print(f"  base loss: {loss_base:.4f}")
    base_snapshot = {
        n: p.detach().float().cpu().clone()
        for n, p in list(model.named_parameters())[:40]
    }

    section("INJECTION")
    apply_coh(model, d_tau=D_TAU)
    wrappers = [m for m in model.modules() if isinstance(m, CoHBlockWrapper)]
    ok = check("wrappers", len(wrappers), 32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok &= check("trainable", trainable, 5_901_312)

    coh_before = [
        w.coh.W_tau.weight.detach().float().cpu().clone() for w in wrappers
    ]

    section("DATASET")

    class Blocks(Dataset):
        def __init__(self, texto, tok, largo=128, n=64):
            enc = tok(texto * 4, return_tensors="pt")["input_ids"][0]
            self.trozos = [enc[i:i + largo] for i in range(0, len(enc) - largo, largo)]
            self.trozos = (self.trozos * (n // max(len(self.trozos), 1) + 1))[:n]

        def __len__(self):
            return len(self.trozos)

        def __getitem__(self, i):
            return {"input_ids": self.trozos[i], "labels": self.trozos[i].clone()}

    ds = Blocks(CORPUS, tok)
    print(f"  examples: {len(ds)}  tokens/example: {len(ds[0]['input_ids'])}")

    section(f"TRAINING ({STEPS} steps)")
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
        result = trainer.train()
        print(f"  final training loss: {result.training_loss:.4f}")
    except Exception:
        print("  FAILURE during training:")
        traceback.print_exc()
        return 1

    section("POST-TRAINING CHECKS")
    model.eval()
    with torch.no_grad():
        loss_trained = model(**ids, labels=ids["input_ids"]).loss.item()
    print(f"  base loss     : {loss_base:.4f}")
    print(f"  loss with CoH : {loss_trained:.4f}")
    print("  (single run, tiny corpus: this is not a performance measurement)")

    moved = [
        n for n, p in model.named_parameters()
        if n in base_snapshot
        and not torch.equal(p.detach().float().cpu(), base_snapshot[n])
    ]
    ok &= check("base weights modified", len(moved), 0)
    if moved:
        print(f"        {moved[:5]}")

    # After trainer.train() the gradients are None: the Trainer calls
    # zero_grad(set_to_none=True) at the end of each step. What has to be
    # checked is not that a gradient exists but that the weights moved.
    changed = sum(
        1 for w, before in zip(wrappers, coh_before)
        if not torch.equal(w.coh.W_tau.weight.detach().float().cpu(), before)
    )
    ok &= check("CoH layers whose weights changed", changed, 32)

    finite = all(
        torch.isfinite(p).all().item() for w in wrappers for p in w.coh.parameters()
    )
    ok &= check("CoH params finite", finite, True)

    betas = {round(w.coh.beta.item(), 6) for w in wrappers}
    ok &= check("betas frozen at 0.5", betas, {0.5})

    devs = {str(p.device) for w in wrappers for p in w.coh.parameters()}
    ok &= check("CoH on the model device", devs, {str(next(model.parameters()).device)})
    dtypes = {str(p.dtype) for w in wrappers for p in w.coh.parameters()}
    ok &= check("CoH in fp32 (base may be bf16)", dtypes, {"torch.float32"})

    section("ADAPTER")
    try:
        import os
        f = os.path.join(tempfile.mkdtemp(), "adapter.pt")
        meta = save_adapter(model, f)
        print(f"  saved: {os.path.getsize(f)/1e6:.1f} MB  ({len(meta['injected_paths'])} layers)")

        clean = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        apply_coh(clean, d_tau=D_TAU)
        load_adapter(clean, f)
        clean.eval()
        with torch.no_grad():
            loss_reloaded = clean(**ids, labels=ids["input_ids"]).loss.item()
        print(f"  loss after reload: {loss_reloaded:.4f}")
        ok &= check(
            "reproduces the loss",
            abs(loss_reloaded - loss_trained) < 1e-4,
            True,
        )
    except Exception:
        print("  FAILURE in the adapter cycle:")
        traceback.print_exc()
        return 1

    section("GENERATION")
    try:
        output = model.generate(**ids, max_new_tokens=12, do_sample=False)
        print("  " + tok.decode(output[0])[-160:])
    except Exception:
        print("  FAILURE in generate:")
        traceback.print_exc()
        ok = False

    section("RESULT")
    print("  F4 TRAINING: " + ("ALL OK" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
