"""
Suite F4 — ciclo mínimo de entrenamiento.

La pregunta que responde: ¿PyCoH participa en un entrenamiento real sin
romper el backward, la actualización de parámetros, el checkpoint del
adaptador ni la estabilidad numérica?

Todo lo de aquí corre en CPU en segundos. La verificación contra un
modelo y un Trainer reales está en tests/integration/smollm2_train.py.

Ejecutar:  pytest -q tests/test_training.py
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pycoh.integration.injector import apply_coh
from pycoh.integration.serialization import load_adapter, save_adapter
from pycoh.integration.wrapper import CoHBlockWrapper
from tests.test_integration import D_MODEL, D_TAU, N_LAYERS, TinyModel

VOCAB, SEQ, BATCH = 32, 12, 4


def fixed_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=g)


def lm_loss(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
    )


def train(model, steps=150, lr=1e-2, ids=None, amp=False, opt=None):
    """Bucle de usuario deliberadamente ingenuo: una sola bolsa de parámetros."""
    ids = fixed_batch() if ids is None else ids
    opt = opt or torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    historia = []
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        if amp:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = lm_loss(model, ids)
        else:
            loss = lm_loss(model, ids)
        loss.backward()
        opt.step()
        historia.append(loss.item())
    return historia


def build(**kw):
    torch.manual_seed(0)
    return apply_coh(TinyModel(), d_tau=D_TAU, **kw)


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ── 1. el ciclo funciona ─────────────────────────────────────────────────

def test_loss_decreases_with_only_coh_trainable():
    """
    Prueba de memorización: con la base congelada y solo CoH entrenable,
    la pérdida sobre un lote fijo debe bajar. Si no baja, el gradiente no
    está llegando a donde creemos.
    """
    model = build()
    h = train(model)
    assert h[-1] < h[0] * 0.95, f"la pérdida apenas se movió: {h[0]:.4f} → {h[-1]:.4f}"
    assert all(torch.isfinite(torch.tensor(v)) for v in h)


def test_parameters_actually_change():
    """
    Que haya gradiente no implica que el optimizador escriba. Se compara
    antes y después.
    """
    model = build()
    antes = snapshot(model)
    train(model, steps=20)
    despues = snapshot(model)

    movidos = [n for n in antes if not torch.equal(antes[n], despues[n])]
    esperados = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sorted(movidos) == sorted(esperados)
    assert len(movidos) == N_LAYERS * 3  # W_tau, phi_proj, out_proj por capa


def test_base_weights_never_move_during_training():
    model = build()
    coh_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, CoHBlockWrapper)
        for p in m.coh.parameters()
    }
    base_antes = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if id(p) not in coh_ids
    }
    train(model, steps=50)
    for n, p in model.named_parameters():
        if n in base_antes:
            assert torch.equal(p, base_antes[n]), f"{n} se movió"


def test_beta_frozen_stays_frozen_through_training():
    model = build()
    betas = [m.coh.beta.item() for m in model.layers]
    train(model, steps=50)
    assert [m.coh.beta.item() for m in model.layers] == betas


def test_trainable_beta_moves():
    model = build(trainable_beta=True)
    antes = [m.coh.beta.item() for m in model.layers]
    train(model, steps=50)
    despues = [m.coh.beta.item() for m in model.layers]
    assert any(a != d for a, d in zip(antes, despues))


# ── 2. estabilidad numérica ──────────────────────────────────────────────

def test_no_nan_under_amp():
    model = build()
    h = train(model, steps=100, amp=True)
    assert all(v == v for v in h), "apareció NaN bajo precisión mixta"
    assert h[-1] < h[0]


def test_no_nan_with_aggressive_lr():
    """
    Con un lr alto la pérdida puede no bajar, pero el mecanismo no debe
    producir NaN ni Inf: el clamp y la normalización acotan la corrección.
    """
    model = build()
    h = train(model, steps=80, lr=1.0)
    assert all(torch.isfinite(torch.tensor(v)) for v in h)
    for m in model.modules():
        if isinstance(m, CoHBlockWrapper):
            for p in m.coh.parameters():
                assert torch.isfinite(p).all()


def test_scale_stays_within_bounds_after_training():
    model = build()
    train(model, steps=100)
    h = torch.randn(2, SEQ, D_MODEL)
    s_max = (1.0 / torch.sqrt(1.0 - torch.tensor(0.98) ** 2) - 1.0).item()
    for m in model.layers:
        st = m.coh._stages(h)
        assert (st["scale"] >= 0).all()
        assert st["scale"].max().item() <= s_max * (1 + 1e-5)


# ── 3. gradient checkpointing ────────────────────────────────────────────

class CheckpointedModel(nn.Module):
    """Llama a cada capa a través de torch.utils.checkpoint, posicionalmente."""

    def __init__(self, base: TinyModel):
        super().__init__()
        self.embed, self.layers, self.head = base.embed, base.layers, base.head
        self.config = base.config

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = checkpoint(layer, h, use_reentrant=False)
        return self.head(h)


def test_gradient_checkpointing_path():
    """
    El camino que HuggingFace usa para ahorrar memoria: recalcular el
    forward durante el backward, llamando a la capa posicionalmente.
    """
    model = build()
    ck = CheckpointedModel(model)
    ids = fixed_batch()

    directo = lm_loss(model, ids)
    recalc = lm_loss(ck, ids)
    assert torch.allclose(directo, recalc, rtol=1e-5)

    h = train(ck, steps=30)
    assert h[-1] < h[0]
    for m in model.modules():
        if isinstance(m, CoHBlockWrapper):
            assert m.coh.W_tau.weight.grad is not None


# ── 4. el adaptador sobrevive al entrenamiento ───────────────────────────

def test_adapter_checkpoint_mid_training(tmp_path):
    """
    Guardar a mitad de entrenamiento, cargar en un modelo limpio y
    continuar debe dar exactamente la misma trayectoria.
    """
    ids = fixed_batch()

    a = build()
    train(a, steps=40, ids=ids)
    f = tmp_path / "mitad.pt"
    save_adapter(a, f)
    resto_a = train(a, steps=20, ids=ids)

    b = build()
    load_adapter(b, f)
    resto_b = train(b, steps=20, ids=ids)

    assert resto_a[0] == pytest.approx(resto_b[0], rel=1e-6)
    assert resto_a[-1] == pytest.approx(resto_b[-1], rel=1e-6)


def test_trained_adapter_transfers_to_fresh_model(tmp_path):
    """
    Un adaptador entrenado, cargado sobre una instancia limpia del mismo
    modelo base, reproduce la pérdida. Es el caso de uso real: distribuir
    23 MB en vez de 700.
    """
    ids = fixed_batch()
    entrenado = build()
    train(entrenado, steps=80, ids=ids)
    entrenado.eval()
    with torch.no_grad():
        objetivo = lm_loss(entrenado, ids).item()

    f = tmp_path / "entrenado.pt"
    save_adapter(entrenado, f)

    limpio = build()
    load_adapter(limpio, f)
    limpio.eval()
    with torch.no_grad():
        assert lm_loss(limpio, ids).item() == pytest.approx(objetivo, rel=1e-6)


def test_coh_beats_frozen_baseline():
    """
    Control: el mismo modelo sin CoH y con la base congelada no tiene nada
    que entrenar, así que su pérdida no se mueve. La mejora que se observa
    viene del adaptador y no del bucle.
    """
    ids = fixed_batch()
    torch.manual_seed(0)
    base = TinyModel()
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        sin_coh = lm_loss(base, ids).item()

    con_coh = build()
    h = train(con_coh, steps=150, ids=ids)
    assert h[-1] < sin_coh
