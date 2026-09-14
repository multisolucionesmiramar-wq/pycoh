"""
Suite F4 -- minimal training cycle.

The question it answers: does PyCoH take part in real training without
breaking backward, parameter updates, adapter checkpointing or numerical
stability?

Everything here runs on CPU in seconds. Verification against a real model
and a real Trainer lives in tests/integration/smollm2_train.py.

Run with:  pytest -q tests/test_training.py
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
    """Deliberately naive user loop: a single parameter group."""
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


# -- 1. the cycle works ---------------------------------------------------

def test_loss_decreases_with_only_coh_trainable():
    """
    Memorization test: with the base frozen and only CoH trainable, the
    loss on a fixed batch must go down. If it does not, the gradient is not
    reaching where we think it is.
    """
    model = build()
    h = train(model)
    assert h[-1] < h[0] * 0.95, f"the loss barely moved: {h[0]:.4f} -> {h[-1]:.4f}"
    assert all(torch.isfinite(torch.tensor(v)) for v in h)


def test_parameters_actually_change():
    """
    Having a gradient does not mean the optimizer writes. Compare before
    and after.
    """
    model = build()
    antes = snapshot(model)
    train(model, steps=20)
    despues = snapshot(model)

    movidos = [n for n in antes if not torch.equal(antes[n], despues[n])]
    esperados = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sorted(movidos) == sorted(esperados)
    assert len(movidos) == N_LAYERS * 3  # W_tau, phi_proj, out_proj per layer


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
            assert torch.equal(p, base_antes[n]), f"{n} moved"


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


# -- 2. numerical stability -----------------------------------------------

def test_no_nan_under_amp():
    model = build()
    h = train(model, steps=100, amp=True)
    assert all(v == v for v in h), "NaN appeared under mixed precision"
    assert h[-1] < h[0]


def test_no_nan_with_aggressive_lr():
    """
    At a high lr the loss may not go down, but the mechanism must not
    produce NaN or Inf: the clamp and the normalization bound the
    correction.
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


# -- 3. gradient checkpointing --------------------------------------------

class CheckpointedModel(nn.Module):
    """Calls each layer through torch.utils.checkpoint, positionally."""

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
    The path HuggingFace uses to save memory: recompute the forward during
    the backward, calling the layer positionally.
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


# -- 4. the adapter survives training -------------------------------------

def test_adapter_checkpoint_mid_training(tmp_path):
    """
    Saving mid-training, loading into a clean model and carrying on must
    yield exactly the same trajectory.
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
    A trained adapter, loaded onto a clean instance of the same base model,
    reproduces the loss. This is the real use case: shipping 23 MB instead
    of 700.
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
    Control: the same model without CoH and with the base frozen has
    nothing to train, so its loss does not move. The improvement observed
    comes from the adapter, not from the loop.
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
