"""
Suite F5.1 — desinstalación del adaptador.

El contrato no es "quitar el wrapper" sino restaurar la topología: tras
remove_coh, el modelo debe ser indistinguible del modelo limpio para las
APIs que dependen de la estructura.

Ejecutar:  pytest -q tests/test_removal.py
"""

import pytest
import torch
import torch.nn as nn

from pycoh import apply_coh, remove_coh, save_adapter
from pycoh.integration.wrapper import CoHBlockWrapper
from tests.test_integration import D_MODEL, D_TAU, N_LAYERS, Block, TinyModel
from tests.test_training import fixed_batch, lm_loss, train


def limpio():
    torch.manual_seed(0)
    return TinyModel()


def inyectado(**kw):
    torch.manual_seed(0)
    return apply_coh(TinyModel(), d_tau=D_TAU, **kw)


# ── 1. la topología vuelve a ser la original ─────────────────────────────

def test_module_paths_match_clean_model():
    ref = [n for n, _ in limpio().named_modules()]
    m = inyectado()
    assert [n for n, _ in m.named_modules()] != ref  # inyectado difiere
    remove_coh(m)
    assert [n for n, _ in m.named_modules()] == ref


def test_state_dict_keys_match_clean_model():
    """Lo que hace que save_pretrained vuelva a funcionar."""
    ref = set(limpio().state_dict())
    m = inyectado()
    assert set(m.state_dict()) != ref
    remove_coh(m)
    assert set(m.state_dict()) == ref


def test_original_blocks_preserved_by_identity():
    m = inyectado()
    originales = [w.block for w in m.layers]
    remove_coh(m)
    for original, actual in zip(originales, m.layers):
        assert actual is original
    assert all(isinstance(b, Block) for b in m.layers)


def test_no_coh_modules_remain():
    m = inyectado()
    remove_coh(m)
    assert not any(isinstance(x, CoHBlockWrapper) for x in m.modules())
    from pycoh.core.coh import CoH
    assert not any(isinstance(x, CoH) for x in m.modules())


# ── 2. equivalencia funcional ────────────────────────────────────────────

def test_forward_equals_clean_model():
    ids = torch.randint(0, 32, (2, 5))
    ref = limpio()
    ref.eval()
    with torch.no_grad():
        esperado = ref(ids)

    m = inyectado()
    remove_coh(m)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(ids), esperado)


def test_base_weights_intact_after_train_and_remove():
    """
    El caso real: inyectar, entrenar, desinstalar. Los pesos base nunca se
    tocaron, así que el modelo debe quedar idéntico al original.
    """
    ref = {n: p.clone() for n, p in limpio().named_parameters()}
    m = inyectado()
    train(m, steps=30)
    remove_coh(m)
    for n, p in m.named_parameters():
        assert torch.equal(p, ref[n]), f"{n} difiere"


def test_adapter_saved_before_removal_still_loads(tmp_path):
    """Desinstalar no invalida un adaptador guardado antes."""
    from pycoh import load_adapter

    m = inyectado()
    train(m, steps=30)
    f = tmp_path / "a.pt"
    save_adapter(m, f)

    ids = fixed_batch()
    m.eval()
    with torch.no_grad():
        objetivo = lm_loss(m, ids).item()

    remove_coh(m)
    apply_coh(m, d_tau=D_TAU)
    load_adapter(m, f)
    m.eval()
    with torch.no_grad():
        assert lm_loss(m, ids).item() == pytest.approx(objetivo, rel=1e-6)


# ── 3. selección parcial y reaplicación ──────────────────────────────────

def test_partial_selection_round_trip():
    ref = [n for n, _ in limpio().named_modules()]
    m = inyectado(layers=[0, 2])
    assert sum(isinstance(b, CoHBlockWrapper) for b in m.layers) == 2
    remove_coh(m)
    assert [n for n, _ in m.named_modules()] == ref


def test_reapply_after_removal():
    m = inyectado()
    remove_coh(m)
    apply_coh(m, d_tau=D_TAU)  # no debe fallar por doble inyección
    assert sum(isinstance(b, CoHBlockWrapper) for b in m.layers) == N_LAYERS


# ── 4. contrato de errores y congelamiento ───────────────────────────────

def test_remove_on_clean_model_raises():
    with pytest.raises(RuntimeError, match="has no CoH"):
        remove_coh(limpio())


def test_double_removal_raises():
    m = inyectado()
    remove_coh(m)
    with pytest.raises(RuntimeError, match="has no CoH"):
        remove_coh(m)


def test_requires_grad_untouched_by_default():
    m = inyectado()
    remove_coh(m)
    assert all(p.requires_grad is False for p in m.parameters())


def test_unfreeze_flag():
    m = inyectado()
    remove_coh(m, unfreeze=True)
    assert all(p.requires_grad is True for p in m.parameters())


def test_returns_same_object():
    m = inyectado()
    assert remove_coh(m) is m
