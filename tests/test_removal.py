"""
Suite F5.1 -- adapter removal.

The contract is not "drop the wrapper" but restore the topology: after
remove_coh the model must be indistinguishable from a clean one for every
API that depends on structure.

Run with:  pytest -q tests/test_removal.py
"""

import pytest
import torch
import torch.nn as nn

from pycoh import apply_coh, remove_coh, save_adapter
from pycoh.integration.wrapper import CoHBlockWrapper
from tests.test_integration import D_MODEL, D_TAU, N_LAYERS, Block, TinyModel
from tests.test_training import fixed_batch, lm_loss, train


def clean():
    torch.manual_seed(0)
    return TinyModel()


def injected(**kw):
    torch.manual_seed(0)
    return apply_coh(TinyModel(), d_tau=D_TAU, **kw)


# -- 1. the topology goes back to the original -----------------------------

def test_module_paths_match_clean_model():
    ref = [n for n, _ in clean().named_modules()]
    m = injected()
    assert [n for n, _ in m.named_modules()] != ref  # injected differs
    remove_coh(m)
    assert [n for n, _ in m.named_modules()] == ref


def test_state_dict_keys_match_clean_model():
    """What makes save_pretrained work again."""
    ref = set(clean().state_dict())
    m = injected()
    assert set(m.state_dict()) != ref
    remove_coh(m)
    assert set(m.state_dict()) == ref


def test_original_blocks_preserved_by_identity():
    m = injected()
    originales = [w.block for w in m.layers]
    remove_coh(m)
    for original, actual in zip(originales, m.layers):
        assert actual is original
    assert all(isinstance(b, Block) for b in m.layers)


def test_no_coh_modules_remain():
    m = injected()
    remove_coh(m)
    assert not any(isinstance(x, CoHBlockWrapper) for x in m.modules())
    from pycoh.core.coh import CoH
    assert not any(isinstance(x, CoH) for x in m.modules())


# -- 2. functional equivalence --------------------------------------------

def test_forward_equals_clean_model():
    ids = torch.randint(0, 32, (2, 5))
    ref = clean()
    ref.eval()
    with torch.no_grad():
        expected = ref(ids)

    m = injected()
    remove_coh(m)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(ids), expected)


def test_base_weights_intact_after_train_and_remove():
    """
    The real case: inject, train, uninstall. The base weights were never
    touched, so the model must end up identical to the original.
    """
    ref = {n: p.clone() for n, p in clean().named_parameters()}
    m = injected()
    train(m, steps=30)
    remove_coh(m)
    for n, p in m.named_parameters():
        assert torch.equal(p, ref[n]), f"{n} differs"


def test_adapter_saved_before_removal_still_loads(tmp_path):
    """Removal does not invalidate an adapter saved beforehand."""
    from pycoh import load_adapter

    m = injected()
    train(m, steps=30)
    f = tmp_path / "a.pt"
    save_adapter(m, f)

    ids = fixed_batch()
    m.eval()
    with torch.no_grad():
        target_loss = lm_loss(m, ids).item()

    remove_coh(m)
    apply_coh(m, d_tau=D_TAU)
    load_adapter(m, f)
    m.eval()
    with torch.no_grad():
        assert lm_loss(m, ids).item() == pytest.approx(target_loss, rel=1e-6)


# -- 3. partial selection and re-application -------------------------------

def test_partial_selection_round_trip():
    ref = [n for n, _ in clean().named_modules()]
    m = injected(layers=[0, 2])
    assert sum(isinstance(b, CoHBlockWrapper) for b in m.layers) == 2
    remove_coh(m)
    assert [n for n, _ in m.named_modules()] == ref


def test_reapply_after_removal():
    m = injected()
    remove_coh(m)
    apply_coh(m, d_tau=D_TAU)  # must not fail as a double injection
    assert sum(isinstance(b, CoHBlockWrapper) for b in m.layers) == N_LAYERS


# -- 4. error contract and freezing ----------------------------------------

def test_remove_on_clean_model_raises():
    with pytest.raises(RuntimeError, match="has no CoH"):
        remove_coh(clean())


def test_double_removal_raises():
    m = injected()
    remove_coh(m)
    with pytest.raises(RuntimeError, match="has no CoH"):
        remove_coh(m)


def test_requires_grad_untouched_by_default():
    m = injected()
    remove_coh(m)
    assert all(p.requires_grad is False for p in m.parameters())


def test_unfreeze_flag():
    m = injected()
    remove_coh(m, unfreeze=True)
    assert all(p.requires_grad is True for p in m.parameters())


def test_returns_same_object():
    m = injected()
    assert remove_coh(m) is m
