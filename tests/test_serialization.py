"""
Suite F3 — serialización del adaptador.

Ejecutar:  pytest -q tests/test_serialization.py
"""

import copy

import pytest
import torch
import torch.nn as nn

from pycoh.integration.injector import apply_coh
from pycoh.integration.serialization import (
    FORMAT_VERSION,
    adapter_metadata,
    load_adapter,
    save_adapter,
)
from pycoh.integration.wrapper import CoHBlockWrapper
from tests.test_integration import D_MODEL, D_TAU, N_LAYERS, TinyModel


def injected(**kw):
    torch.manual_seed(0)
    return apply_coh(TinyModel(), d_tau=D_TAU, **kw)


def coh_state(model):
    return {
        path: {k: v.clone() for k, v in m.coh.state_dict().items()}
        for path, m in model.named_modules()
        if isinstance(m, CoHBlockWrapper)
    }


def randomize_coh(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CoHBlockWrapper):
                for p in m.coh.parameters():
                    p.copy_(torch.randn_like(p))


# ── 1. round-trip ────────────────────────────────────────────────────────

def test_round_trip_is_exact(tmp_path):
    origen = injected()
    randomize_coh(origen)
    esperado = coh_state(origen)

    f = tmp_path / "adapter.pt"
    save_adapter(origen, f)

    destino = injected()
    assert coh_state(destino) != esperado  # pesos distintos antes de cargar
    load_adapter(destino, f)

    got = coh_state(destino)
    assert set(got) == set(esperado)
    for path in esperado:
        for k in esperado[path]:
            assert torch.equal(got[path][k], esperado[path][k]), f"{path}.{k}"


def test_round_trip_reproduces_delta(tmp_path):
    """El round-trip se verifica también funcionalmente, no solo por estado."""
    origen = injected()
    randomize_coh(origen)
    h = torch.randn(2, 5, D_MODEL)
    ref = [m.coh(h) for m in origen.layers]

    f = tmp_path / "adapter.pt"
    save_adapter(origen, f)
    destino = injected()
    load_adapter(destino, f)

    for esperado, m in zip(ref, destino.layers):
        assert torch.equal(m.coh(h), esperado)


def test_beta_value_prevails_over_beta_init(tmp_path):
    origen = apply_coh(TinyModel(), d_tau=D_TAU, beta_init=0.73)
    f = tmp_path / "a.pt"
    save_adapter(origen, f)

    destino = apply_coh(TinyModel(), d_tau=D_TAU, beta_init=0.5)
    load_adapter(destino, f)
    for m in destino.layers:
        assert m.coh.beta.item() == pytest.approx(0.73)


# ── 2. contenido del archivo ─────────────────────────────────────────────

def test_adapter_contains_only_coh_weights(tmp_path):
    model = injected()
    f = tmp_path / "a.pt"
    save_adapter(model, f)

    payload = torch.load(f, map_location="cpu", weights_only=True)
    keys = list(payload["state_dict"])

    esperadas = {"W_tau.weight", "phi_proj.weight", "out_proj.weight", "beta"}
    assert len(keys) == N_LAYERS * len(esperadas)
    for k in keys:
        assert k.rsplit(".", 2)[-2] + "." + k.rsplit(".", 1)[-1] in esperadas or k.endswith(".beta")

    total = sum(v.numel() for v in payload["state_dict"].values())
    por_capa = 2 * D_MODEL * D_TAU + D_TAU + 1
    assert total == N_LAYERS * por_capa

    # ningún peso del huésped
    base = sum(p.numel() for p in TinyModel().parameters())
    assert total < base * 10  # sanity: no se coló el modelo entero
    assert not any("attn" in k or "mlp" in k or "embed" in k or "head" in k for k in keys)


def test_metadata_fields(tmp_path):
    model = apply_coh(TinyModel(), d_tau=D_TAU, r_max=0.9, trainable_beta=True)
    f = tmp_path / "a.pt"
    escrito = save_adapter(model, f)
    leido = adapter_metadata(f)

    assert escrito == leido
    assert leido["format_version"] == FORMAT_VERSION
    assert leido["mechanism"] == "coh"
    assert leido["d_model"] == D_MODEL
    assert leido["d_tau"] == D_TAU
    assert leido["r_max"] == 0.9
    assert leido["trainable_beta"] is True
    assert leido["injected_paths"] == [f"layers.{i}" for i in range(N_LAYERS)]


def test_save_requires_injected_model(tmp_path):
    with pytest.raises(RuntimeError, match="no tiene CoH"):
        save_adapter(TinyModel(), tmp_path / "a.pt")


def test_save_rejects_heterogeneous_configs(tmp_path):
    from pycoh.core.coh import CoH

    model = injected()
    model.layers[1] = CoHBlockWrapper(model.layers[1].block, CoH(D_MODEL, D_TAU + 4))
    with pytest.raises(ValueError, match="no comparten configuración"):
        save_adapter(model, tmp_path / "a.pt")


# ── 3. rechazos ──────────────────────────────────────────────────────────

def test_load_requires_injected_model(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(injected(), f)
    with pytest.raises(RuntimeError, match="no tiene CoH"):
        load_adapter(TinyModel(), f)


def test_topology_mismatch_rejected(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(apply_coh(TinyModel(), d_tau=D_TAU, layers=[0, 1]), f)

    destino = apply_coh(TinyModel(), d_tau=D_TAU, layers=[0, 2])
    with pytest.raises(RuntimeError, match="topología"):
        load_adapter(destino, f)


def test_layer_count_mismatch_rejected(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(apply_coh(TinyModel(), d_tau=D_TAU), f)
    destino = apply_coh(TinyModel(), d_tau=D_TAU, layers=2)
    with pytest.raises(RuntimeError, match="topología"):
        load_adapter(destino, f)


def test_d_tau_mismatch_rejected(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(apply_coh(TinyModel(), d_tau=D_TAU), f)
    destino = apply_coh(TinyModel(), d_tau=D_TAU * 2)
    with pytest.raises(RuntimeError, match="d_tau incompatible"):
        load_adapter(destino, f)


def test_r_max_mismatch_rejected(tmp_path):
    """
    r_max no cambia ninguna forma, así que load_state_dict no lo detecta:
    sin esta validación el adaptador se cargaría y el mecanismo se
    comportaría distinto en silencio.
    """
    f = tmp_path / "a.pt"
    save_adapter(apply_coh(TinyModel(), d_tau=D_TAU, r_max=0.98), f)
    destino = apply_coh(TinyModel(), d_tau=D_TAU, r_max=0.90)
    with pytest.raises(RuntimeError, match="r_max incompatible"):
        load_adapter(destino, f)


@pytest.mark.parametrize(
    "mutacion, patron",
    [
        ({"mechanism": "lora"}, "mecanismo desconocido"),
        ({"format_version": 99}, "format_version"),
    ],
)
def test_foreign_payload_rejected(tmp_path, mutacion, patron):
    f = tmp_path / "a.pt"
    save_adapter(injected(), f)
    payload = torch.load(f, map_location="cpu", weights_only=True)
    payload["metadata"].update(mutacion)
    torch.save(payload, f)

    with pytest.raises(ValueError, match=patron):
        load_adapter(injected(), f)


def test_malformed_payload_rejected(tmp_path):
    f = tmp_path / "basura.pt"
    torch.save({"cualquier": "cosa"}, f)
    with pytest.raises(ValueError, match="no tiene la forma"):
        load_adapter(injected(), f)


def test_missing_keys_rejected(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(injected(), f)
    payload = torch.load(f, map_location="cpu", weights_only=True)
    del payload["state_dict"]["layers.2.beta"]
    torch.save(payload, f)

    with pytest.raises(RuntimeError, match="claves incompatibles"):
        load_adapter(injected(), f)


def test_orphan_keys_rejected(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(injected(), f)
    payload = torch.load(f, map_location="cpu", weights_only=True)
    payload["state_dict"]["layers.99.beta"] = torch.tensor(0.5)
    torch.save(payload, f)

    with pytest.raises(RuntimeError, match="no corresponden"):
        load_adapter(injected(), f)


# ── 4. sin carga parcial ─────────────────────────────────────────────────

def test_no_partial_load_on_bad_shape(tmp_path):
    """
    La capa 0 sería válida y la 3 no. Con validación previa completa, el
    modelo debe quedar intacto: ni siquiera la capa 0 se escribe.
    """
    f = tmp_path / "a.pt"
    origen = injected()
    randomize_coh(origen)
    save_adapter(origen, f)

    payload = torch.load(f, map_location="cpu", weights_only=True)
    payload["state_dict"]["layers.3.W_tau.weight"] = torch.randn(D_TAU + 1, D_MODEL)
    torch.save(payload, f)

    destino = injected()
    antes = coh_state(destino)
    with pytest.raises(RuntimeError, match="forma incompatible"):
        load_adapter(destino, f)

    despues = coh_state(destino)
    for path in antes:
        for k in antes[path]:
            assert torch.equal(despues[path][k], antes[path][k]), f"{path}.{k} mutado"


def test_no_partial_load_on_topology_error(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(apply_coh(TinyModel(), d_tau=D_TAU, layers=[0, 1]), f)

    destino = apply_coh(TinyModel(), d_tau=D_TAU, layers=[0, 2])
    antes = coh_state(destino)
    with pytest.raises(RuntimeError):
        load_adapter(destino, f)
    despues = coh_state(destino)
    for p in antes:
        for k in antes[p]:
            assert torch.equal(despues[p][k], antes[p][k])


# ── 5. el modelo base no se toca ─────────────────────────────────────────

def test_base_weights_untouched_by_load(tmp_path):
    f = tmp_path / "a.pt"
    origen = injected()
    randomize_coh(origen)
    save_adapter(origen, f)

    destino = injected()
    base_antes = {
        n: p.clone()
        for n, p in destino.named_parameters()
        if ".coh." not in n
    }
    load_adapter(destino, f)
    for n, p in destino.named_parameters():
        if ".coh." not in n:
            assert torch.equal(p, base_antes[n]), f"{n} mutado"


def test_trainable_beta_is_informative_not_prescriptive(tmp_path):
    f = tmp_path / "a.pt"
    save_adapter(apply_coh(TinyModel(), d_tau=D_TAU, trainable_beta=True), f)

    destino = apply_coh(TinyModel(), d_tau=D_TAU, trainable_beta=False)
    meta = load_adapter(destino, f)
    assert meta["trainable_beta"] is True
    for m in destino.layers:
        assert m.coh.beta.requires_grad is False
