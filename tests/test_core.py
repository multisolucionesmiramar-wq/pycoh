"""
Suite F0 — contrato del núcleo CoH.

Ejecutar:  pytest -q tests/test_core.py
"""

import math

import pytest
import torch
import torch.nn.functional as F

from pycoh.core.coh import CoH

D_MODEL, D_TAU, B, T = 16, 8, 2, 5
SEED = 0

# Cota superior de scale con r_max=0.98, evaluada en la misma precisión
# que usa el núcleo.
S_MAX_FP32 = (
    1.0 / torch.sqrt(1.0 - torch.tensor(0.98, dtype=torch.float32) ** 2) - 1.0
).item()


def make(**kw) -> CoH:
    torch.manual_seed(SEED)
    return CoH(D_MODEL, D_TAU, **kw)


def reference_delta(coh: CoH, h: torch.Tensor) -> dict:
    """
    Referencia matemática independiente, escrita directamente desde la
    especificación. No llama a ningún método del módulo bajo prueba.
    """
    h32 = h.float()
    W_tau = coh.W_tau.weight.detach().float()
    W_phi = coh.phi_proj.weight.detach().float()
    W_out = coh.out_proj.weight.detach().float()
    beta = coh.beta.detach().float()

    z = h32 @ W_tau.T
    gate = torch.sigmoid(z @ W_phi.T)
    ratio = gate.clamp(max=coh.r_max)
    F_val = 1.0 / torch.sqrt(1.0 - ratio ** 2)
    scale = F_val - 1.0

    raw = z @ W_out.T
    norm = raw.norm(dim=-1, keepdim=True).clamp_min(CoH.NORMALIZE_EPS)
    d_hat = raw / norm

    return {
        "z": z,
        "ratio": ratio,
        "F_val": F_val,
        "scale": scale,
        "d_hat": d_hat,
        "delta": (beta * d_hat) * scale,
    }


# ── 1. matemática ────────────────────────────────────────────────────────

def test_arithmetic_stages_match_reference_bitwise():
    """Etapas de aritmética pura: igualdad bit a bit, sin tolerancia."""
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    got = coh._stages(h)
    ref = reference_delta(coh, h)
    for k in ("z", "ratio", "F_val", "scale"):
        assert torch.equal(got[k], ref[k]), f"divergencia en {k}"


def test_direction_and_delta_match_reference():
    """
    d_hat y delta pasan por F.normalize. La referencia reproduce su
    semántica matemática, no su implementación, así que aquí se exige
    equivalencia numérica y no identidad de bits.
    """
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    got = coh._stages(h)
    ref = reference_delta(coh, h)
    for k in ("d_hat", "delta"):
        torch.testing.assert_close(got[k], ref[k], rtol=1e-6, atol=1e-7)


def test_normalize_semantics_match_definition():
    """
    Aísla la suposición frágil: que F.normalize sea exactamente
    x / max(||x||, eps). Si una versión de PyTorch cambia su
    implementación, falla este test y no la referencia matemática.
    """
    x = torch.randn(4, 7)
    ref = x / x.norm(dim=-1, keepdim=True).clamp_min(CoH.NORMALIZE_EPS)
    got = F.normalize(x, dim=-1, eps=CoH.NORMALIZE_EPS)
    assert torch.equal(got, ref)


def test_output_shape_and_finiteness():
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    delta = coh(h)
    assert delta.shape == h.shape
    assert torch.isfinite(delta).all()


def test_deterministic():
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    assert torch.equal(coh(h), coh(h))


def test_wrong_last_dim_raises():
    coh = make()
    with pytest.raises(ValueError):
        coh(torch.randn(B, T, D_MODEL + 1))


# ── 2. clamp ─────────────────────────────────────────────────────────────

def test_clamp_binds_and_is_applied_before_F():
    coh = make()
    # phi_proj grande ⇒ sigmoid saturada por encima de r_max
    with torch.no_grad():
        coh.phi_proj.weight.fill_(50.0)
        coh.W_tau.weight.fill_(1.0)
    h = torch.ones(B, T, D_MODEL)

    st = coh._stages(h)
    assert (st["gate"] > coh.r_max).all(), "el test no forzó el clamp"
    assert torch.equal(st["ratio"], torch.full_like(st["ratio"], coh.r_max))

    assert st["scale"].max().item() == pytest.approx(S_MAX_FP32, rel=1e-6)


def test_scale_strictly_positive_on_moderate_input():
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    st = coh._stages(h)
    assert (st["scale"] > 0).all()


def test_scale_bounded_on_extreme_input():
    """
    Con logits muy negativos sigmoid puede saturar a 0 exacto y scale
    valer 0. Eso es admisible: la cota superior es lo que el contrato
    garantiza siempre.
    """
    coh = make()
    # La cota se evalúa en FP32, igual que el núcleo: en float64 da
    # 4.0251891 y en float32 da 4.0251918. La diferencia (~2.7e-6) es el
    # error de redondeo propio de 1/sqrt(1-r^2) con r=0.98, así que la
    # tolerancia tiene que ser relativa, no absoluta.
    s_max = S_MAX_FP32
    for mult in (1.0, 1e2, 1e4):
        st = coh._stages(torch.randn(B, T, D_MODEL) * mult)
        assert (st["scale"] >= 0).all()
        assert st["scale"].max().item() <= s_max * (1.0 + 1e-5)
        assert torch.isfinite(st["scale"]).all()


def test_direction_is_unit_norm():
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    n = coh._stages(h)["d_hat"].norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-6)


# ── 3. dtype y AMP ───────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_roundtrip(dtype):
    coh = make()
    h = torch.randn(B, T, D_MODEL).to(dtype)
    delta = coh(h)
    assert delta.dtype == dtype
    assert torch.isfinite(delta.float()).all()


def test_module_in_half_precision_still_computes_in_fp32():
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    ref = coh._stages(h)["z"]
    coh_half = make().half()
    # los pesos son fp16, pero el cálculo debe subir a fp32 sin error
    z = coh_half._stages(h.half())["z"]
    assert z.dtype == torch.float32
    assert ref.dtype == torch.float32


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16])
def test_autocast_does_not_change_result(amp_dtype):
    """
    Sin el bloque autocast(enabled=False) del núcleo, torch castearía las
    operaciones lineales a precisión baja y este test fallaría.
    """
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    outside = coh(h)
    with torch.autocast(device_type="cpu", dtype=amp_dtype):
        inside = coh(h)
    assert inside.dtype == outside.dtype
    assert torch.equal(inside, outside)


# ── 4. gradientes ────────────────────────────────────────────────────────

def test_backward_default_beta_frozen():
    coh = make()
    h = torch.randn(B, T, D_MODEL)
    coh(h).pow(2).mean().backward()
    assert coh.W_tau.weight.grad is not None
    assert coh.phi_proj.weight.grad is not None
    assert coh.out_proj.weight.grad is not None
    assert coh.beta.grad is None
    assert coh.trainable_beta is False


def test_backward_trainable_beta():
    coh = make(trainable_beta=True)
    h = torch.randn(B, T, D_MODEL)
    coh(h).pow(2).mean().backward()
    assert coh.beta.grad is not None
    assert torch.isfinite(coh.beta.grad).all()
    # grad is not None no basta: exigimos participación efectiva
    assert coh.beta.grad.abs().sum().item() > 0
    assert coh.trainable_beta is True


def test_trainable_beta_flag_does_not_change_initial_value():
    frozen = make(beta_init=0.5, trainable_beta=False)
    live = make(beta_init=0.5, trainable_beta=True)
    assert frozen.beta.item() == live.beta.item() == 0.5
    assert frozen.beta.requires_grad is False
    assert live.beta.requires_grad is True


def test_gradients_are_finite():
    coh = make(trainable_beta=True)
    h = torch.randn(B, T, D_MODEL)
    coh(h).pow(2).mean().backward()
    for name, p in coh.named_parameters():
        assert torch.isfinite(p.grad).all(), f"gradiente no finito en {name}"


# ── 5. estado y conteo ───────────────────────────────────────────────────

def test_state_dict_keys_exact():
    coh = make()
    assert set(coh.state_dict().keys()) == {
        "W_tau.weight",
        "phi_proj.weight",
        "out_proj.weight",
        "beta",
    }


def test_beta_persists_through_state_dict():
    a = make(beta_init=0.7)
    b = make(beta_init=0.5)
    b.load_state_dict(a.state_dict())
    assert b.beta.item() == pytest.approx(0.7)


def test_parameter_counts():
    d_model, d_tau = 960, 96
    coh = CoH(d_model, d_tau)
    total = 2 * d_model * d_tau + d_tau + 1
    trainable = 2 * d_model * d_tau + d_tau
    assert coh.num_parameters() == total == 184_417
    assert coh.num_parameters(trainable_only=True) == trainable == 184_416


def test_constructor_validation():
    with pytest.raises(ValueError):
        CoH(0, 8)
    with pytest.raises(ValueError):
        CoH(16, 0)
    with pytest.raises(ValueError):
        CoH(16, 8, r_max=1.0)
    with pytest.raises(ValueError):
        CoH(16, 8, r_max=0.0)


# ── 6. CUDA (opcional) ───────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin GPU")
def test_cuda_autocast_equivalence():
    coh = make().cuda()
    h = torch.randn(B, T, D_MODEL, device="cuda")
    outside = coh(h)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inside = coh(h)
    assert torch.equal(inside, outside)
