"""
Suite F1 — contrato del wrapper.

Ejecutar:  pytest -q tests/test_wrapper.py
"""

from collections import OrderedDict, namedtuple

import pytest
import torch
import torch.nn as nn

from pycoh.core.coh import CoH
from pycoh.integration.wrapper import CoHBlockWrapper

D_MODEL, D_TAU, B, T = 16, 8, 2, 5
SEED = 0


class RecordingBlock(nn.Module):
    """
    Bloque no trivial que además registra exactamente con qué lo llamaron.
    La transformación es deliberadamente no lineal y no conmutativa para
    que B(h) + CoH(h) no pueda confundirse con B(h + CoH(h)).
    """

    def __init__(self, d_model: int, mode: str = "tensor") -> None:
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
        self.mode = mode
        self.seen_args = None
        self.seen_kwargs = None
        # referencias estables: si se recrearan en cada forward, el test
        # de identidad compararía contra objetos nuevos
        self.extras = (torch.arange(4), {"cache": 1})

    def forward(self, hidden_states, *args, **kwargs):
        self.seen_args = args
        self.seen_kwargs = kwargs
        h = torch.tanh(self.lin(hidden_states)) * 3.0
        if self.mode == "tensor":
            return h
        if self.mode == "tuple":
            return (h,) + self.extras
        if self.mode == "dict":
            return {"hidden_states": h}
        if self.mode == "odict":
            return OrderedDict(hidden_states=h)
        if self.mode == "list":
            return [h]
        if self.mode == "empty_tuple":
            return ()
        if self.mode == "bad_tuple":
            return ("no soy un tensor", h)
        raise AssertionError(self.mode)


class ZeroBlock(nn.Module):
    """Anula su entrada: la salida del wrapper debe ser exactamente delta."""

    def forward(self, hidden_states, *args, **kwargs):
        return torch.zeros_like(hidden_states)


Output = namedtuple("Output", ["hidden_states", "cache"])


class NamedTupleBlock(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return Output(hidden_states=hidden_states * 2.0, cache=torch.arange(3))


def build(mode: str = "tensor"):
    torch.manual_seed(SEED)
    block = RecordingBlock(D_MODEL, mode)
    coh = CoH(D_MODEL, D_TAU)
    return CoHBlockWrapper(block, coh), block, coh


# ── 1. composición ───────────────────────────────────────────────────────

def test_tensor_output_is_block_plus_coh():
    w, block, coh = build("tensor")
    h = torch.randn(B, T, D_MODEL)
    got = w(h)
    expected = block(h) + coh(h)
    assert torch.equal(got, expected)
    assert got.shape == h.shape


def test_coh_reads_the_input_not_the_block_output():
    """
    Con un bloque que anula su entrada, la salida debe ser exactamente
    CoH(h). Si el wrapper calculase CoH(B(h)) el resultado sería CoH(0),
    que es distinto.
    """
    torch.manual_seed(SEED)
    coh = CoH(D_MODEL, D_TAU)
    w = CoHBlockWrapper(ZeroBlock(), coh)
    h = torch.randn(B, T, D_MODEL)
    assert torch.equal(w(h), coh(h))
    assert not torch.equal(coh(h), coh(torch.zeros_like(h)))


def test_output_dtype_matches_input():
    w, _, _ = build("tensor")
    h = torch.randn(B, T, D_MODEL)
    assert w(h).dtype == h.dtype


# ── 2. tuplas ────────────────────────────────────────────────────────────

def test_tuple_output_preserves_tail_by_reference():
    w, block, coh = build("tuple")
    h = torch.randn(B, T, D_MODEL)
    out = w(h)
    assert isinstance(out, tuple) and len(out) == 3
    assert torch.equal(out[0], block(h)[0] + coh(h))
    # identidad de referencia, no igualdad de valor
    assert out[1] is block.extras[0]
    assert out[2] is block.extras[1]


def test_namedtuple_type_is_preserved():
    torch.manual_seed(SEED)
    coh = CoH(D_MODEL, D_TAU)
    w = CoHBlockWrapper(NamedTupleBlock(), coh)
    h = torch.randn(B, T, D_MODEL)
    out = w(h)
    assert type(out) is Output
    assert torch.equal(out.hidden_states, h * 2.0 + coh(h))


# ── 3. rechazo explícito ─────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["dict", "odict", "list", "empty_tuple", "bad_tuple"])
def test_unsupported_output_raises_typeerror(mode):
    w, _, _ = build(mode)
    with pytest.raises(TypeError):
        w(torch.randn(B, T, D_MODEL))


def test_missing_hidden_state_raises():
    w, _, _ = build("tensor")
    with pytest.raises(TypeError):
        w(attention_mask=torch.ones(B, T))


def test_non_tensor_hidden_state_raises():
    w, _, _ = build("tensor")
    with pytest.raises(TypeError):
        w("no soy un tensor")


def test_constructor_type_validation():
    coh = CoH(D_MODEL, D_TAU)
    with pytest.raises(TypeError):
        CoHBlockWrapper("no soy un módulo", coh)
    with pytest.raises(TypeError):
        CoHBlockWrapper(nn.Identity(), "no soy CoH")


# ── 4. transparencia de ruteo ────────────────────────────────────────────

def test_args_and_kwargs_reach_the_block_untouched():
    w, block, _ = build("tensor")
    h = torch.randn(B, T, D_MODEL)
    extra_positional = torch.arange(3)
    mask = torch.ones(B, T)
    pos = torch.arange(T)
    sentinel = object()

    w(h, extra_positional, attention_mask=mask, position_ids=pos, use_cache=sentinel)

    assert block.seen_args[0] is extra_positional
    assert block.seen_kwargs["attention_mask"] is mask
    assert block.seen_kwargs["position_ids"] is pos
    assert block.seen_kwargs["use_cache"] is sentinel
    assert set(block.seen_kwargs) == {"attention_mask", "position_ids", "use_cache"}


def test_hidden_state_as_keyword_works():
    w, block, coh = build("tensor")
    h = torch.randn(B, T, D_MODEL)
    got = w(hidden_states=h, attention_mask=None)
    # capturar antes de volver a llamar al bloque: la segunda llamada
    # sobrescribe lo registrado
    seen = dict(block.seen_kwargs)
    assert seen == {"attention_mask": None}
    assert torch.equal(got, block(h) + coh(h))


# ── 5. estructura y gradientes ───────────────────────────────────────────

def test_structural_identity():
    w, block, coh = build("tensor")
    assert isinstance(w, CoHBlockWrapper)
    assert w.block is block
    assert w.coh is coh
    found = [m for m in w.modules() if isinstance(m, CoHBlockWrapper)]
    assert found == [w]


def test_gradients_flow_to_both_block_and_coh():
    w, block, coh = build("tensor")
    h = torch.randn(B, T, D_MODEL)
    w(h).pow(2).mean().backward()
    assert block.lin.weight.grad is not None
    assert coh.W_tau.weight.grad is not None
    assert coh.out_proj.weight.grad is not None
    assert coh.beta.grad is None  # congelado por defecto


def test_wrapper_adds_no_parameters_of_its_own():
    w, block, coh = build("tensor")
    own = set(id(p) for p in w.parameters())
    inner = set(id(p) for p in list(block.parameters()) + list(coh.parameters()))
    assert own == inner
