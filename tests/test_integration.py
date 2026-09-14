"""
Suite F2 -- discovery, resolution and injection.

Run with:  pytest -q tests/test_integration.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pycoh.core.coh import CoH
from pycoh.integration.injector import apply_coh, freeze_base
from pycoh.integration.inspector import inspect_model
from pycoh.integration.resolver import resolve_layers, resolve_target
from pycoh.integration.wrapper import CoHBlockWrapper

D_MODEL, D_TAU, N_LAYERS = 16, 8, 4


class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, *args, **kwargs):
        return torch.tanh(self.mlp(self.attn(hidden_states)))


class TinyModel(nn.Module):
    """Minimal synthetic model shaped like a transformer."""

    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, declare="hidden_size"):
        super().__init__()
        self.embed = nn.Embedding(32, d_model)
        self.layers = nn.ModuleList(Block(d_model) for _ in range(n_layers))
        self.head = nn.Linear(d_model, 32)
        if declare is None:
            self.config = SimpleNamespace()
        else:
            self.config = SimpleNamespace(**{declare: d_model})

    def forward(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class TwoStackModel(nn.Module):
    """Two sibling stacks, structurally indistinguishable."""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.stack_a = nn.ModuleList(Block(d_model) for _ in range(3))
        self.stack_b = nn.ModuleList(Block(d_model) for _ in range(3))
        self.config = SimpleNamespace(hidden_size=d_model)


class NestedModel(nn.Module):
    """The outer stack holds blocks that themselves own a ModuleList."""

    class Expert(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.lin = nn.Linear(d_model, d_model)

    class MoEBlock(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.experts = nn.ModuleList(
                NestedModel.Expert(d_model) for _ in range(2)
            )

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states + self.experts[0].lin(hidden_states)

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.layers = nn.ModuleList(self.MoEBlock(d_model) for _ in range(3))
        self.config = SimpleNamespace(hidden_size=d_model)


# -- 1. Inspector: it only observes ---------------------------------------

def test_inspector_finds_homogeneous_container():
    info = inspect_model(TinyModel())
    paths = [c.path for c in info.candidates]
    assert "layers" in paths
    c = next(c for c in info.candidates if c.path == "layers")
    assert c.length == N_LAYERS
    assert c.block_type is Block
    assert c.touches_hidden_size is True
    assert c.is_nested is False
    assert len(c.evidence) >= 3


def test_inspector_does_not_mutate():
    model = TinyModel()
    before = [id(m) for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    inspect_model(model)
    assert [id(m) for m in model.modules()] == before
    assert [p.requires_grad for p in model.parameters()] == flags


def test_inspector_reports_hidden_size_sources():
    assert inspect_model(TinyModel(declare="hidden_size")).hidden_size_source == "config.hidden_size"
    assert inspect_model(TinyModel(declare="d_model")).hidden_size_source == "config.d_model"
    info = inspect_model(TinyModel(declare=None))
    assert info.hidden_size is None and info.hidden_size_source is None


def test_inspector_marks_nested_containers():
    info = inspect_model(NestedModel())
    outer = next(c for c in info.candidates if c.path == "layers")
    inner = [c for c in info.candidates if c.path.startswith("layers.")]
    assert outer.is_nested is False
    assert inner and all(c.is_nested for c in inner)


def test_inspector_reports_existing_injection_without_raising():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    info = inspect_model(model)
    assert info.already_injected is True
    assert len(info.injected_paths) == N_LAYERS


# -- 2. Resolver: decide or fail ------------------------------------------

def test_resolver_picks_outermost_unambiguous():
    target = resolve_target(inspect_model(NestedModel()))
    assert target.path == "layers"
    assert target.block_indices == (0, 1, 2)


def test_resolver_rejects_ambiguity():
    info = inspect_model(TwoStackModel())
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_target(info)
    # with an explicit override it does resolve
    assert resolve_target(info, target_modules="stack_b").path == "stack_b"


def test_resolver_rejects_unknown_target_modules():
    with pytest.raises(ValueError):
        resolve_target(inspect_model(TinyModel()), target_modules="no_existe")


def test_resolver_requires_hidden_size():
    info = inspect_model(TinyModel(declare=None))
    with pytest.raises(ValueError, match="Could not determine d_model"):
        resolve_target(info)
    assert resolve_target(info, hidden_size=D_MODEL).hidden_size == D_MODEL


@pytest.mark.parametrize(
    "spec",
    [[-1], [N_LAYERS], [0, 0], [], 0, -3, N_LAYERS + 1, [1.5], [True], "todas", 2.0],
)
def test_layers_validation_rejects(spec):
    with pytest.raises(ValueError):
        resolve_layers(spec, N_LAYERS)


def test_layers_validation_accepts():
    assert resolve_layers(None, 4) == (0, 1, 2, 3)
    assert resolve_layers(2, 4) == (0, 1)
    assert resolve_layers([3, 0], 4) == (0, 3)
    assert resolve_layers((1,), 4) == (1,)


# -- 3. Injector: it mutates, atomically ----------------------------------

def test_injection_wraps_selected_layers_only():
    model = apply_coh(TinyModel(), d_tau=D_TAU, layers=[0, 2])
    kinds = [isinstance(m, CoHBlockWrapper) for m in model.layers]
    assert kinds == [True, False, True, False]


def test_apply_coh_returns_same_object():
    model = TinyModel()
    assert apply_coh(model, d_tau=D_TAU) is model


def test_double_injection_rejected():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    with pytest.raises(RuntimeError, match="already has CoH"):
        apply_coh(model, d_tau=D_TAU)
    # and the previous topology is intact: a single wrapper level
    assert all(isinstance(m, CoHBlockWrapper) for m in model.layers)
    assert all(not isinstance(m.block, CoHBlockWrapper) for m in model.layers)


def test_no_partial_mutation_on_invalid_layers():
    model = TinyModel()
    with pytest.raises(ValueError):
        apply_coh(model, d_tau=D_TAU, layers=[0, 1, 99])
    assert all(isinstance(m, Block) for m in model.layers)
    assert all(p.requires_grad for p in model.parameters())


def test_no_partial_mutation_on_ambiguity():
    model = TwoStackModel()
    with pytest.raises(ValueError):
        apply_coh(model, d_tau=D_TAU)
    assert not any(isinstance(m, CoHBlockWrapper) for m in model.modules())


# -- 4. Freezing ----------------------------------------------------------

def test_base_frozen_and_coh_trainable_by_identity():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    coh_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, CoHBlockWrapper)
        for p in m.coh.parameters()
    }
    for p in model.parameters():
        if id(p) in coh_ids:
            continue
        assert p.requires_grad is False

    for m in model.modules():
        if isinstance(m, CoHBlockWrapper):
            assert m.coh.W_tau.weight.requires_grad is True
            assert m.coh.phi_proj.weight.requires_grad is True
            assert m.coh.out_proj.weight.requires_grad is True
            assert m.coh.beta.requires_grad is False


def test_trainable_beta_survives_freezing():
    model = apply_coh(TinyModel(), d_tau=D_TAU, trainable_beta=True)
    for m in model.modules():
        if isinstance(m, CoHBlockWrapper):
            assert m.coh.beta.requires_grad is True


def test_freeze_can_be_skipped():
    model = apply_coh(TinyModel(), d_tau=D_TAU, freeze=False)
    assert model.head.weight.requires_grad is True


# -- 5. Forward, backward and exact counts --------------------------------

def test_forward_and_backward_after_injection():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    ids = torch.randint(0, 32, (2, 5))
    model(ids).pow(2).mean().backward()

    for m in model.modules():
        if isinstance(m, CoHBlockWrapper):
            assert m.coh.W_tau.weight.grad is not None
            assert m.coh.out_proj.weight.grad is not None
    assert model.head.weight.grad is None  # frozen


def test_reference_parameter_counts():
    """
    d_model=960, d_tau=96, 32 layers: the contract numbers. A synthetic
    model shaped like the reference target, so the suite does not depend on
    the network.
    """
    d_model, d_tau, n_layers = 960, 96, 32
    model = apply_coh(
        TinyModel(d_model=d_model, n_layers=n_layers), d_tau=d_tau
    )

    coh_modules = [
        m.coh for m in model.modules() if isinstance(m, CoHBlockWrapper)
    ]
    assert len(coh_modules) == n_layers

    per_layer_total = 2 * d_model * d_tau + d_tau + 1
    per_layer_train = 2 * d_model * d_tau + d_tau
    assert per_layer_total == 184_417
    assert per_layer_train == 184_416

    total = sum(p.numel() for c in coh_modules for p in c.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == 5_901_344
    assert trainable == 5_901_312


def test_injected_coh_modules_are_distinct_instances():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    cohs = [m.coh for m in model.layers]
    assert len({id(c) for c in cohs}) == N_LAYERS
    assert all(isinstance(c, CoH) for c in cohs)
    # independent weights, not shared
    assert not torch.equal(cohs[0].W_tau.weight, cohs[1].W_tau.weight)


def test_freeze_base_is_idempotent():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    before = [p.requires_grad for p in model.parameters()]
    freeze_base(model)
    assert [p.requires_grad for p in model.parameters()] == before


# -- 6. device placement --------------------------------------------------

def test_module_device_helper():
    from pycoh.integration.injector import _module_device

    assert _module_device(nn.Linear(4, 4)) == torch.device("cpu")
    assert _module_device(nn.Identity()) is None


def test_coh_lands_on_the_block_device_cpu():
    model = apply_coh(TinyModel(), d_tau=D_TAU)
    for w in model.layers:
        dev_bloque = next(w.block.parameters()).device
        for p in w.coh.parameters():
            assert p.device == dev_bloque


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_coh_follows_model_already_on_gpu():
    """
    The case that used to break: move the model to GPU and inject
    afterwards. Without device inheritance, CoH stayed on CPU and the
    forward failed with 'Expected all tensors to be on the same device'.
    """
    model = TinyModel().cuda()
    apply_coh(model, d_tau=D_TAU)
    for w in model.layers:
        for p in w.coh.parameters():
            assert p.is_cuda
    ids = torch.randint(0, 32, (2, 5), device="cuda")
    assert model(ids).is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_adapter_round_trip_across_devices(tmp_path):
    """Save from GPU, load on GPU: the file tensors travel through CPU."""
    from pycoh.integration.serialization import load_adapter, save_adapter

    origen = apply_coh(TinyModel().cuda(), d_tau=D_TAU)
    with torch.no_grad():
        for w in origen.layers:
            for p in w.coh.parameters():
                p.copy_(torch.randn_like(p))
    f = tmp_path / "a.pt"
    save_adapter(origen, f)

    destino = apply_coh(TinyModel().cuda(), d_tau=D_TAU)
    load_adapter(destino, f)
    for a, b in zip(origen.layers, destino.layers):
        for pa, pb in zip(a.coh.parameters(), b.coh.parameters()):
            assert pb.is_cuda
            assert torch.equal(pa, pb)
