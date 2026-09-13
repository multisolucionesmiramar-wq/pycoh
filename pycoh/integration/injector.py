"""
pycoh.integration.injector -- the only component that mutates the graph.

It makes no topological decisions: it receives a `ResolvedTarget` and
applies it. Injection is atomic -- every wrapper is built first and only
then assigned, so a failure halfway through never leaves the model in a
hybrid state.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from pycoh.core.coh import CoH
from pycoh.integration.inspector import inspect_model
from pycoh.integration.resolver import ResolvedTarget, resolve_target
from pycoh.integration.wrapper import CoHBlockWrapper

__all__ = ["inject_coh", "freeze_base", "apply_coh", "remove_coh"]

LayersSpec = Union[None, int, Sequence[int]]


def _module_device(module: nn.Module) -> Optional[torch.device]:
    """
    Device a module lives on, or None if it holds no tensors.

    Each CoH is created where its wrapped block lives, not where the rest of
    the model lives. With a model sharded across several GPUs, every adapter
    lands next to its own block.
    """
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return None


def inject_coh(
    target: ResolvedTarget,
    *,
    d_tau: int,
    beta_init: float = 0.5,
    r_max: float = 0.98,
    trainable_beta: bool = False,
) -> int:
    """Replace the selected blocks with wrappers. Returns how many."""
    wrappers = {}
    for idx in target.block_indices:
        original = target.container[idx]
        if isinstance(original, CoHBlockWrapper):
            raise RuntimeError(
                f"block {target.path}[{idx}] is already wrapped by CoH"
            )

        coh = CoH(
            d_model=target.hidden_size,
            d_tau=d_tau,
            beta_init=beta_init,
            r_max=r_max,
            trainable_beta=trainable_beta,
        )

        # The device is inherited from the block; the dtype is NOT. CoH stays
        # in FP32 even when the base model is in bf16 or fp16, which is
        # precisely the core's precision policy.
        device = _module_device(original)
        if device is not None:
            coh = coh.to(device)

        wrappers[idx] = CoHBlockWrapper(original, coh)

    # Mutate only after everything has been built.
    for idx, wrapper in wrappers.items():
        target.container[idx] = wrapper
    return len(wrappers)


def freeze_base(model: nn.Module) -> None:
    """
    Freeze everything that does not belong to a CoH, structurally rather
    than by name. CoH parameters are left exactly as the core built them, so
    `beta` keeps the `requires_grad` set by `trainable_beta`: sweeping
    everything to False and re-enabling afterwards would erase that intent.
    """
    coh_params = {
        id(p)
        for module in model.modules()
        if isinstance(module, CoHBlockWrapper)
        for p in module.coh.parameters()
    }
    for p in model.parameters():
        if id(p) not in coh_params:
            p.requires_grad_(False)


def apply_coh(
    model: nn.Module,
    *,
    d_tau: int,
    layers: LayersSpec = None,
    hidden_size: Optional[int] = None,
    target_modules: Optional[str] = None,
    beta_init: float = 0.5,
    r_max: float = 0.98,
    trainable_beta: bool = False,
    freeze: bool = True,
) -> nn.Module:
    """
    Apply CoH to a model, in place, and return the same object.

    Orchestrates inspect -> resolve -> validate -> inject -> freeze. On an
    unsupported architecture, an unresolvable dimension, invalid indices,
    ambiguity or double injection it raises without leaving the model
    partially modified.

    `d_tau` is required: there is no automatic value.
    """
    info = inspect_model(model)

    if info.already_injected:
        raise RuntimeError(
            "The model already has CoH applied at: "
            + ", ".join(info.injected_paths)
            + ". A second application would produce nested wrappers."
        )

    target = resolve_target(
        info,
        layers=layers,
        hidden_size=hidden_size,
        target_modules=target_modules,
    )

    inject_coh(
        target,
        d_tau=d_tau,
        beta_init=beta_init,
        r_max=r_max,
        trainable_beta=trainable_beta,
    )

    if freeze:
        freeze_base(model)

    return model


def remove_coh(model: nn.Module, *, unfreeze: bool = False) -> nn.Module:
    """
    Uninstall CoH and restore the original topology, in place.

    Every `CoHBlockWrapper` is replaced by the block it wrapped -- the same
    object, not a copy -- so `named_modules()` and `state_dict()` go back to
    exactly what the clean model had and the standard HuggingFace APIs
    (`save_pretrained`) work again.

    It restores no weights: the base model was never modified, so there is
    nothing to restore. CoH state is lost unless it was saved beforehand
    with `save_adapter`.

    `requires_grad` is left untouched by default. `apply_coh` froze the base
    and there is no record of the previous state here, so guessing would be
    worse than being explicit: pass `unfreeze=True` to re-enable every
    remaining parameter.

    Raises `RuntimeError` if the model has no CoH: removing something that
    is not there is almost always a caller error, not a no-op.
    """
    removed = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, CoHBlockWrapper):
                setattr(parent, name, child.block)
                removed += 1

    if removed == 0:
        raise RuntimeError("The model has no CoH applied: there is nothing to remove.")

    if unfreeze:
        for p in model.parameters():
            p.requires_grad_(True)

    return model
