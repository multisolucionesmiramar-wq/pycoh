"""
pycoh.integration.inspector -- structural discovery. Read only.

The inspector observes and produces evidence. It does not decide, does not
mutate, and does not raise on unusual topologies: if it finds nothing it
returns an empty candidate list and the resolver is the one that fails.

Discovery is structural. Candidates are never scored or filtered by
substrings of their path ("encoder", "layers", "block"): that works on the
architectures you already know and fails silently on every other one. The
criteria are properties of the module graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch.nn as nn

__all__ = ["CandidateContainer", "ModelInfo", "inspect_model"]


@dataclass(frozen=True)
class CandidateContainer:
    """A homogeneous `nn.ModuleList` that could be the block stack."""

    path: str
    container: nn.ModuleList
    blocks: Tuple[nn.Module, ...]
    block_type: Type[nn.Module]
    length: int
    touches_hidden_size: Optional[bool]
    is_nested: bool
    evidence: Tuple[str, ...]


@dataclass(frozen=True)
class ModelInfo:
    candidates: Tuple[CandidateContainer, ...]
    hidden_size: Optional[int]
    hidden_size_source: Optional[str]
    already_injected: bool
    injected_paths: Tuple[str, ...]


def _declared_hidden_size(model: nn.Module) -> Tuple[Optional[int], Optional[str]]:
    """
    Declarative sources only, in the order fixed by the contract. The value
    is never inferred from the shape of a weight: a guessed dimension can be
    right today and hide an incompatible architecture tomorrow.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None, None
    for attr in ("hidden_size", "d_model"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value, f"config.{attr}"
    return None, None


def _is_homogeneous(container: nn.ModuleList) -> bool:
    if len(container) == 0:
        return False
    first = type(container[0])
    return all(type(m) is first for m in container)


def _touches(block: nn.Module, hidden_size: int) -> bool:
    """Does any parameter of the block operate on the residual dimension?"""
    return any(hidden_size in tuple(p.shape) for p in block.parameters())


def inspect_model(model: nn.Module) -> ModelInfo:
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")

    # Local import: avoids a cycle with integration.wrapper.
    from pycoh.integration.wrapper import CoHBlockWrapper

    injected = tuple(
        path or "<root>"
        for path, module in model.named_modules()
        if isinstance(module, CoHBlockWrapper)
    )

    hidden_size, hs_source = _declared_hidden_size(model)

    raw: list[tuple[str, nn.ModuleList]] = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and _is_homogeneous(module)
    ]
    paths = [p for p, _ in raw]

    candidates = []
    for path, container in raw:
        blocks = tuple(container)
        nested = any(
            path.startswith(other + ".") for other in paths if other and other != path
        )
        touches = _touches(blocks[0], hidden_size) if hidden_size is not None else None

        evidence = [
            f"homogeneous nn.ModuleList of {len(blocks)}x {type(blocks[0]).__name__}",
            "nested inside another candidate" if nested else "outermost container",
        ]
        if touches is True:
            evidence.append(f"its parameters operate on d_model={hidden_size}")
        elif touches is False:
            evidence.append(f"no parameter operates on d_model={hidden_size}")
        else:
            evidence.append("d_model not declared: not verifiable")

        candidates.append(
            CandidateContainer(
                path=path,
                container=container,
                blocks=blocks,
                block_type=type(blocks[0]),
                length=len(blocks),
                touches_hidden_size=touches,
                is_nested=nested,
                evidence=tuple(evidence),
            )
        )

    return ModelInfo(
        candidates=tuple(candidates),
        hidden_size=hidden_size,
        hidden_size_source=hs_source,
        already_injected=bool(injected),
        injected_paths=injected,
    )
