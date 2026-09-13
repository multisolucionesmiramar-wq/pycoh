"""
pycoh.integration.resolver -- turning evidence into a decision.

The inspector is allowed to pick the wrong candidate; the resolver is not
allowed to be wrong silently. On any ambiguity it fails with a message
listing the options and asking for `target_modules`. It never picks
`candidates[0]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch.nn as nn

from pycoh.integration.inspector import CandidateContainer, ModelInfo

__all__ = ["ResolvedTarget", "resolve_target", "resolve_layers"]

LayersSpec = Union[None, int, Sequence[int]]


@dataclass(frozen=True)
class ResolvedTarget:
    container: nn.ModuleList
    path: str
    block_indices: Tuple[int, ...]
    hidden_size: int


def resolve_hidden_size(info: ModelInfo, override: Optional[int]) -> int:
    if override is not None:
        if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {override!r}")
        return override
    if info.hidden_size is not None:
        return info.hidden_size
    raise ValueError(
        "Could not determine d_model: no explicit override was given and the "
        "model declares neither config.hidden_size nor config.d_model. Pass "
        "hidden_size=... It is never inferred from weight shapes."
    )


def resolve_layers(spec: LayersSpec, length: int) -> Tuple[int, ...]:
    if spec is None:
        return tuple(range(length))

    if isinstance(spec, bool):
        raise ValueError(f"layers does not accept a boolean, got {spec!r}")

    if isinstance(spec, int):
        if not 0 < spec <= length:
            raise ValueError(
                f"layers={spec} is out of range: it must satisfy 0 < N <= {length}"
            )
        return tuple(range(spec))

    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("layers cannot be an empty sequence")
        for i in spec:
            if isinstance(i, bool) or not isinstance(i, int):
                raise ValueError(f"non-integer layer index: {i!r}")
            if i < 0:
                raise ValueError(
                    f"negative layer index: {i}. Indexing from the end is not supported"
                )
            if i >= length:
                raise ValueError(
                    f"layer index {i} is out of range: the container holds {length} blocks"
                )
        if len(set(spec)) != len(spec):
            raise ValueError(f"layers contains duplicate indices: {list(spec)}")
        return tuple(sorted(spec))

    raise ValueError(
        f"layers must be None, an integer or a sequence of integers; got "
        f"{type(spec).__name__}"
    )


def _describe(candidates: Sequence[CandidateContainer]) -> str:
    return "\n".join(
        f"  - {c.path}  ({c.length}x {c.block_type.__name__}; " + "; ".join(c.evidence) + ")"
        for c in candidates
    )


def resolve_container(info: ModelInfo, target_modules: Optional[str]) -> CandidateContainer:
    if target_modules is not None:
        exact = [c for c in info.candidates if c.path == target_modules]
        if not exact:
            raise ValueError(
                f"target_modules={target_modules!r} does not match any "
                "homogeneous nn.ModuleList in the model. Candidates:\n"
                + (_describe(info.candidates) or "  (none)")
            )
        return exact[0]

    if not info.candidates:
        raise ValueError(
            "No homogeneous nn.ModuleList was found. Point at the container "
            "with target_modules=..."
        )

    # Structural criteria, in order. None of them uses the path name.
    viable = [c for c in info.candidates if not c.is_nested]
    if any(c.touches_hidden_size for c in viable):
        viable = [c for c in viable if c.touches_hidden_size]

    if len(viable) == 1:
        return viable[0]

    raise ValueError(
        "Automatic resolution is ambiguous: there are "
        f"{len(viable)} structurally equivalent containers. "
        "Pick one with target_modules=...\n" + _describe(viable)
    )


def resolve_target(
    info: ModelInfo,
    *,
    layers: LayersSpec = None,
    hidden_size: Optional[int] = None,
    target_modules: Optional[str] = None,
) -> ResolvedTarget:
    # Container first: if the model has no block stack, that is the problem
    # worth reporting, not the dimension.
    candidate = resolve_container(info, target_modules)
    resolved_hidden = resolve_hidden_size(info, hidden_size)
    indices = resolve_layers(layers, candidate.length)
    return ResolvedTarget(
        container=candidate.container,
        path=candidate.path,
        block_indices=indices,
        hidden_size=resolved_hidden,
    )
