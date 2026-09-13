"""
pycoh.integration.resolver — convierte evidencia en decisión.

El Inspector puede equivocarse de candidato; el Resolver no puede
equivocarse en silencio. Ante cualquier ambigüedad, falla con un mensaje
que enumera las opciones y pide `target_modules`. Nunca escoge
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
            raise ValueError(f"hidden_size debe ser un entero positivo, recibido {override!r}")
        return override
    if info.hidden_size is not None:
        return info.hidden_size
    raise ValueError(
        "No se pudo determinar d_model: no hay override explícito y el modelo "
        "no declara config.hidden_size ni config.d_model. Pásalo con "
        "hidden_size=... No se infiere de la forma de los pesos."
    )


def resolve_layers(spec: LayersSpec, length: int) -> Tuple[int, ...]:
    if spec is None:
        return tuple(range(length))

    if isinstance(spec, bool):
        raise ValueError(f"layers no admite un booleano, recibido {spec!r}")

    if isinstance(spec, int):
        if not 0 < spec <= length:
            raise ValueError(
                f"layers={spec} fuera de rango: debe cumplir 0 < N <= {length}"
            )
        return tuple(range(spec))

    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("layers no puede ser una secuencia vacía")
        for i in spec:
            if isinstance(i, bool) or not isinstance(i, int):
                raise ValueError(f"índice de capa no entero: {i!r}")
            if i < 0:
                raise ValueError(f"índice de capa negativo: {i}. No se admite indexación desde el final")
            if i >= length:
                raise ValueError(f"índice de capa {i} fuera de rango: el contenedor tiene {length} bloques")
        if len(set(spec)) != len(spec):
            raise ValueError(f"layers contiene índices duplicados: {list(spec)}")
        return tuple(sorted(spec))

    raise ValueError(
        f"layers debe ser None, un entero o una secuencia de enteros; recibido {type(spec).__name__}"
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
                f"target_modules={target_modules!r} no corresponde a ningún "
                "nn.ModuleList homogéneo del modelo. Candidatos:\n"
                + (_describe(info.candidates) or "  (ninguno)")
            )
        return exact[0]

    if not info.candidates:
        raise ValueError(
            "No se encontró ningún nn.ModuleList homogéneo. Indica el "
            "contenedor con target_modules=..."
        )

    # Criterios estructurales, en orden. Ninguno usa el nombre del path.
    viable = [c for c in info.candidates if not c.is_nested]
    if any(c.touches_hidden_size for c in viable):
        viable = [c for c in viable if c.touches_hidden_size]

    if len(viable) == 1:
        return viable[0]

    raise ValueError(
        "La resolución automática es ambigua: hay "
        f"{len(viable)} contenedores estructuralmente equivalentes. "
        "Elige uno con target_modules=...\n" + _describe(viable)
    )


def resolve_target(
    info: ModelInfo,
    *,
    layers: LayersSpec = None,
    hidden_size: Optional[int] = None,
    target_modules: Optional[str] = None,
) -> ResolvedTarget:
    # El contenedor primero: si el modelo no tiene una pila de bloques, ese
    # es el problema que hay que reportar, no la dimensión.
    candidate = resolve_container(info, target_modules)
    resolved_hidden = resolve_hidden_size(info, hidden_size)
    indices = resolve_layers(layers, candidate.length)
    return ResolvedTarget(
        container=candidate.container,
        path=candidate.path,
        block_indices=indices,
        hidden_size=resolved_hidden,
    )
