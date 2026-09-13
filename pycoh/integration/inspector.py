"""
pycoh.integration.inspector — descubrimiento estructural. Solo lectura.

El Inspector observa y produce evidencia. No decide, no muta, no lanza
excepciones por topologías extrañas: si no encuentra nada devuelve una
lista vacía de candidatos y el Resolver se encarga de fallar.

El descubrimiento es estructural. No se puntúa ni se filtra por
substrings del path ("encoder", "layers", "block"): eso acierta en las
arquitecturas que uno ya conoce y falla en silencio en las demás. Los
criterios son propiedades del grafo de módulos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch.nn as nn

__all__ = ["CandidateContainer", "ModelInfo", "inspect_model"]


@dataclass(frozen=True)
class CandidateContainer:
    """Un `nn.ModuleList` homogéneo que podría ser la pila de bloques."""

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
    Fuentes declarativas únicamente, en el orden fijado por el contrato.
    Nunca se infiere de la forma de un peso: una dimensión adivinada puede
    ser correcta hoy y ocultar una arquitectura incompatible mañana.
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
    """¿Algún parámetro del bloque opera sobre la dimensión del residual?"""
    return any(hidden_size in tuple(p.shape) for p in block.parameters())


def inspect_model(model: nn.Module) -> ModelInfo:
    if not isinstance(model, nn.Module):
        raise TypeError(f"model debe ser nn.Module, recibido {type(model).__name__}")

    # Import local: evita un ciclo con integration.wrapper.
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
            f"nn.ModuleList homogéneo de {len(blocks)}x {type(blocks[0]).__name__}",
            "anidado dentro de otro candidato" if nested else "contenedor más externo",
        ]
        if touches is True:
            evidence.append(f"sus parámetros operan sobre d_model={hidden_size}")
        elif touches is False:
            evidence.append(f"ningún parámetro opera sobre d_model={hidden_size}")
        else:
            evidence.append("d_model no declarado: no verificable")

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
