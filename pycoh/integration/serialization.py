"""
pycoh.integration.serialization — guardar y cargar adaptadores.

Un adaptador contiene exclusivamente el estado de los módulos CoH. No
contiene un solo peso del modelo huésped: para reconstruir hay que partir
del checkpoint base, aplicar `apply_coh` y después cargar el adaptador.

La extracción es estructural (`isinstance(module, CoHBlockWrapper)`), no
por filtrado de nombres: filtrar por la subcadena ".coh." funcionaría hoy
y volvería a atarnos a los nombres, que es justo lo que F2 eliminó.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from pycoh.integration.wrapper import CoHBlockWrapper

__all__ = ["save_adapter", "load_adapter", "adapter_metadata", "FORMAT_VERSION"]

FORMAT_VERSION = 1
MECHANISM = "coh"


def _collect(model: nn.Module) -> List[Tuple[str, CoHBlockWrapper]]:
    found = [
        (path, module)
        for path, module in model.named_modules()
        if isinstance(module, CoHBlockWrapper)
    ]
    if not found:
        raise RuntimeError(
            "El modelo no tiene CoH aplicado. Llama a apply_coh(...) antes."
        )
    return found


def _shared_config(wrappers: List[Tuple[str, CoHBlockWrapper]]) -> Dict[str, Any]:
    """
    El adaptador declara una sola configuración, así que exige que todos
    los CoH la compartan. Si alguien construyó a mano capas con distinto
    d_tau, se falla en vez de registrar el valor de la primera.
    """
    configs = {
        path: (w.coh.d_model, w.coh.d_tau, w.coh.r_max, w.coh.trainable_beta)
        for path, w in wrappers
    }
    distintas = set(configs.values())
    if len(distintas) > 1:
        detalle = "\n".join(f"  {p}: d_model={c[0]}, d_tau={c[1]}, r_max={c[2]}, trainable_beta={c[3]}"
                            for p, c in configs.items())
        raise ValueError(
            "Los módulos CoH del modelo no comparten configuración y el "
            "formato de adaptador declara una sola:\n" + detalle
        )
    d_model, d_tau, r_max, trainable_beta = distintas.pop()
    return {
        "d_model": d_model,
        "d_tau": d_tau,
        "r_max": r_max,
        "trainable_beta": trainable_beta,
    }


def save_adapter(model: nn.Module, path: str | os.PathLike) -> Dict[str, Any]:
    """
    Guarda el estado CoH del modelo. Devuelve los metadatos escritos.

    El archivo resultante pesa lo que pesa CoH (unos 24 MB con d_model=960,
    d_tau=96 y 32 capas), no lo que pesa el modelo.
    """
    wrappers = _collect(model)
    config = _shared_config(wrappers)

    state: Dict[str, torch.Tensor] = {}
    for wrapper_path, wrapper in wrappers:
        for key, tensor in wrapper.coh.state_dict().items():
            state[f"{wrapper_path}.{key}"] = tensor.detach().cpu().clone()

    metadata = {
        "format_version": FORMAT_VERSION,
        "mechanism": MECHANISM,
        "injected_paths": [p for p, _ in wrappers],
        **config,
    }

    torch.save({"metadata": metadata, "state_dict": state}, path)
    return metadata


def adapter_metadata(path: str | os.PathLike) -> Dict[str, Any]:
    """Lee solo los metadatos, sin tocar ningún modelo."""
    return _read_payload(path)["metadata"]


def _read_payload(path: str | os.PathLike) -> Dict[str, Any]:
    # weights_only=True impide que un archivo descargado ejecute código
    # arbitrario al deserializarse. Los metadatos son tipos simples, así
    # que pasan el filtro sin problema.
    payload = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(payload, dict) or "metadata" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path} no tiene la forma de un adaptador PyCoH")

    meta = payload["metadata"]
    if meta.get("mechanism") != MECHANISM:
        raise ValueError(
            f"mecanismo desconocido: {meta.get('mechanism')!r}, se esperaba {MECHANISM!r}"
        )
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"format_version {version!r} no soportada por esta versión de PyCoH "
            f"(soporta {FORMAT_VERSION})"
        )
    return payload


def load_adapter(model: nn.Module, path: str | os.PathLike) -> Dict[str, Any]:
    """
    Restaura el estado CoH sobre un modelo que YA tiene CoH aplicado.

    No modifica la topología ni crea módulos: si el modelo no fue
    inyectado, o lo fue con otra configuración o en otras capas, falla sin
    tocar nada. La validación es completa antes de escribir el primer
    tensor, de modo que un adaptador incompatible no deja el modelo a
    medio cargar.

    `trainable_beta` del archivo es informativo: describe cómo se entrenó,
    no cambia lo que el usuario configuró en este modelo.
    """
    payload = _read_payload(path)
    meta = payload["metadata"]
    state = payload["state_dict"]

    wrappers = _collect(model)
    current_paths = [p for p, _ in wrappers]
    saved_paths = list(meta["injected_paths"])

    if current_paths != saved_paths:
        faltan = [p for p in saved_paths if p not in current_paths]
        sobran = [p for p in current_paths if p not in saved_paths]
        raise RuntimeError(
            "La topología del adaptador no coincide con la del modelo.\n"
            f"  adaptador: {len(saved_paths)} capas\n"
            f"  modelo:    {len(current_paths)} capas\n"
            + (f"  en el adaptador y no en el modelo: {faltan}\n" if faltan else "")
            + (f"  en el modelo y no en el adaptador: {sobran}\n" if sobran else "")
            + "Aplica apply_coh con las mismas capas antes de cargar."
        )

    config = _shared_config(wrappers)
    for campo in ("d_model", "d_tau", "r_max"):
        if meta[campo] != config[campo]:
            raise RuntimeError(
                f"{campo} incompatible: el adaptador declara {meta[campo]!r} y "
                f"el modelo tiene {config[campo]!r}"
            )

    # Reparto y validación completos ANTES de escribir nada.
    por_modulo: Dict[str, Dict[str, torch.Tensor]] = {}
    for wrapper_path, wrapper in wrappers:
        prefijo = wrapper_path + "."
        local = {
            k[len(prefijo):]: v for k, v in state.items() if k.startswith(prefijo)
        }
        esperado = wrapper.coh.state_dict()

        if set(local) != set(esperado):
            faltan = sorted(set(esperado) - set(local))
            sobran = sorted(set(local) - set(esperado))
            raise RuntimeError(
                f"claves incompatibles en {wrapper_path}: "
                f"faltan {faltan}, sobran {sobran}"
            )
        for k, v in local.items():
            if tuple(v.shape) != tuple(esperado[k].shape):
                raise RuntimeError(
                    f"forma incompatible en {wrapper_path}.{k}: "
                    f"adaptador {tuple(v.shape)}, modelo {tuple(esperado[k].shape)}"
                )
        por_modulo[wrapper_path] = local

    huerfanas = set(state) - {
        f"{p}.{k}" for p, local in por_modulo.items() for k in local
    }
    if huerfanas:
        raise RuntimeError(
            f"el adaptador contiene {len(huerfanas)} claves que no corresponden "
            f"a ningún módulo CoH: {sorted(huerfanas)[:5]}"
        )

    for wrapper_path, wrapper in wrappers:
        wrapper.coh.load_state_dict(por_modulo[wrapper_path], strict=True)

    return meta
