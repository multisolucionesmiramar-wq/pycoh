"""
pycoh.integration.serialization -- saving and loading adapters.

An adapter holds the state of the CoH modules and nothing else. It carries
no weight of the host model: rebuilding means starting from the base
checkpoint, calling `apply_coh` and then loading the adapter.

Extraction is structural (`isinstance(module, CoHBlockWrapper)`) rather
than name-based: filtering on the substring ".coh." would work today and
would tie us back to names, which is exactly what F2 removed.
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
            "The model has no CoH applied. Call apply_coh(...) first."
        )
    return found


def _shared_config(wrappers: List[Tuple[str, CoHBlockWrapper]]) -> Dict[str, Any]:
    """
    The adapter declares a single configuration, so every CoH must share it.
    If someone hand-built layers with different d_tau values we fail instead
    of recording whichever came first.
    """
    configs = {
        path: (w.coh.d_model, w.coh.d_tau, w.coh.r_max, w.coh.trainable_beta)
        for path, w in wrappers
    }
    distinct = set(configs.values())
    if len(distinct) > 1:
        detail = "\n".join(
            f"  {p}: d_model={c[0]}, d_tau={c[1]}, r_max={c[2]}, trainable_beta={c[3]}"
            for p, c in configs.items()
        )
        raise ValueError(
            "The CoH modules in this model do not share a configuration and "
            "the adapter format declares a single one:\n" + detail
        )
    d_model, d_tau, r_max, trainable_beta = distinct.pop()
    return {
        "d_model": d_model,
        "d_tau": d_tau,
        "r_max": r_max,
        "trainable_beta": trainable_beta,
    }


def save_adapter(model: nn.Module, path: str | os.PathLike) -> Dict[str, Any]:
    """
    Save the model's CoH state. Returns the metadata that was written.

    The resulting file weighs what CoH weighs (about 24 MB with d_model=960,
    d_tau=96 and 32 layers), not what the model weighs.
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
    """Read the metadata only, without touching any model."""
    return _read_payload(path)["metadata"]


def _read_payload(path: str | os.PathLike) -> Dict[str, Any]:
    # weights_only=True stops a downloaded file from executing arbitrary
    # code while being deserialized. The metadata are plain types, so they
    # pass the filter without trouble.
    payload = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(payload, dict) or "metadata" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path} does not have the shape of a PyCoH adapter")

    meta = payload["metadata"]
    if meta.get("mechanism") != MECHANISM:
        raise ValueError(
            f"unknown mechanism: {meta.get('mechanism')!r}, expected {MECHANISM!r}"
        )
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"format_version {version!r} is not supported by this build of PyCoH "
            f"(it supports {FORMAT_VERSION})"
        )
    return payload


def load_adapter(model: nn.Module, path: str | os.PathLike) -> Dict[str, Any]:
    """
    Restore CoH state onto a model that ALREADY has CoH applied.

    It does not change the topology and creates no modules: if the model was
    not injected, or was injected with a different configuration or on
    different layers, it fails without touching anything. Validation is
    complete before the first tensor is written, so an incompatible adapter
    never leaves the model half-loaded.

    `trainable_beta` from the file is informational: it describes how the
    adapter was trained and does not change what the user configured on this
    model.
    """
    payload = _read_payload(path)
    meta = payload["metadata"]
    state = payload["state_dict"]

    wrappers = _collect(model)
    current_paths = [p for p, _ in wrappers]
    saved_paths = list(meta["injected_paths"])

    if current_paths != saved_paths:
        missing = [p for p in saved_paths if p not in current_paths]
        extra = [p for p in current_paths if p not in saved_paths]
        raise RuntimeError(
            "The adapter topology does not match the model.\n"
            f"  adapter: {len(saved_paths)} layers\n"
            f"  model:   {len(current_paths)} layers\n"
            + (f"  in the adapter but not in the model: {missing}\n" if missing else "")
            + (f"  in the model but not in the adapter: {extra}\n" if extra else "")
            + "Call apply_coh with the same layers before loading."
        )

    config = _shared_config(wrappers)
    for field in ("d_model", "d_tau", "r_max"):
        if meta[field] != config[field]:
            raise RuntimeError(
                f"incompatible {field}: the adapter declares {meta[field]!r} and "
                f"the model has {config[field]!r}"
            )

    # Full split and validation BEFORE writing anything.
    per_module: Dict[str, Dict[str, torch.Tensor]] = {}
    for wrapper_path, wrapper in wrappers:
        prefix = wrapper_path + "."
        local = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        expected = wrapper.coh.state_dict()

        if set(local) != set(expected):
            missing = sorted(set(expected) - set(local))
            extra = sorted(set(local) - set(expected))
            raise RuntimeError(
                f"incompatible keys at {wrapper_path}: "
                f"missing {missing}, unexpected {extra}"
            )
        for k, v in local.items():
            if tuple(v.shape) != tuple(expected[k].shape):
                raise RuntimeError(
                    f"incompatible shape at {wrapper_path}.{k}: "
                    f"adapter {tuple(v.shape)}, model {tuple(expected[k].shape)}"
                )
        per_module[wrapper_path] = local

    orphans = set(state) - {
        f"{p}.{k}" for p, local in per_module.items() for k in local
    }
    if orphans:
        raise RuntimeError(
            f"the adapter holds {len(orphans)} keys that match no CoH module: "
            f"{sorted(orphans)[:5]}"
        )

    for wrapper_path, wrapper in wrappers:
        wrapper.coh.load_state_dict(per_module[wrapper_path], strict=True)

    return meta
