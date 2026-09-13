"""
pycoh.integration.injector — el único componente que muta el grafo.

No toma decisiones topológicas: recibe un `ResolvedTarget` y lo aplica.
La inyección es atómica: se construyen todos los wrappers primero y solo
después se reasignan, de modo que un fallo a mitad de camino no deja el
modelo en un estado híbrido.
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
    Dispositivo donde vive un módulo, o None si no tiene tensores.

    Cada CoH se crea donde está el bloque que envuelve, no donde esté el
    resto del modelo. Con un modelo repartido entre varias GPUs, cada
    adaptador aterriza junto a su bloque.
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
    """Sustituye los bloques seleccionados por wrappers. Devuelve cuántos."""
    wrappers = {}
    for idx in target.block_indices:
        original = target.container[idx]
        if isinstance(original, CoHBlockWrapper):
            raise RuntimeError(
                f"el bloque {target.path}[{idx}] ya está envuelto por CoH"
            )
        coh = CoH(
            d_model=target.hidden_size,
            d_tau=d_tau,
            beta_init=beta_init,
            r_max=r_max,
            trainable_beta=trainable_beta,
        )

        # El dispositivo se hereda del bloque; el dtype NO. CoH se queda en
        # FP32 aunque la base esté en bf16 o fp16, que es justamente la
        # política de precisión del núcleo.
        device = _module_device(original)
        if device is not None:
            coh = coh.to(device)

        wrappers[idx] = CoHBlockWrapper(original, coh)

    # Mutación solo después de haber construido todo.
    for idx, wrapper in wrappers.items():
        target.container[idx] = wrapper
    return len(wrappers)


def freeze_base(model: nn.Module) -> None:
    """
    Congela todo lo que no pertenezca a un CoH, por estructura y no por
    nombre. Los parámetros de CoH se dejan exactamente como los construyó
    el núcleo, de modo que `beta` conserva el `requires_grad` que fijó
    `trainable_beta`: recorrer todo poniendo False y después reactivar a
    mano borraría esa intención.
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
    Aplica CoH a un modelo, in-place, y devuelve el mismo objeto.

    Orquesta inspeccionar → resolver → validar → inyectar → congelar.
    Ante arquitectura no soportada, dimensión no resoluble, índices
    inválidos, ambigüedad o doble inyección, lanza una excepción sin dejar
    el modelo parcialmente modificado.

    `d_tau` es obligatorio: no existe un valor automático.
    """
    info = inspect_model(model)

    if info.already_injected:
        raise RuntimeError(
            "El modelo ya tiene CoH aplicado en: "
            + ", ".join(info.injected_paths)
            + ". Una segunda aplicación produciría wrappers anidados."
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
    Desinstala CoH y restaura la topología original, in-place.

    Cada `CoHBlockWrapper` se sustituye por el bloque que envolvía — el
    mismo objeto, no una copia — de modo que `named_modules()` y
    `state_dict()` vuelven a ser exactamente los del modelo limpio y las
    APIs normales de HuggingFace (`save_pretrained`) vuelven a funcionar.

    No restaura pesos: el modelo base nunca se modificó, así que no hay
    nada que restaurar. El estado de CoH se pierde salvo que se haya
    guardado antes con `save_adapter`.

    `requires_grad` **no se toca** por defecto. `apply_coh` congeló la base
    y aquí no se sabe cuál era el estado previo, así que adivinarlo sería
    peor que dejarlo explícito: pasa `unfreeze=True` para reactivar todos
    los parámetros restantes.

    Lanza `RuntimeError` si el modelo no tiene CoH: quitar algo que no está
    es casi siempre un error de quien llama, no una operación vacía.
    """
    removed = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, CoHBlockWrapper):
                setattr(parent, name, child.block)
                removed += 1

    if removed == 0:
        raise RuntimeError(
            "El modelo no tiene CoH aplicado: no hay nada que quitar."
        )

    if unfreeze:
        for p in model.parameters():
            p.requires_grad_(True)

    return model
