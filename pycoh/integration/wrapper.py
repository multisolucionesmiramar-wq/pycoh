"""
pycoh.integration.wrapper — composición de CoH con un bloque huésped.

    h' = B(h) + CoH(h)

La corrección se calcula sobre la ENTRADA del bloque, no sobre su salida.
El wrapper es deliberadamente tonto: recibe un bloque ya resuelto, no
descubre nada, no resuelve capas, no congela parámetros y no serializa.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pycoh.core.coh import CoH

__all__ = ["CoHBlockWrapper"]


def _is_namedtuple(obj) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


class CoHBlockWrapper(nn.Module):
    """
    Envuelve un bloque de transformer y le suma la corrección de CoH.

    Contrato de la salida del bloque huésped:

    - `Tensor`  → devuelve `out + delta`.
    - `tuple`   → devuelve `(out[0] + delta,) + out[1:]`, conservando las
                  referencias exactas de `out[1:]`. Si es un `namedtuple`,
                  se reconstruye con su mismo tipo.
    - cualquier otro tipo → `TypeError`. No se adivina cuál de los campos
      de un `dict` o un `ModelOutput` es el hidden state.

    Todos los `*args` y `**kwargs` se reenvían al bloque sin tocar. El
    hidden state se toma del primer posicional o del kwarg
    `hidden_states`, lo que permite tanto la llamada posicional del
    gradient checkpointing como la llamada por nombre de HuggingFace.
    """

    def __init__(self, block: nn.Module, coh: CoH) -> None:
        super().__init__()
        if not isinstance(block, nn.Module):
            raise TypeError(f"block debe ser nn.Module, recibido {type(block).__name__}")
        if not isinstance(coh, CoH):
            raise TypeError(f"coh debe ser CoH, recibido {type(coh).__name__}")
        self.block = block
        self.coh = coh

    @staticmethod
    def _extract_hidden(args, kwargs) -> torch.Tensor:
        if args:
            hidden = args[0]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        else:
            raise TypeError(
                "CoHBlockWrapper no encontró el hidden state: pásalo como "
                "primer argumento posicional o como kwarg 'hidden_states'"
            )
        if not torch.is_tensor(hidden):
            raise TypeError(
                f"el hidden state debe ser un Tensor, recibido {type(hidden).__name__}"
            )
        return hidden

    def _combine(self, out, delta: torch.Tensor):
        if torch.is_tensor(out):
            return out + delta

        if isinstance(out, tuple):
            if len(out) == 0:
                raise TypeError("el bloque devolvió una tupla vacía")
            head = out[0]
            if not torch.is_tensor(head):
                raise TypeError(
                    "el primer elemento de la tupla debe ser el hidden state "
                    f"(Tensor), recibido {type(head).__name__}"
                )
            corrected = (head + delta,) + tuple(out[1:])
            if _is_namedtuple(out):
                return type(out)(*corrected)
            return corrected

        raise TypeError(
            "CoHBlockWrapper solo admite bloques que devuelvan Tensor o "
            f"tuple; recibido {type(out).__name__}. Envuelve el bloque en un "
            "adaptador que exponga el hidden state explícitamente."
        )

    def forward(self, *args, **kwargs):
        hidden = self._extract_hidden(args, kwargs)
        out = self.block(*args, **kwargs)
        delta = self.coh(hidden)
        return self._combine(out, delta)

    def extra_repr(self) -> str:
        return f"d_model={self.coh.d_model}, d_tau={self.coh.d_tau}"
