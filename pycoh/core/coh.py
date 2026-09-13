"""
pycoh.core.coh — núcleo del mecanismo CoH.

    z  = W_tau h
    r  = min(sigmoid(phi(z)), r_max)
    s  = 1/sqrt(1 - r^2) - 1
    J  = normalize(W_out z)
    dh = beta * J * s

La corrección se calcula sobre la ENTRADA del bloque. Toda la aritmética
ocurre en FP32 y el resultado se devuelve en el dtype de entrada.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CoH"]

# device_type aceptados por torch.autocast en las versiones soportadas.
_AUTOCAST_DEVICES = ("cuda", "cpu", "xpu")


def _fp32_context(device_type: str):
    """
    Desactiva autocast dentro del núcleo.

    Bajo AMP, torch.autocast castea las entradas de las operaciones
    lineales a la precisión baja sin importar el dtype que se les pase:
    `hidden.float()` por si solo NO garantiza aritmetica FP32. Este
    contexto es la unica garantia real.
    """
    if device_type in _AUTOCAST_DEVICES:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


class CoH(nn.Module):
    """
    Núcleo de CoH. Devuelve la corrección `delta`, no el estado corregido.
    La suma `B(h) + delta` es responsabilidad del wrapper.

    Args:
        d_model: dimensión del estado oculto del bloque.
        d_tau: dimensión del cuello de botella. Requerido, sin default.
        beta_init: valor inicial de la amplitud. Default 0.5.
        r_max: techo del gate. Default 0.98.
        trainable_beta: si True, beta recibe gradiente. Default False.

    Atención: `beta` es un `nn.Parameter` persistente (viaja en el
    `state_dict`) aunque por defecto esté congelado. Un adapter guardado
    contiene su propio valor de beta y no depende de que el usuario
    vuelva a suministrarlo por configuración.
    """

    NORMALIZE_EPS: float = 1e-12

    def __init__(
        self,
        d_model: int,
        d_tau: int,
        beta_init: float = 0.5,
        r_max: float = 0.98,
        trainable_beta: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model debe ser un entero positivo, recibido {d_model!r}")
        if not isinstance(d_tau, int) or d_tau <= 0:
            raise ValueError(f"d_tau debe ser un entero positivo, recibido {d_tau!r}")
        if not 0.0 < float(r_max) < 1.0:
            raise ValueError(f"r_max debe estar en (0, 1), recibido {r_max!r}")

        self.d_model = int(d_model)
        self.d_tau = int(d_tau)
        self.r_max = float(r_max)

        self.W_tau = nn.Linear(d_model, d_tau, bias=False)
        self.phi_proj = nn.Linear(d_tau, 1, bias=False)
        self.out_proj = nn.Linear(d_tau, d_model, bias=False)

        self.beta = nn.Parameter(
            torch.tensor(float(beta_init)),
            requires_grad=bool(trainable_beta),
        )

    # ── introspección ────────────────────────────────────────────────────

    @property
    def trainable_beta(self) -> bool:
        return bool(self.beta.requires_grad)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_tau={self.d_tau}, "
            f"r_max={self.r_max}, beta={self.beta.item():.4g}, "
            f"trainable_beta={self.trainable_beta}"
        )

    # ── núcleo ───────────────────────────────────────────────────────────

    def _stages(self, hidden: torch.Tensor) -> dict:
        """
        Devuelve todas las etapas intermedias en FP32. Existe para que los
        tests puedan comparar contra una referencia matemática etapa por
        etapa; no forma parte de la API pública.

        Los pesos se castean con `.float()` antes de cada operación: si el
        tensor ya es FP32 la llamada es un no-op sin copia, y si el módulo
        fue convertido a media precisión la aritmética sigue siendo FP32.
        """
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"última dimensión {hidden.shape[-1]} != d_model {self.d_model}"
            )

        with _fp32_context(hidden.device.type):
            h32 = hidden.float()

            z = F.linear(h32, self.W_tau.weight.float())
            gate = torch.sigmoid(F.linear(z, self.phi_proj.weight.float()))
            ratio = gate.clamp(max=self.r_max)
            F_val = 1.0 / torch.sqrt(1.0 - ratio ** 2)
            scale = F_val - 1.0

            d_hat = F.normalize(
                F.linear(z, self.out_proj.weight.float()),
                dim=-1,
                eps=self.NORMALIZE_EPS,
            )

            delta = (self.beta.float() * d_hat) * scale

        return {
            "z": z,
            "gate": gate,
            "ratio": ratio,
            "F_val": F_val,
            "scale": scale,
            "d_hat": d_hat,
            "delta": delta,
        }

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._stages(hidden)["delta"].to(hidden.dtype)
