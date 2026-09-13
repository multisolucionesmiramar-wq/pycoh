"""
pycoh.core.coh -- the CoH mechanism.

    z  = W_tau h
    r  = min(sigmoid(phi(z)), r_max)
    s  = 1/sqrt(1 - r^2) - 1
    J  = normalize(W_out z)
    dh = beta * J * s

The correction is computed from the block's INPUT. All arithmetic runs in
FP32 and the result is returned in the input dtype.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CoH"]

# device_type values accepted by torch.autocast on supported versions.
_AUTOCAST_DEVICES = ("cuda", "cpu", "xpu")


def _fp32_context(device_type: str):
    """
    Disable autocast inside the core.

    Under AMP, torch.autocast casts the inputs of linear operations down to
    low precision regardless of the dtype they are given: `hidden.float()`
    alone does NOT guarantee FP32 arithmetic. This context is the only real
    guarantee.
    """
    if device_type in _AUTOCAST_DEVICES:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


class CoH(nn.Module):
    """
    CoH core. Returns the correction `delta`, not the corrected state.
    Adding `B(h) + delta` is the wrapper's job.

    Args:
        d_model: hidden size of the block.
        d_tau: bottleneck width. Required, no default.
        beta_init: initial correction amplitude. Defaults to 0.5.
        r_max: gate ceiling. Defaults to 0.98.
        trainable_beta: if True, beta receives gradients. Defaults to False.

    Note: `beta` is a persistent `nn.Parameter` (it travels in the
    `state_dict`) even though it is frozen by default. A saved adapter
    therefore carries its own beta and does not depend on the user
    supplying it again through configuration.
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
            raise ValueError(f"d_model must be a positive integer, got {d_model!r}")
        if not isinstance(d_tau, int) or d_tau <= 0:
            raise ValueError(f"d_tau must be a positive integer, got {d_tau!r}")
        if not 0.0 < float(r_max) < 1.0:
            raise ValueError(f"r_max must lie in (0, 1), got {r_max!r}")

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

    # -- introspection ----------------------------------------------------

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

    # -- core -------------------------------------------------------------

    def _stages(self, hidden: torch.Tensor) -> dict:
        """
        Return every intermediate stage in FP32. It exists so that tests can
        compare against an independent mathematical reference stage by
        stage; it is not part of the public API.

        Weights are cast with `.float()` before each operation: on a tensor
        that is already FP32 the call is a no-op with no copy, and if the
        module was converted to half precision the arithmetic still runs in
        FP32.
        """
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"last dimension {hidden.shape[-1]} != d_model {self.d_model}"
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
