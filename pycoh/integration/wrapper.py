"""
pycoh.integration.wrapper -- composing CoH with a host block.

    h' = B(h) + CoH(h)

The correction is computed from the block's input, not from its output.
The wrapper is deliberately dumb: it receives an already-resolved block, it
discovers nothing, resolves nothing, freezes nothing and serializes
nothing.
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
    Wrap a transformer block and add the CoH correction to its output.

    Contract for the host block's output:

    - `Tensor` -> returns `out + delta`.
    - `tuple`  -> returns `(out[0] + delta,) + out[1:]`, preserving the
                  exact references in `out[1:]`. A `namedtuple` is rebuilt
                  with its own type.
    - anything else -> `TypeError`. The library does not guess which field
      of a `dict` or a `ModelOutput` holds the hidden state.

    All `*args` and `**kwargs` are forwarded to the block untouched. The
    hidden state is taken from the first positional argument or from the
    `hidden_states` keyword, which supports both the positional call used
    by gradient checkpointing and the keyword call used by HuggingFace.
    """

    def __init__(self, block: nn.Module, coh: CoH) -> None:
        super().__init__()
        if not isinstance(block, nn.Module):
            raise TypeError(f"block must be an nn.Module, got {type(block).__name__}")
        if not isinstance(coh, CoH):
            raise TypeError(f"coh must be a CoH, got {type(coh).__name__}")
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
                "CoHBlockWrapper could not find the hidden state: pass it as "
                "the first positional argument or as the 'hidden_states' keyword"
            )
        if not torch.is_tensor(hidden):
            raise TypeError(
                f"the hidden state must be a Tensor, got {type(hidden).__name__}"
            )
        return hidden

    def _combine(self, out, delta: torch.Tensor):
        if torch.is_tensor(out):
            return out + delta

        if isinstance(out, tuple):
            if len(out) == 0:
                raise TypeError("the block returned an empty tuple")
            head = out[0]
            if not torch.is_tensor(head):
                raise TypeError(
                    "the first element of the tuple must be the hidden state "
                    f"(Tensor), got {type(head).__name__}"
                )
            corrected = (head + delta,) + tuple(out[1:])
            if _is_namedtuple(out):
                return type(out)(*corrected)
            return corrected

        raise TypeError(
            "CoHBlockWrapper only supports blocks returning a Tensor or a "
            f"tuple; got {type(out).__name__}. Wrap the block in an adapter "
            "that exposes the hidden state explicitly."
        )

    def forward(self, *args, **kwargs):
        hidden = self._extract_hidden(args, kwargs)
        out = self.block(*args, **kwargs)
        delta = self.coh(hidden)
        return self._combine(out, delta)

    def extra_repr(self) -> str:
        return f"d_model={self.coh.d_model}, d_tau={self.coh.d_tau}"
