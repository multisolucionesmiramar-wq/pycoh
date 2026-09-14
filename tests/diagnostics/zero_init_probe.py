"""
F0 diagnostic -- backward behaviour with out_proj initialized to zero.
This is NOT a test: it asserts no result, it measures and reports one.

    python tests/diagnostics/zero_init_probe.py
"""

import torch

from pycoh.core.coh import CoH


def probe_normalize_at_origin() -> None:
    """Isolated probe: F.normalize alone, without the rest of the core."""
    import torch.nn.functional as F

    print("-- F.normalize at the origin --")
    x = torch.zeros(3, 5, requires_grad=True)
    y = F.normalize(x, dim=-1, eps=CoH.NORMALIZE_EPS)
    print(f"output all zeros  : {bool((y == 0).all())}")
    y.sum().backward()
    print(
        f"grad norm         : {x.grad.norm().item():.6e}  "
        f"finite={bool(torch.isfinite(x.grad).all())}"
    )
    print()


def main() -> None:
    probe_normalize_at_origin()
    torch.manual_seed(0)
    coh = CoH(16, 8)
    with torch.no_grad():
        coh.out_proj.weight.zero_()

    h = torch.randn(2, 5, 16)
    delta = coh(h)

    print(f"delta max |.|     : {delta.abs().max().item():.6e}")
    print(f"delta all zeros   : {bool((delta == 0).all())}")

    # A quadratic loss on delta yields a null upstream gradient when
    # delta == 0, masking the pathology. An arbitrary non-zero upstream is
    # used instead, which is what happens when delta enters the residual of
    # a real model.
    g = torch.randn_like(delta)
    (delta * g).sum().backward()
    for name, p in coh.named_parameters():
        if p.grad is None:
            print(f"{name:20s}: no gradient")
            continue
        g = p.grad
        print(
            f"{name:20s}: norm={g.norm().item():.6e}  "
            f"finite={bool(torch.isfinite(g).all())}"
        )


if __name__ == "__main__":
    main()
