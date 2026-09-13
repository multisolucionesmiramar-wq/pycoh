"""
Diagnóstico F0 — comportamiento del backward con out_proj inicializado a
cero. NO es un test: no afirma un resultado, lo mide y lo reporta.

    python tests/diagnostics/zero_init_probe.py
"""

import torch

from pycoh.core.coh import CoH


def probe_normalize_at_origin() -> None:
    """Sonda aislada: solo F.normalize, sin el resto del núcleo."""
    import torch.nn.functional as F

    print("-- F.normalize en el origen --")
    x = torch.zeros(3, 5, requires_grad=True)
    y = F.normalize(x, dim=-1, eps=CoH.NORMALIZE_EPS)
    print(f"salida todo cero   : {bool((y == 0).all())}")
    y.sum().backward()
    print(
        f"grad norma         : {x.grad.norm().item():.6e}  "
        f"finito={bool(torch.isfinite(x.grad).all())}"
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

    print(f"delta max |.|      : {delta.abs().max().item():.6e}")
    print(f"delta todo cero    : {bool((delta == 0).all())}")

    # Un loss cuadrático sobre delta da gradiente upstream nulo cuando
    # delta == 0, y enmascara la patología. Se usa un upstream arbitrario
    # no nulo, que es lo que ocurre cuando delta entra en el residual de
    # un modelo real.
    g = torch.randn_like(delta)
    (delta * g).sum().backward()
    for name, p in coh.named_parameters():
        if p.grad is None:
            print(f"{name:20s}: sin gradiente")
            continue
        g = p.grad
        print(
            f"{name:20s}: norma={g.norm().item():.6e}  "
            f"finito={bool(torch.isfinite(g).all())}"
        )


if __name__ == "__main__":
    main()
