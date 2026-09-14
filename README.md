# PyCoH

*English · [Español](README.es.md)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22738977.svg)](https://doi.org/10.5281/zenodo.22738977)

Parameter-efficient fine-tuning adapter for transformers. It injects a
directional correction into the residual stream of every block, with the
base model fully frozen.

```python
from transformers import AutoModelForCausalLM
from pycoh import apply_coh, remove_coh, save_adapter, load_adapter

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M").cuda()
apply_coh(model, d_tau=96)

# ...your usual training loop...

save_adapter(model, "my_adapter.pt")
```

To reuse it on a clean instance of the same base model:

```python
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M").cuda()
apply_coh(model, d_tau=96)
load_adapter(model, "my_adapter.pt")
```

The lifecycle is symmetric:

```
apply_coh     install the adapter
remove_coh    uninstall it and restore the topology
save_adapter  persist it
load_adapter  restore it
```

---

## What it does

For the hidden state `h` entering each block:

```
z  = W_tau h                  compress to d_tau dimensions
r  = min(sigmoid(phi(z)), r_max)   per-token gate, with a ceiling
s  = 1/sqrt(1 - r^2) - 1      amplitude
J  = normalize(W_out z)       direction, unit norm
dh = beta * J * s
h' = B(h) + dh
```

The correction is computed from the block's **input**, not from its output.
The direction depends on the current state; its norm is always 1, so the
amplitude is decoupled from the magnitude of the state. The ceiling `r_max`
bounds `s` at 4.0252 and prevents `1/sqrt(1-r^2)` from diverging.

Trainable parameters per layer: `2 * d_model * d_tau + d_tau`.

## Measured numbers

On `SmolLM2-360M` with `d_tau=96`, 32 layers:

| | |
|---|---|
| Adapter parameters | 5,901,344 (1.6% of the model) |
| Trainable by default | 5,901,312 |
| File size | 23.6 MB (against 724 MB for the model) |
| Base model modified | 0 tensors, verified after training |

Verified by execution on a T4: HuggingFace `Trainer`, base in `bfloat16`,
gradient checkpointing enabled, adapter saved and reloaded onto a freshly
downloaded model reproducing the loss.

**PyCoH does not yet publish any performance measurement.** There is no
comparison against LoRA or full fine-tuning in this repository. What is
verified is that the mechanism applies, trains, saves and reloads
correctly.

## Numerical precision

All core arithmetic runs in FP32 and the result is returned in the input
dtype. This holds **under mixed precision with no configuration on your
part**: `torch.autocast` casts the inputs of linear operations down to low
precision regardless of the dtype it is given, so the core disables
autocast internally. A base model in `bfloat16` with an FP32 adapter is the
normal setup and it works on its own.

## Limitations

These are documented because they are design properties, not defects
awaiting a fix.

**It cannot be merged into the weights.** `dh` is a function of the
activation, not a weight delta, so there is no `merge_and_unload()`.
Unlike LoRA, the cost is paid on every forward pass. The latency overhead
has not been measured and is not claimed.

**`apply_coh()` is not the identity at initialization.** `out_proj` starts
from PyTorch's default init, so the correction is non-zero from the first
forward pass. On SmolLM2-360M that produces a maximum logit change of about
0.65 before any training. Anyone coming from LoRA expects the opposite.

Initializing `out_proj` to zero is **not** the fix: `F.normalize` divides
by `||x||.clamp_min(eps)`, so at the origin the derivative is
`1/eps ~ 1e12`. Measured: gradient norm `2.18e12` as soon as the incoming
gradient is non-zero. Achieving identity at initialization requires a
different design and is left for a later version.

**`save_pretrained()` does not work while CoH is installed.** Wrapping each
block shifts the keys from `model.layers.3.self_attn...` to
`model.layers.3.block.self_attn...`, so `from_pretrained` can no longer
rebuild it. That is why the adapter is saved separately with
`save_adapter`. If you need the base checkpoint, `remove_coh(model)`
restores the original topology and `save_pretrained` works again.

**`d_tau` is required and has no default.** The only ratio with empirical
support is `d_tau/d_model = 0.1`, measured at `d_model=960`. Extrapolating
it to other sizes is a hypothesis, not a rule, and the library does not
apply it silently.

**`trainable_beta=True` and weight decay.** If you make `beta` trainable
and put it in an AdamW group with `weight_decay`, `beta` decays. Measured
on a toy task over 150 steps: `0.5 -> 1.0255` without weight decay,
`0.5 -> 0.9743` with the default 0.01. That is a modest drift, not a
collapse, but over long runs `beta` deserves its own group with
`weight_decay=0.0`.

## API

### `apply_coh(model, *, d_tau, ...)`

Applies CoH **in place** and returns the same object. It discovers the
block stack structurally, not by name: it looks for homogeneous
`nn.ModuleList` containers, discards the ones nested inside other
candidates, and checks that the block's parameters operate on `d_model`.
**On ambiguity it fails** and lists the candidates; it never decides on its
own.

| Argument | Default | |
|---|---|---|
| `d_tau` | — | required |
| `layers` | `None` | `None`=all, `N`=first N, `[i,j]`=indices |
| `hidden_size` | `None` | override; otherwise `config.hidden_size` or `config.d_model` |
| `target_modules` | `None` | exact container path, to resolve ambiguity |
| `beta_init` | `0.5` | |
| `r_max` | `0.98` | |
| `trainable_beta` | `False` | |
| `freeze` | `True` | freezes everything that is not CoH |

Each adapter is created on its own block's device, so a model sharded
across several GPUs works without adjustment. The dtype is not inherited:
CoH stays in FP32.

A second call on an already-injected model raises `RuntimeError`. On any
validation error the model is left **untouched**: every wrapper is built
before the first one is installed.

### `remove_coh(model, *, unfreeze=False)`

Uninstalls CoH and restores the original topology, in place. Each wrapper
is replaced by the very block object it wrapped, so `named_modules()` and
`state_dict()` go back to exactly what the clean model had.

It restores no weights: the base model was never modified. CoH state is
lost unless it was saved beforehand with `save_adapter`.

`requires_grad` is left untouched: `apply_coh` froze the base and there is
no record of the previous state here, so guessing would be worse than being
explicit. `unfreeze=True` re-enables every remaining parameter.

On a model without CoH it raises `RuntimeError`.

### `save_adapter(model, path)` / `load_adapter(model, path)`

The file holds the state of the CoH modules and nothing else — not a single
host weight — plus metadata: format version, mechanism, `d_model`,
`d_tau`, `r_max`, `trainable_beta` and the list of injected paths.

`load_adapter` requires the model to **already** have CoH applied; it does
not change the topology. It validates mechanism, version, paths, `d_model`,
`d_tau`, `r_max`, keys and shapes **before** writing a single tensor, so an
incompatible adapter never leaves the model half-loaded. It never uses
`strict=False`.

`r_max` is validated explicitly because it changes no shape: without that
check, an adapter trained under a different ceiling would load without
error and the mechanism would behave differently in silence.

Deserialization uses `weights_only=True`. PyTorch checkpoints execute code
when opened; an adapter downloaded from the internet should not be able to.

`trainable_beta` in the file is informational: it describes how the adapter
was trained and does not change the receiving model's configuration. The
value of `beta`, by contrast, is restored from the file.

## Compatibility

Requires the model to have a homogeneous list of blocks, those blocks to
take the hidden state as their first argument and return a `Tensor` or a
`tuple`, and the config to declare the hidden size.

Verified on `SmolLM2-360M` with `transformers 5.16`, `torch 2.11` and
`2.14`. Encoder-decoder models have two stacks and require an explicit
`target_modules`. Models declaring the dimension as `n_embd` (the GPT-2
family) need `hidden_size` passed by hand.

A block output that is neither a `Tensor` nor a `tuple` raises an explicit
`TypeError`: the library does not guess which field of a `dict` holds the
hidden state.

## Installation

```bash
pip install pycoh
```

From the repository:

```bash
git clone https://github.com/multisolucionesmiramar-wq/pycoh && cd pycoh
pip install -e ".[dev]"
pytest -q
```

## Tests

```bash
pytest -q                                          # full suite
python tests/diagnostics/zero_init_probe.py        # diagnostic
PYTHONPATH=. python tests/integration/smollm2_run.py    # needs network
PYTHONPATH=. python tests/integration/smollm2_train.py  # needs network and GPU
```

Core tests compare every stage of the computation **bit for bit** against a
reference written from the specification, which never calls the code under
test.
