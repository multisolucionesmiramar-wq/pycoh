"""PyCoH — adaptador CoH para transformers."""
from pycoh.core.coh import CoH
from pycoh.integration.injector import apply_coh, remove_coh
from pycoh.integration.serialization import adapter_metadata, load_adapter, save_adapter
from pycoh.integration.wrapper import CoHBlockWrapper

__all__ = [
    "CoH",
    "CoHBlockWrapper",
    "apply_coh",
    "remove_coh",
    "save_adapter",
    "load_adapter",
    "adapter_metadata",
]
__version__ = "0.1.0"
