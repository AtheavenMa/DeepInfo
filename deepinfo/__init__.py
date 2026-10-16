"""DeepInfo: entropy-based neural-network information discarding."""
from .cid import CIDResult, explain
from .ru import RUResult, ConvDecoder, explain_ru, train_decoder
from .metrics import concentration, layerwise

__all__ = ["CIDResult", "RUResult", "ConvDecoder", "explain", "explain_ru",
           "train_decoder", "concentration", "layerwise"]
__version__ = "0.1.0"
