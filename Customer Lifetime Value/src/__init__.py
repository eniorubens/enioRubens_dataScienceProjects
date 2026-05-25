"""
Customer Lifetime Value — src package.
"""

from .cltv_model import CLTVModel
from .evaluation import CLTVEvaluator
from .preprocessing import OnlineRetailPreprocessor
from .segmentation import (
    ACTION_ORDER,
    ACTION_PALETTE,
    SEGMENT_ORDER,
    VALUE_PALETTE,
    CustomerSegmenter,
)
from .utils import add_chart_labels, format_currency_axis, get_logger

__version__ = "1.0.0"

__all__ = [
    "OnlineRetailPreprocessor",
    "CLTVModel",
    "CustomerSegmenter",
    "CLTVEvaluator",
    "format_currency_axis",
    "add_chart_labels",
    "get_logger",
    "SEGMENT_ORDER",
    "ACTION_ORDER",
    "VALUE_PALETTE",
    "ACTION_PALETTE",
]
