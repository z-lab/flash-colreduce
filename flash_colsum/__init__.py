"""
Flash-ColSum: Efficient attention column-sum operations with Triton kernels.
"""

from .ops import flash_colsum
from .baselines import naive_colsum

__all__ = ["flash_colsum", "naive_colsum"]

__version__ = "1.0.0"