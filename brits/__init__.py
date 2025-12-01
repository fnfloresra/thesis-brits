"""
BRITS: Bidirectional Recurrent Imputation for Time Series

A PyTorch implementation of the BRITS model for multivariate time series imputation.

Reference:
    Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
    BRITS: Bidirectional Recurrent Imputation for Time Series.
    Advances in Neural Information Processing Systems, 31.
"""

from brits.models.brits import BRITS
from brits.models.rits import RITS

__version__ = "0.1.0"
__all__ = ["BRITS", "RITS"]
