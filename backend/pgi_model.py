# ============================================================
# Productivity Gain Model (PGI)
# ============================================================

import pandas as pd
import numpy as np

class PGIModel:
    """
    Productivity Gain Index (PGI) Model
    P = P0 * (1 + αA)
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha  # elasticity of productivity w.r.t automation

    # --------------------------------------------------------
    # Raw Productivity
    # --------------------------------------------------------
    def compute_pgi_raw(self, earnings, A):
        """
        Base productivity model:
        P = P0 * (1 + αA)
        """
        if pd.isna(earnings) or pd.isna(A):
            return np.nan
        return earnings * (1 + self.alpha * A)

    # --------------------------------------------------------
    # PGI Percent Gain
    # --------------------------------------------------------
    def compute_pgi_percent(self, earnings, A):
        """
        Percentage productivity increase:
        (P - P0) / P0
        """
        if pd.isna(earnings) or earnings == 0 or pd.isna(A):
            return np.nan

        P = self.compute_pgi_raw(earnings, A)
        return (P - earnings) / earnings

    # --------------------------------------------------------
    # PGI Normalized Index (0–1)
    # --------------------------------------------------------
    def compute_pgi_index(self, earnings, A):
        """
        Convert percentage productivity increase to a 0–1 index.
        Scale based on α so that:
            A = 1 → PGI_index = 1
            A = 0 → PGI_index = 0
        """
        pct = self.compute_pgi_percent(earnings, A)
        if pd.isna(pct):
            return np.nan

        # normalization relative to α
        # max possible pct gain = α
        return float(np.clip(pct / self.alpha, 0.0, 1.0))

    # --------------------------------------------------------
    # Generic normalizer for any series (used in loader)
    # --------------------------------------------------------
    def normalize(self, series):
        """Scale a Pandas series to a 0–1 index."""
        series = pd.to_numeric(series, errors="coerce")
        if series.isna().all():
            return series
        return (series - series.min()) / (series.max() - series.min() + 1e-9)
