# ============================================================
# Employment Displacement Model (EDM)
# D(t) = D₀ * e^(βAt)
# ============================================================

import pandas as pd
import numpy as np

class EDMModel:
    """
    Employment Displacement Model (EDM)
    D(t) = D₀ * e^(βAt)
    
    Where:
    - D₀: baseline displacement (initial jobs at risk)
    - β: sensitivity of displacement to automation speed
    - A: automation speed (0-1)
    - t: time horizon (years)
    """

    def __init__(self, beta=0.3):
        self.beta = beta  # sensitivity of displacement w.r.t automation

    # --------------------------------------------------------
    # Raw Displacement (exponential growth)
    # --------------------------------------------------------
    def compute_edm_raw(self, baseline_jobs, A, time_years):
        """
        Base displacement model:
        D(t) = D₀ * e^(βAt)
        
        Args:
            baseline_jobs: Initial number of jobs at risk (D₀)
            A: Automation speed (0-1)
            time_years: Time horizon in years (t)
        
        Returns:
            Number of displaced jobs
        """
        if pd.isna(baseline_jobs) or pd.isna(A) or pd.isna(time_years):
            return np.nan
        
        if baseline_jobs <= 0 or time_years < 0:
            return np.nan
        
        exponent = self.beta * A * time_years
        # Guard against overflow
        if exponent > 100:
            exponent = 100
        
        return baseline_jobs * np.exp(exponent)

    # --------------------------------------------------------
    # Displacement Percent Increase
    # --------------------------------------------------------
    def compute_edm_percent(self, baseline_jobs, A, time_years):
        """
        Percentage job displacement:
        (D(t) - D₀) / D₀
        
        Returns the fractional increase in displaced jobs
        """
        if pd.isna(baseline_jobs) or baseline_jobs == 0 or pd.isna(A) or pd.isna(time_years):
            return np.nan

        D = self.compute_edm_raw(baseline_jobs, A, time_years)
        if pd.isna(D):
            return np.nan
        
        return (D - baseline_jobs) / baseline_jobs

    # --------------------------------------------------------
    # EDM Normalized Index (0–1)
    # --------------------------------------------------------
    def compute_edm_index(self, baseline_jobs, A, time_years):
        """
        Convert percentage job displacement to a 0–1 index.
        Scale based on β and time horizon so that:
            A = 1, t = 10 → EDM_index approaches 1
            A = 0 or t = 0 → EDM_index = 0
        
        Uses logistic function to bound output to [0, 1]
        """
        pct = self.compute_edm_percent(baseline_jobs, A, time_years)
        if pd.isna(pct):
            return np.nan

        # Scale by reference displacement (β*A*t at reference 10 years)
        ref_displacement = np.exp(self.beta * 1.0 * 10) - 1  # max reference
        
        # Apply logistic scaling to normalize to [0, 1]
        normalized = pct / ref_displacement if ref_displacement > 0 else 0
        
        # Logistic function: 1 / (1 + e^(-x))
        edm_index = 1.0 / (1.0 + np.exp(-normalized))
        
        return float(np.clip(edm_index, 0.0, 1.0))

    # --------------------------------------------------------
    # Time to Displacement Threshold
    # --------------------------------------------------------
    def compute_time_to_threshold(self, baseline_jobs, A, displacement_threshold_pct=0.5):
        """
        Calculate years required to reach a displacement threshold.
        Inverse of the displacement model.
        
        Args:
            baseline_jobs: Initial jobs at risk (D₀)
            A: Automation speed (0-1)
            displacement_threshold_pct: Target displacement percentage (e.g., 0.5 = 50%)
        
        Returns:
            Time in years to reach threshold (or NaN if unreachable)
        """
        if pd.isna(A) or A <= 0 or self.beta <= 0:
            return np.nan
        
        target_multiplier = 1.0 + displacement_threshold_pct
        
        # Solve: target_multiplier = e^(βAt)
        # ln(target_multiplier) = βAt
        # t = ln(target_multiplier) / (βA)
        
        try:
            time_result = np.log(target_multiplier) / (self.beta * A)
            return float(max(0, time_result))
        except (ValueError, ZeroDivisionError):
            return np.nan

    # --------------------------------------------------------
    # Generic normalizer for any series (used in loader)
    # --------------------------------------------------------
    def normalize(self, series):
        """Scale a Pandas series to a 0–1 index."""
        series = pd.to_numeric(series, errors="coerce")
        if series.isna().all():
            return series
        return (series - series.min()) / (series.max() - series.min() + 1e-9)