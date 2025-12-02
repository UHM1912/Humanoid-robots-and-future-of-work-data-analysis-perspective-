"""
ERI Model Backend (Advanced)
----------------------------
Supports multiple curve modes:
- linear
- quadratic
- exponential
- logistic
"""

import numpy as np
import pandas as pd


class ERIModel:
    def __init__(self, weight_A=1.0, weight_W=1.0, weight_S=1.0):
        self.weight_A = weight_A
        self.weight_W = weight_W
        self.weight_S = weight_S

    def normalize(self, x):
        if isinstance(x, (list, np.ndarray, pd.Series)):
            x = np.array(x, dtype=float)
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        else:
            return min(max(x, 0), 1)

    def compute_eri(self, A, W, S, mode="linear"):
        """
        Compute Employment Risk Index using the selected curve mode.
        """
        A = self.normalize(A)
        W = self.normalize(W)
        S = self.normalize(S)

        if mode == "linear":
            eri = (A * W) / (S + 1)
        elif mode == "quadratic":
            eri = (W * A**2) / (S + 1)
        elif mode == "exponential":
            eri = (W * (np.exp(A) - 1)) / (S + 1)
        elif mode == "logistic":
            k = 10  # controls steepness
            eri = (W / (S + 1)) * (1 / (1 + np.exp(-k * (A - 0.5))))
        else:
            raise ValueError("Invalid mode: choose linear, quadratic, exponential, or logistic.")

        return round(float(eri), 4)

    def interpret(self, eri_value):
        if eri_value < 0.2:
            return "Low Employment Risk"
        elif eri_value < 0.6:
            return "Moderate Employment Risk"
        else:
            return "High Employment Risk"

    def simulate_scenarios(self, A_values, W, S, mode="linear"):
        results = []
        for A in A_values:
            eri = self.compute_eri(A, W, S, mode=mode)
            results.append({
                "Automation Speed": A,
                "ERI": eri,
                "Risk": self.interpret(eri)
            })
        return pd.DataFrame(results)

    