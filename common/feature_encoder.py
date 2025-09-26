"""
Feature encoding using Deep Forest (gcForest) if available, otherwise fallback to RandomForestRegressor.
Encodes raw tabular features into learned representation for downstream DDPG.
"""
from __future__ import annotations
import numpy as np

try:  # Attempt to import deep-forest (may not be available for Python 3.13)
    from deepforest import CascadeForestRegressor  # type: ignore
    _HAS_DEEP_FOREST = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_DEEP_FOREST = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class DeepForestEncoder:
    """Tabular feature encoder.

    Workflow:
    1. Fit underlying model (Deep Forest if available else RandomForest) on (X, y).
    2. Extract intermediate representations as encoded features.
    3. Optionally concatenate original scaled features.

    If Deep Forest is available we call ``estimators_`` of each layer to get leaf-based embeddings.
    Fallback RandomForest uses leaf indices (one-hot like via normalization) for representation.
    """
    def __init__(self, add_original: bool = True, random_state: int = 42, n_estimators: int = 200):
        self.add_original = add_original
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        Xs = self.scaler.fit_transform(X)
        if _HAS_DEEP_FOREST:
            # Deep Forest automatically builds multi-grain scanning & cascade; simple usage
            self.model = CascadeForestRegressor(random_state=self.random_state)
            self.model.fit(Xs, y)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            self.model.fit(Xs, y)
        self._fitted = True
        return self

    def _encode_deep_forest(self, Xs: np.ndarray) -> np.ndarray:
        # Use predict_proba is for classification; for regression we extract layer outputs if accessible
        # Fallback: use predictions of each estimator in final layer
        # deep-forest CascadeForestRegressor exposes estimators_ as list of list of estimators per layer
        layers = getattr(self.model, 'estimators_', [])
        if not layers:
            return self.model.predict(Xs).reshape(-1, 1)
        reprs = []
        for layer in layers:  # each layer is list of estimators
            layer_preds = [est.predict(Xs).reshape(-1, 1) for est in layer]
            reprs.append(np.hstack(layer_preds))
        return np.hstack(reprs)

    def _encode_random_forest(self, Xs: np.ndarray) -> np.ndarray:
        # Leaf indices per tree -> normalized to [0,1]
        leaves = self.model.apply(Xs)  # shape (n_samples, n_trees)
        # Normalize leaf indices per tree to [0,1]
        max_leaf = leaves.max(axis=0, keepdims=True)
        encoded = leaves / (max_leaf + 1e-6)
        return encoded.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, 'Encoder not fitted.'
        X = np.asarray(X)
        Xs = self.scaler.transform(X)
        if _HAS_DEEP_FOREST:
            core = self._encode_deep_forest(Xs)
        else:
            core = self._encode_random_forest(Xs)
        if self.add_original:
            return np.hstack([Xs, core]).astype(np.float32)
        return core.astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

__all__ = ["DeepForestEncoder"]

