"""Filter Bank CSP + SVM implementation for four-class motor imagery."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import joblib
from scipy import signal
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class FBCSPConfig:
    sample_rate: float
    filter_banks: Sequence[Tuple[float, float]]
    components_per_band: int
    svm_kernel: str = "linear"
    svm_c: float = 1.0
    svm_class_weight: Dict[int, float] | str | None = None

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "FBCSPConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class FilterBankCSPClassifier:
    """Minimal Filter Bank CSP classifier with an SVM backend."""

    def __init__(self, config: FBCSPConfig) -> None:
        self.config = config
        self._sos_filters: List[np.ndarray] = []
        self._spatial_filters: List[np.ndarray] = []
        self._scaler = StandardScaler()
        self._svm = SVC(
            kernel=config.svm_kernel,
            C=config.svm_c,
            class_weight=config.svm_class_weight,
            probability=True,
        )

    # -------------------- public API --------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit CSP filters and SVM using trials shaped (n_trials, n_channels, n_samples)."""
        self._design_filter_bank()
        self._spatial_filters = []
        features = self._extract_features(X, fit_csp=True, labels=y)
        scaled = self._scaler.fit_transform(features)
        self._svm.fit(scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self.transform(X)
        return self._svm.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = self.transform(X)
        return self._svm.predict_proba(feats)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        feats = self.transform(X)
        return self._svm.decision_function(feats)

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = self._extract_features(X, fit_csp=False)
        return self._scaler.transform(features)

    @property
    def classes_(self) -> np.ndarray:
        return self._svm.classes_

    # -------------------- persistence --------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        artifact = {
            "sos_filters": self._sos_filters,
            "spatial_filters": self._spatial_filters,
            "scaler": self._scaler,
            "svm": self._svm,
        }
        joblib.dump(artifact, directory / "model.joblib")
        self.config.to_json(directory / "config.json")

    @classmethod
    def load(cls, directory: Path) -> "FilterBankCSPClassifier":
        config = FBCSPConfig.from_json(directory / "config.json")
        model = cls(config)
        artifact = joblib.load(directory / "model.joblib")
        model._sos_filters = artifact["sos_filters"]
        model._spatial_filters = artifact["spatial_filters"]
        model._scaler = artifact["scaler"]
        model._svm = artifact["svm"]
        return model

    # -------------------- internal helpers --------------------
    def _design_filter_bank(self) -> None:
        nyq = 0.5 * self.config.sample_rate
        self._sos_filters = [
            signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
            for low, high in self.config.filter_banks
        ]

    def _bandpass(self, data: np.ndarray, sos: np.ndarray) -> np.ndarray:
        return signal.sosfiltfilt(sos, data, axis=-1)

    def _extract_features(
        self,
        X: np.ndarray,
        fit_csp: bool,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        feature_list: List[np.ndarray] = []
        for band_idx, sos in enumerate(self._sos_filters):
            filtered = self._bandpass(X, sos)
            if fit_csp:
                assert labels is not None
                spatial_filters = self._fit_csp(filtered, labels)
                self._spatial_filters.append(spatial_filters)
            else:
                spatial_filters = self._spatial_filters[band_idx]
            projected = np.einsum("fc,tcw->tfw", spatial_filters, filtered, optimize=True)
            variances = np.var(projected, axis=-1)
            # Normalize variance per trial to enforce scale invariance
            denom = np.maximum(variances.sum(axis=1, keepdims=True), 1e-12)
            variances /= denom
            feature_list.append(np.log(variances))
        return np.concatenate(feature_list, axis=1)

    def _fit_csp(self, X_band: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique_labels = np.unique(labels)
        comps = self.config.components_per_band
        filters_per_class = max(1, comps // 2)
        spatial_filters = []
        for cls in unique_labels:
            cls_idx = labels == cls
            rest_idx = ~cls_idx
            cov_cls = self._mean_covariance(X_band[cls_idx])
            cov_rest = self._mean_covariance(X_band[rest_idx])
            eigenvalues, eigenvectors = self._solve_generalized_eigen(cov_cls, cov_rest)
            spatial_filters.append(eigenvectors[:, :filters_per_class])
            spatial_filters.append(eigenvectors[:, -filters_per_class:])
        return np.concatenate(spatial_filters, axis=1).T

    def _mean_covariance(self, trials: np.ndarray) -> np.ndarray:
        cov = np.zeros((trials.shape[1], trials.shape[1]))
        for trial in trials:
            c = trial @ trial.T
            cov += c / np.trace(c)
        return cov / trials.shape[0]

    def _solve_generalized_eigen(self, cov_a: np.ndarray, cov_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cov_sum = cov_a + cov_b
        eigenvalues, eigenvectors = eigh(cov_a, cov_sum)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvalues[order], eigenvectors[:, order]

