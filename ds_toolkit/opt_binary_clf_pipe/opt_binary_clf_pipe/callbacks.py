"""
callbacks.py
------------
Optuna callbacks used during hyperparameter optimisation.
"""
from __future__ import annotations

import optuna


class EarlyStoppingCallback:
    """
    Stop an Optuna study when no improvement is observed for *patience* trials.

    Parameters
    ----------
    patience : int, default=20
        Number of consecutive trials without improvement before stopping.
    min_delta : float, default=1e-4
        Minimum absolute improvement to be considered as progress.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.no_improvement_count: int = 0

    def __call__(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        if trial.value is None:
            return

        current_score = study.best_value

        if self.best_score is None:
            self.best_score = current_score
            return

        improvement = current_score - self.best_score

        if improvement > self.min_delta:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()
