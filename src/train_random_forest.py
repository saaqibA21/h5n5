from dataclasses import dataclass
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


@dataclass
class RandomForestResult:
    model: RandomForestClassifier
    best_params: dict
    best_cv_score: float
    test_accuracy: float | None = None
    test_roc_auc: float | None = None
    test_f1: float | None = None


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> RandomForestResult:
    """
    Advanced Random Forest with:
    - Balanced class weighting
    - Grid search for n_estimators and max_depth
    - Stratified CV
    - Probability support (native)
    """

    # Base model
    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    # Stratified CV
    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0
    )

    # Fit grid search
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_

    result = RandomForestResult(
        model=best_rf,
        best_params=grid.best_params_,
        best_cv_score=float(grid.best_score_)
    )

    # Optional validation metrics
    if X_val is not None and y_val is not None:
        y_pred = best_rf.predict(X_val)
        y_prob = best_rf.predict_proba(X_val)[:, 1]

        result.test_accuracy = accuracy_score(y_val, y_pred)
        result.test_roc_auc = roc_auc_score(y_val, y_prob)
        result.test_f1 = f1_score(y_val, y_pred)

    return result
