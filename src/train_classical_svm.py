from dataclasses import dataclass
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


@dataclass
class ClassicalResult:
    model: SVC
    best_params: dict
    best_cv_score: float
    test_accuracy: float | None = None
    test_roc_auc: float | None = None
    test_f1: float | None = None


def train_classical_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> ClassicalResult:
    """
    Advanced Classical SVM with:
    - RBF kernel
    - Class balancing
    - Nested CV-style grid search
    - Probability calibration
    - Multiple evaluation metrics
    """

    # Base model
    base_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,      # calibrated later
        random_state=42
    )

    # Hyperparameter grid
    param_grid = {
        "C": [1, 3, 10, 30, 100],
        "gamma": ["scale", 0.1, 0.3, 1.0, 3.0],
    }

    # Stratified CV (important for biological data)
    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=base_svm,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0
    )

    # Fit grid search
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_

    # Probability calibration (Platt scaling)
    calibrated_svm = CalibratedClassifierCV(
        estimator=best_svm,
        method="sigmoid",
        cv=3
    )
    calibrated_svm.fit(X_train, y_train)

    result = ClassicalResult(
        model=calibrated_svm,
        best_params=grid.best_params_,
        best_cv_score=float(grid.best_score_)
    )

    # Optional validation metrics
    if X_val is not None and y_val is not None:
        y_pred = calibrated_svm.predict(X_val)
        y_prob = calibrated_svm.predict_proba(X_val)[:, 1]

        result.test_accuracy = accuracy_score(y_val, y_pred)
        result.test_roc_auc = roc_auc_score(y_val, y_prob)
        result.test_f1 = f1_score(y_val, y_pred)

    return result
