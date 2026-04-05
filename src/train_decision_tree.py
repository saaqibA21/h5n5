from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

@dataclass
class DecisionTreeResult:
    model: DecisionTreeClassifier
    best_params: dict
    best_cv_score: float
    test_accuracy: float | None = None
    test_roc_auc: float | None = None
    test_f1: float | None = None

def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> DecisionTreeResult:
    """
    Advanced Decision Tree with grid search.
    """
    
    base_dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )
    
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
    }
    
    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    grid = GridSearchCV(
        estimator=base_dt,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    best_dt = grid.best_estimator_
    
    result = DecisionTreeResult(
        model=best_dt,
        best_params=grid.best_params_,
        best_cv_score=float(grid.best_score_)
    )
    
    if X_val is not None and y_val is not None:
        y_pred = best_dt.predict(X_val)
        y_prob = best_dt.predict_proba(X_val)[:, 1]
        
        result.test_accuracy = accuracy_score(y_val, y_pred)
        result.test_roc_auc = roc_auc_score(y_val, y_prob)
        result.test_f1 = f1_score(y_val, y_pred)
        
    return result
