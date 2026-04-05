from dataclasses import dataclass
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

@dataclass
class NaiveBayesResult:
    model: GaussianNB
    test_accuracy: float | None = None
    test_roc_auc: float | None = None
    test_f1: float | None = None

def train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> NaiveBayesResult:
    """
    Gaussian Naive Bayes training.
    Naive Bayes doesn't have many hyperparameters to tune via GridSearch
    but we'll keep the same structure for consistency.
    """
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    result = NaiveBayesResult(model=model)
    
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        result.test_accuracy = accuracy_score(y_val, y_pred)
        result.test_roc_auc = roc_auc_score(y_val, y_prob)
        result.test_f1 = f1_score(y_val, y_pred)
        
    return result
