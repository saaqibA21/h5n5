import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, title: str):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("\n" + "="*70)
    print(title)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred, digits=4))
    return acc
