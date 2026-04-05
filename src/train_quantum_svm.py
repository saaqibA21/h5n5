from dataclasses import dataclass
import numpy as np

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.calibration import CalibratedClassifierCV

@dataclass
class QuantumResult:
    model: CalibratedClassifierCV | QSVC
    feature_map_reps: int

def train_quantum_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int
) -> QuantumResult:
    """
    Quantum Support Vector Classifier with:
    - Deepened ZZFeatureMap (reps=3) for better feature extraction.
    - Calibrated probabilities using Platt Scaling (Sigmoid) for higher confidence.
    """

    # Optimized Quantum Feature Map with increased depth
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=3,
        entanglement="full"
    )

    # Fidelity Quantum Kernel
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    # Base QSVC implementation
    base_qsvc = QSVC(
        quantum_kernel=quantum_kernel
    )

    # Wrap in Calibrated Classifier to get high-confidence probabilities
    model = CalibratedClassifierCV(
        base_qsvc, 
        cv=3, 
        method='sigmoid'
    )

    model.fit(X_train, y_train)

    return QuantumResult(
        model=model,
        feature_map_reps=3
    )