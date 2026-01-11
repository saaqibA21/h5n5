from dataclasses import dataclass
import numpy as np

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


@dataclass
class QuantumResult:
    model: QSVC
    feature_map_reps: int


def train_quantum_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int
) -> QuantumResult:
    """
    Quantum Support Vector Classifier using a fidelity-based quantum kernel.

    Notes:
    - Feature dimension MUST equal number of qubits.
    - Uses a version-safe kernel constructor (works across Qiskit releases).
    - Runs on quantum simulator backend implicitly.
    """

    # --------------------------------------------------
    # Quantum Feature Map
    # --------------------------------------------------
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2,
        entanglement="full"
    )

    # --------------------------------------------------
    # Fidelity Quantum Kernel (VERSION-SAFE)
    # --------------------------------------------------
    try:
        # Newer Qiskit ML versions
        from qiskit.primitives import Sampler
        sampler = Sampler()
        quantum_kernel = FidelityQuantumKernel(
            feature_map=feature_map,
            sampler=sampler
        )
    except TypeError:
        # Older Qiskit ML versions (no sampler argument)
        quantum_kernel = FidelityQuantumKernel(
            feature_map=feature_map
        )

    # --------------------------------------------------
    # Quantum SVM (QSVC)
    # --------------------------------------------------
    model = QSVC(
        quantum_kernel=quantum_kernel
    )

    model.fit(X_train, y_train)

    return QuantumResult(
        model=model,
        feature_map_reps=2
    )
