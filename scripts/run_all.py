import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from src.config import CFG
from src.ncbi_fetch import download_h5n5_and_negative
from src.preprocess import filter_sequences
from src.features import build_kmer_features
from src.train_classical_svm import train_classical_svm
from src.train_quantum_svm import train_quantum_svm
from src.evaluate import evaluate

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1) Download
    print("Downloading sequences from NCBI...")
    pos, neg = download_h5n5_and_negative(
        email=CFG.ncbi_email,
        api_key=CFG.ncbi_api_key,
        segment_keyword=CFG.target_segment_keyword,
        h5n5_max=CFG.h5n5_max,
        negative_max=CFG.negative_max
    )

    # 2) Clean/filter
    pos_f = filter_sequences(pos, CFG.min_seq_len, CFG.max_ambiguous_frac)
    neg_f = filter_sequences(neg, CFG.min_seq_len, CFG.max_ambiguous_frac)

    print(f"Filtered H5N5: {len(pos_f)}")
    print(f"Filtered NEG : {len(neg_f)}")

    # Keep balanced for accuracy stability
    n = min(len(pos_f), len(neg_f))
    pos_f = pos_f[:n]
    neg_f = neg_f[:n]

    rows = []
    for acc, seq in pos_f:
        rows.append({"accession": acc, "sequence": seq, "label": 1})
    for acc, seq in neg_f:
        rows.append({"accession": acc, "sequence": seq, "label": 0})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=CFG.random_state).reset_index(drop=True)
    df.to_csv("data/processed/dataset.csv", index=False)
    print("Saved: data/processed/dataset.csv")

    sequences = df["sequence"].tolist()
    y = df["label"].values.astype(int)

    # 3) Features (k-mers -> SVD -> scale)
    X, artifacts = build_kmer_features(
        sequences=sequences,
        k=CFG.kmer_k,
        svd_components=CFG.svd_components
    )

    # Save feature artifacts
    dump(artifacts, "data/processed/feature_artifacts.joblib")

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.test_size,
        random_state=CFG.random_state,
        stratify=y
    )

    # 5) Classical SVM
    print("\nTraining Classical SVM (RBF) with grid search...")
    classical = train_classical_svm(X_train, y_train)
    dump(classical.model, "data/processed/classical_svm.joblib")
    print("Best CV:", classical.best_cv_score, "Params:", classical.best_params)
    acc_c = evaluate(classical.model, X_test, y_test, "Classical SVM (RBF)")

    # 6) Quantum SVM (QSVC)
    # Ensure input dimension == n_qubits
    if X_train.shape[1] != CFG.n_qubits:
        raise ValueError(f"Feature dim {X_train.shape[1]} != n_qubits {CFG.n_qubits}. "
                         f"Set CFG.svd_components = CFG.n_qubits.")

    print("\nTraining Quantum SVM (QSVC, quantum kernel)...")
    quantum = train_quantum_svm(X_train, y_train, n_qubits=CFG.n_qubits)
    dump(quantum.model, "data/processed/quantum_qsvc.joblib")
    acc_q = evaluate(quantum.model, X_test, y_test, "Quantum SVM (QSVC + Fidelity Quantum Kernel)")

    # 7) Save summary
    summary = {
        "n_samples": int(len(df)),
        "n_pos": int(df["label"].sum()),
        "n_neg": int((df["label"] == 0).sum()),
        "kmer_k": CFG.kmer_k,
        "svd_components": CFG.svd_components,
        "classical_acc": float(acc_c),
        "quantum_acc": float(acc_q),
        "classical_best_params": classical.best_params,
    }
    with open("data/processed/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary: data/processed/run_summary.json")

if __name__ == "__main__":
    main()
