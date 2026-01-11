from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    ncbi_email: str = os.getenv("NCBI_EMAIL", "")
    ncbi_api_key: str = os.getenv("NCBI_API_KEY", "")

    # Dataset sizing (adjust if needed)
    h5n5_max: int = 350
    negative_max: int = 350

    # Which sequence type?
    # "HA" (hemagglutinin) tends to separate subtypes well.
    target_segment_keyword: str = "hemagglutinin"

    # Feature extraction
    kmer_k: int = 6  # 4-6 are common; 5 often works well
    min_seq_len: int = 800  # filter out too-short fragments
    max_ambiguous_frac: float = 0.05  # allow up to 2% N's

    # Train/test
    test_size: float = 0.2
    random_state: int = 42

    # Quantum compression / qubits
    n_qubits: int = 8  # keep small for simulation speed
    svd_components: int = 8  # must match n_qubits for ZZFeatureMap input

CFG = Config()
