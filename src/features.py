from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

@dataclass
class FeatureArtifacts:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    scaler: MinMaxScaler

def build_kmer_features(
    sequences: List[str],
    k: int,
    svd_components: int
) -> Tuple[np.ndarray, FeatureArtifacts]:
    """
    1) TF-IDF over character k-mers
    2) TruncatedSVD to svd_components
    3) MinMax scale to [0,1] (nice for quantum feature maps)
    """
    vec = TfidfVectorizer(analyzer="char", ngram_range=(k, k))
    X_sparse = vec.fit_transform(sequences)

    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_low = svd.fit_transform(X_sparse)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_low)

    return X_scaled, FeatureArtifacts(vec, svd, scaler)

def transform_kmer_features(
    sequences: List[str],
    artifacts: FeatureArtifacts
) -> np.ndarray:
    X_sparse = artifacts.vectorizer.transform(sequences)
    X_low = artifacts.svd.transform(X_sparse)
    X_scaled = artifacts.scaler.transform(X_low)
    return X_scaled
