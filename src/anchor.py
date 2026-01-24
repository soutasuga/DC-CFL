# src/anchor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def make_anc_smote(X_pub: pd.DataFrame, r: int, k: int, a: float, rs: int = 0) -> pd.DataFrame:
    if not isinstance(X_pub, pd.DataFrame):
        raise TypeError("X_pub must be a pandas DataFrame.")
    if len(X_pub) == 0:
        raise ValueError("X_pub must not be empty.")
    if r % len(X_pub) != 0:
        raise ValueError("r must be a multiple of len(X_pub).")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if a <= 0:
        raise ValueError("a must be > 0.")

    np.random.seed(rs)

    X_cols = X_pub.columns
    p = len(X_pub)
    samples_per_point = r // p

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pub)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    indices = nbrs.kneighbors(X_scaled, return_distance=False)
    neighbor_indices_all = indices[:, 1:]  # exclude self

    new_samples = []
    for i in range(p):
        x_i = X_scaled[i]
        selected_indices = np.random.choice(neighbor_indices_all[i], size=samples_per_point)
        selected_neighbors = X_scaled[selected_indices]

        for x_j in selected_neighbors:
            c = np.random.uniform(0.0, a)
            new_samples.append(x_i + c * (x_j - x_i))

    X_anc_scaled = np.asarray(new_samples)
    X_anc = scaler.inverse_transform(X_anc_scaled)

    return pd.DataFrame(X_anc, columns=X_cols)
