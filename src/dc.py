# src/dc.py
from __future__ import annotations

import numpy as np
from scipy import linalg
from typing import Dict, Hashable, Tuple


def get_Gfanc_dict(anc_ir_dict: Dict[Hashable, np.ndarray], dd: int) -> Dict[Hashable, np.ndarray]:
    if not anc_ir_dict:
        raise ValueError("anc_ir_dict must not be empty.")
    if dd < 1:
        raise ValueError("dd must be >= 1.")

    # Make key order deterministic across runs
    keys = sorted(anc_ir_dict.keys(), key=lambda x: str(x))

    anc_list = []
    r_ref = None
    for k in keys:
        A = np.asarray(anc_ir_dict[k])
        if A.ndim != 2:
            raise ValueError(f"anc_ir_dict[{k}] must be a 2D array.")
        if r_ref is None:
            r_ref = A.shape[0]
        if A.shape[0] != r_ref:
            raise ValueError("All anc_ir matrices must have the same number of rows (anchor samples).")
        anc_list.append(A)

    anc_merged = np.hstack(anc_list)  # shape: (r, sum(d_i))

    U, S, Vt = linalg.svd(anc_merged, full_matrices=False, lapack_driver="gesvd")

    dd_eff = min(dd, S.shape[0])
    # Z shape: (dd, r) — consistent with your original implementation
    Z = (U[:, :dd_eff] * S[:dd_eff]).T  # (dd, r)

    G_fanc = {}
    for k in keys:
        A = np.asarray(anc_ir_dict[k])  # (r, d_k)
        G = (Z @ np.linalg.pinv(A.T)).T  # (d_k, dd)
        G_fanc[k] = G

    return G_fanc


def merge_DC(
    Xtrain_ir: np.ndarray,
    Xtest_ir: np.ndarray,
    anc_ir: np.ndarray,
    G: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xtrain_ir = np.asarray(Xtrain_ir)
    Xtest_ir = np.asarray(Xtest_ir)
    anc_ir = np.asarray(anc_ir)
    G = np.asarray(G)

    Xtrain_hat = Xtrain_ir @ G
    Xtest_hat = Xtest_ir @ G
    anc_hat = anc_ir @ G

    return Xtrain_hat, Xtest_hat, anc_hat
