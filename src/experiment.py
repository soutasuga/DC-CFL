# src/experiment.py
"""
DC-CFL demo pipeline (single-run / small-scale experimental driver).

This module is intended to keep the notebook minimal:
- The notebook loads data and sets a config
- Then calls `run_dc_cfl_demo(Xall, Yall, config)`
- The core logic lives here (reusable for scripts / repeated trials)

Dependencies:
- src.anchor.make_anc_smote
- src.dc.get_Gfanc_dict, src.dc.merge_DC
- src.models.train_and_evaluate_pytorch, src.models.evaluate_pytorch_model

Notes on dataset-dependent dimensions:
- PCA output dimension is typically set to D-1 (D = number of feature columns).
- Integrated representation dimension can be set from SVD singular values threshold:
    d = #{ k | s_k >= sv_threshold }  (default sv_threshold = 1e-2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import linalg

from src.anchor import make_anc_smote
from src.dc import get_Gfanc_dict, merge_DC
from src.models import train_and_evaluate_pytorch, evaluate_pytorch_model


# ----------------------------
# Configuration (dict-friendly)
# ----------------------------

@dataclass
class DemoConfig:
    # Experiment repetitions
    n_trials: int = 1
    seed_base: int = 0

    # Public split
    public_size: int = 100

    # Client construction
    num_clients: int = 100
    k_labels_per_client: int = 3

    # Client filtering & split
    min_samples: int = 5
    test_ratio: float = 0.2

    # Local representation
    pca_dim: Optional[int] = None  # if None, set to D-1

    # Anchor generation
    anchor_r: int = 1000
    anchor_k: int = 25
    anchor_alpha: float = 1.5

    # Clustering
    tv_threshold: float = 0.2
    linkage_method: str = "complete"

    # Integrated representation dimension rule
    sv_threshold: float = 1e-2
    integrated_dim: Optional[int] = None  # if None, auto from SVD
    integrated_dim_min: int = 1
    integrated_dim_max: Optional[int] = None  # optional safety cap

    # Training hyperparameters (PyTorch MLP wrapper)
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.5
    max_epochs_local: int = 50
    max_epochs_global: int = 50
    max_epochs_cluster: int = 50

    # Verbosity
    verbose: bool = True


def _cfg_from_dict(cfg: Dict[str, Any]) -> DemoConfig:
    base = DemoConfig()
    for k, v in cfg.items():
        if not hasattr(base, k):
            raise KeyError(f"Unknown config key: {k}")
        setattr(base, k, v)
    return base


# ----------------------------
# Helper utilities
# ----------------------------

def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _ensure_series(y: Any, index: pd.Index) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y), index=index)


def _estimate_integrated_dim_from_anchors(
    anchor_irs: Dict[Hashable, np.ndarray],
    sv_threshold: float,
    dd_min: int = 1,
    dd_max: Optional[int] = None,
) -> int:
    """
    Estimate integrated dimension from SVD singular values threshold:
      d = #{ k | s_k >= sv_threshold }.

    Optional dd_max can cap the value for runtime safety.
    """
    if not anchor_irs:
        raise ValueError("anchor_irs must not be empty.")

    keys = sorted(anchor_irs.keys(), key=lambda x: str(x))
    mats = [np.asarray(anchor_irs[k]) for k in keys]

    r_ref = mats[0].shape[0]
    for A in mats:
        if A.ndim != 2:
            raise ValueError("All anchor matrices must be 2D.")
        if A.shape[0] != r_ref:
            raise ValueError("All anchor matrices must share the same number of rows (anchor samples).")

    anc_merged = np.hstack(mats)  # (r, sum d_i)
    _, S, _ = linalg.svd(anc_merged, full_matrices=False, lapack_driver="gesvd")

    dd = int(np.sum(S >= sv_threshold))
    dd = max(dd, dd_min)
    if dd_max is not None:
        dd = min(dd, dd_max)
    dd = max(1, dd)
    return dd


def _build_label_histograms(
    client_train_test: Dict[str, Dict[str, Any]],
    total_num_classes: int,
) -> Dict[str, np.ndarray]:
    """
    Build normalized label histograms from each client's training labels.
    """
    hists: Dict[str, np.ndarray] = {}
    for cid, d in client_train_test.items():
        y_tr = d["Y_train"]
        y_tr = np.asarray(y_tr).astype(int)
        if y_tr.size == 0:
            h = np.zeros(total_num_classes, dtype=float)
        else:
            counts = np.bincount(y_tr, minlength=total_num_classes).astype(float)
            h = counts / counts.sum() if counts.sum() > 0 else np.zeros(total_num_classes, dtype=float)
        hists[cid] = h
    return hists


def _tv_distance_matrix(hists: Dict[str, np.ndarray], client_ids: List[str]) -> np.ndarray:
    """
    TV(p,q) = 0.5 * sum(|p-q|)
    """
    n = len(client_ids)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        pi = hists[client_ids[i]]
        for j in range(i + 1, n):
            pj = hists[client_ids[j]]
            dist = 0.5 * float(np.sum(np.abs(pi - pj)))
            M[i, j] = dist
            M[j, i] = dist
    return M


def _cluster_clients_tv(
    client_train_test: Dict[str, Dict[str, Any]],
    total_num_classes: int,
    tv_threshold: float,
    linkage_method: str,
) -> Tuple[Dict[int, List[str]], Dict[str, int], Any, np.ndarray]:
    """
    Returns:
      clusters_dict: {cluster_id: [client_ids]}
      client_to_cluster: {client_id: cluster_id}
      linked: linkage matrix
      dist_matrix: full TV distance matrix
    """
    client_ids = list(client_train_test.keys())
    hists = _build_label_histograms(client_train_test, total_num_classes)
    dist_matrix = _tv_distance_matrix(hists, client_ids)

    condensed = squareform(dist_matrix)
    linked = linkage(condensed, method=linkage_method)
    cluster_labels = fcluster(linked, t=tv_threshold, criterion="distance")

    clusters_dict: Dict[int, List[str]] = {}
    client_to_cluster: Dict[str, int] = {}
    for cid, cl in zip(client_ids, cluster_labels):
        clusters_dict.setdefault(int(cl), []).append(cid)
        client_to_cluster[cid] = int(cl)

    return clusters_dict, client_to_cluster, linked, dist_matrix


# ----------------------------
# Main pipeline
# ----------------------------

def run_dc_cfl_demo(
    Xall: pd.DataFrame,
    Yall: pd.Series,
    config: Dict[str, Any] | DemoConfig,
) -> Dict[str, Any]:
    """
    Run a DC-CFL-style evaluation demo.

    Inputs:
      Xall: features for all samples (DataFrame)
      Yall: labels for all samples (Series), expected to be 0-based integers
      config: DemoConfig or dict with DemoConfig keys

    Outputs (dict):
      - local_results: DataFrame(trial, client_id, local_acc)
      - dc_results: DataFrame(trial, client_id, cluster, dc_global, dc_cluster)
      - merged_results: DataFrame joined local+dc
      - per_trial_mean: DataFrame (trial means)
      - overall_mean: Series (overall means)
      - clusters: dict(cluster_id -> list of client_ids) from the last trial
      - linkage: linkage matrix from the last trial
      - tv_distance_matrix: TV distance matrix from the last trial
      - config_used: DemoConfig (resolved)
    """
    cfg = _cfg_from_dict(config) if isinstance(config, dict) else config

    if not isinstance(Xall, pd.DataFrame):
        raise TypeError("Xall must be a pandas DataFrame.")
    if not isinstance(Yall, pd.Series):
        Yall = pd.Series(np.asarray(Yall), index=Xall.index)

    # Basic label sanity
    y_np = np.asarray(Yall).astype(int)
    if y_np.min(initial=0) < 0:
        raise ValueError("Yall must be non-negative integer labels (preferably 0-based).")

    total_num_classes = int(pd.Series(y_np).nunique())

    # PCA dimension rule (default D-1)
    D = int(Xall.shape[1])
    pca_dim = cfg.pca_dim if cfg.pca_dim is not None else max(1, D - 1)

    # Collect results across trials
    local_rows: List[Dict[str, Any]] = []
    dc_rows: List[Dict[str, Any]] = []

    last_clusters: Dict[int, List[str]] = {}
    last_linkage = None
    last_tv_matrix = None

    for t in range(cfg.n_trials):
        base_seed = cfg.seed_base + (t * 10)
        if cfg.verbose:
            print("\n" + "=" * 80)
            print(f"TRIAL {t + 1}/{cfg.n_trials} (Base Seed: {base_seed})")
            print("=" * 80)

        _set_seeds(base_seed)

        # -------------------------
        # Step 1/2: Public split
        # -------------------------
        X_others, X_public, Y_others, Y_public = train_test_split(
            Xall,
            Yall,
            test_size=cfg.public_size,
            random_state=base_seed,
            stratify=Yall,
        )
        Y_others = _ensure_series(Y_others, X_others.index)

        # -------------------------
        # Step 3: Assign labels to clients + distribute indices
        # -------------------------
        _set_seeds(base_seed + 1)

        all_labels = sorted(Y_others.unique())
        indices_per_label = {lab: Y_others.index[Y_others == lab].tolist() for lab in all_labels}

        # Each client gets K distinct labels (design-time)
        client_label_assignments: Dict[int, List[int]] = {}
        for client_idx in range(cfg.num_clients):
            assigned = np.random.choice(all_labels, cfg.k_labels_per_client, replace=False)
            client_label_assignments[client_idx] = list(map(int, assigned))

        # Map label -> clients that own it
        label_to_clients: Dict[int, List[int]] = {lab: [] for lab in all_labels}
        for client_idx, labs in client_label_assignments.items():
            for lab in labs:
                label_to_clients[int(lab)].append(client_idx)

        # Distribute indices for each label evenly among owning clients
        client_indices: Dict[int, List[int]] = {i: [] for i in range(cfg.num_clients)}
        for lab, idxs in indices_per_label.items():
            owners = label_to_clients.get(int(lab), [])
            if not owners:
                continue
            shuffled = idxs.copy()
            random.shuffle(shuffled)
            chunks = np.array_split(shuffled, len(owners))
            for owner, chunk in zip(owners, chunks):
                if chunk.size > 0:
                    client_indices[owner].extend(chunk.tolist())

        # Build raw client data
        client_data: Dict[str, Dict[str, Any]] = {}
        for client_idx in range(cfg.num_clients):
            cname = f"client_{client_idx}"
            idxs = client_indices.get(client_idx, [])
            if not idxs:
                client_data[cname] = {"X": pd.DataFrame(), "Y": pd.Series(dtype=int)}
            else:
                client_data[cname] = {
                    "X": X_others.loc[idxs],
                    "Y": Y_others.loc[idxs],
                }

        # -------------------------
        # Step 3.5: Filter clients that actually have exactly K labels
        # -------------------------
        client_data = {
            cname: d for cname, d in client_data.items()
            if int(pd.Series(d["Y"]).nunique()) == cfg.k_labels_per_client
        }

        # -------------------------
        # Step 4: Train/Test split per client
        # -------------------------
        client_train_test: Dict[str, Dict[str, Any]] = {}
        for cid, d in client_data.items():
            X_c, Y_c = d["X"], d["Y"]
            if len(X_c) < cfg.min_samples:
                continue
            try:
                X_tr, X_te, Y_tr, Y_te = train_test_split(
                    X_c,
                    Y_c,
                    test_size=cfg.test_ratio,
                    random_state=base_seed + 2,
                    stratify=None,
                )
            except ValueError:
                continue

            if X_tr.empty:
                continue

            client_train_test[cid] = {
                "X_train": X_tr,
                "Y_train": Y_tr,
                "X_test": X_te,
                "Y_test": Y_te,
            }

        if cfg.verbose:
            print(f"Valid clients after split: {len(client_train_test)}")

        if len(client_train_test) == 0:
            # No valid clients => return empty results for this trial
            continue

        # -------------------------
        # Step 5: Local representations (scaler + PCA) and Local model eval
        # -------------------------
        # PCA feasibility filter: need n_train >= pca_dim
        client_train_test = {
            cid: d for cid, d in client_train_test.items()
            if int(d["X_train"].shape[0]) >= pca_dim
        }

        if cfg.verbose:
            print(f"Valid clients after PCA feasibility check: {len(client_train_test)}")

        if len(client_train_test) == 0:
            continue

        # Anchor generation from public data
        X_anchor = make_anc_smote(
            X_pub=X_public,
            r=cfg.anchor_r,
            k=cfg.anchor_k,
            a=cfg.anchor_alpha,
            rs=base_seed + 3,
        )

        # Build intermediate representations for each client
        client_ir: Dict[str, Dict[str, np.ndarray]] = {}
        for cid, d in client_train_test.items():
            scaler = StandardScaler()
            X_tr = d["X_train"].to_numpy()
            X_te = d["X_test"].to_numpy()

            X_tr_scaled = scaler.fit_transform(X_tr)

            pca = PCA(n_components=pca_dim, random_state=base_seed + 4)
            pca.fit(X_tr_scaled)

            # Train IR
            train_ir = pca.transform(X_tr_scaled)

            # Test IR (may be empty)
            if X_te.shape[0] == 0:
                test_ir = np.zeros((0, pca_dim), dtype=float)
                X_te_scaled = np.zeros((0, X_tr_scaled.shape[1]), dtype=float)
            else:
                X_te_scaled = scaler.transform(X_te)
                test_ir = pca.transform(X_te_scaled)

            # Anchor IR (always computed)
            anc_ir = pca.transform(scaler.transform(X_anchor.to_numpy()))

            client_ir[cid] = {
                "train_ir": train_ir,
                "test_ir": test_ir,
                "anc_ir": anc_ir,
                # keep scalers? not needed beyond this stage
            }

            # Local model evaluation (on scaled raw features, consistent with your prior code)
            metrics, _ = train_and_evaluate_pytorch(
                X_train=X_tr_scaled,
                y_train=np.asarray(d["Y_train"]).astype(int),
                X_test=X_te_scaled,
                y_test=np.asarray(d["Y_test"]).astype(int),
                total_num_classes=total_num_classes,
                seed=base_seed + 5,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                momentum=cfg.momentum,
                max_epochs=cfg.max_epochs_local,
            )
            local_rows.append({
                "trial": t,
                "client_id": cid,
                "local_acc": metrics.get("Accuracy", float("nan")),
            })

        # -------------------------
        # Step 6: Clustering by label-distribution TV distance
        # -------------------------
        clusters_dict, client_to_cluster, linked, tv_matrix = _cluster_clients_tv(
            client_train_test=client_train_test,
            total_num_classes=total_num_classes,
            tv_threshold=cfg.tv_threshold,
            linkage_method=cfg.linkage_method,
        )

        last_clusters = clusters_dict
        last_linkage = linked
        last_tv_matrix = tv_matrix

        if cfg.verbose:
            print(f"Clusters (TV threshold={cfg.tv_threshold}): {len(clusters_dict)}")

        # -------------------------
        # Step 7: Global DC model
        # -------------------------
        # Determine integrated dimension (global) if not specified
        anchor_irs_global = {cid: ir["anc_ir"] for cid, ir in client_ir.items()}
        if cfg.integrated_dim is None:
            dd_global = _estimate_integrated_dim_from_anchors(
                anchor_irs=anchor_irs_global,
                sv_threshold=cfg.sv_threshold,
                dd_min=cfg.integrated_dim_min,
                dd_max=cfg.integrated_dim_max,
            )
        else:
            dd_global = int(cfg.integrated_dim)

        if cfg.verbose:
            print(f"Integrated dim (global): {dd_global}")

        G_global = get_Gfanc_dict(anchor_irs_global, dd=dd_global)

        global_train_list: List[np.ndarray] = []
        global_y_train_list: List[pd.Series] = []
        global_test_map: Dict[str, np.ndarray] = {}

        for cid, ir in client_ir.items():
            train_hat, test_hat, _ = merge_DC(
                ir["train_ir"],
                ir["test_ir"],
                ir["anc_ir"],
                G_global[cid],
            )
            global_train_list.append(train_hat)
            global_y_train_list.append(client_train_test[cid]["Y_train"])
            global_test_map[cid] = test_hat

        X_train_global = np.vstack(global_train_list)
        y_train_global = pd.concat(global_y_train_list).to_numpy().astype(int)

        scaler_global = StandardScaler()
        X_train_global_std = scaler_global.fit_transform(X_train_global)

        # Train global model (no test during training; evaluation is per-client)
        _, model_global = train_and_evaluate_pytorch(
            X_train=X_train_global_std,
            y_train=y_train_global,
            X_test=np.array([]).reshape(0, dd_global),
            y_test=np.array([]).astype(int),
            total_num_classes=total_num_classes,
            seed=base_seed + 6,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            momentum=cfg.momentum,
            max_epochs=cfg.max_epochs_global,
        )

        global_scores: Dict[str, float] = {}
        for cid in client_ir.keys():
            X_te_hat = global_test_map[cid]
            y_te = np.asarray(client_train_test[cid]["Y_test"]).astype(int)

            if X_te_hat.shape[0] == 0 or y_te.size == 0:
                global_scores[cid] = float("nan")
                continue

            X_te_std = scaler_global.transform(X_te_hat)
            acc = evaluate_pytorch_model(model_global, X_te_std, y_te, batch_size=cfg.batch_size)
            global_scores[cid] = float(acc)

        # -------------------------
        # Step 8: Cluster-wise DC models
        # -------------------------
        for cluster_id, members in clusters_dict.items():
            # If a cluster is too small, fall back to local score (as in your code)
            if len(members) < 2:
                for cid in members:
                    dc_rows.append({
                        "trial": t,
                        "client_id": cid,
                        "cluster": cluster_id,
                        "dc_global": global_scores.get(cid, float("nan")),
                        "dc_cluster": float("nan"),  # will be replaced by local in merged view if desired
                        "cluster_model_used": False,
                    })
                continue

            # Cluster integrated dimension (optional: recompute per cluster)
            cluster_anchor_irs = {cid: client_ir[cid]["anc_ir"] for cid in members}
            if cfg.integrated_dim is None:
                dd_cluster = _estimate_integrated_dim_from_anchors(
                    anchor_irs=cluster_anchor_irs,
                    sv_threshold=cfg.sv_threshold,
                    dd_min=cfg.integrated_dim_min,
                    dd_max=cfg.integrated_dim_max,
                )
            else:
                dd_cluster = int(cfg.integrated_dim)

            G_cluster = get_Gfanc_dict(cluster_anchor_irs, dd=dd_cluster)

            X_tr_list: List[np.ndarray] = []
            y_tr_list: List[pd.Series] = []
            test_map: Dict[str, np.ndarray] = {}

            for cid in members:
                ir = client_ir[cid]
                train_hat, test_hat, _ = merge_DC(
                    ir["train_ir"],
                    ir["test_ir"],
                    ir["anc_ir"],
                    G_cluster[cid],
                )
                X_tr_list.append(train_hat)
                y_tr_list.append(client_train_test[cid]["Y_train"])
                test_map[cid] = test_hat

            X_tr_cluster = np.vstack(X_tr_list)
            y_tr_cluster = pd.concat(y_tr_list).to_numpy().astype(int)

            scaler_cluster = StandardScaler()
            X_tr_cluster_std = scaler_cluster.fit_transform(X_tr_cluster)

            _, model_cluster = train_and_evaluate_pytorch(
                X_train=X_tr_cluster_std,
                y_train=y_tr_cluster,
                X_test=np.array([]).reshape(0, dd_cluster),
                y_test=np.array([]).astype(int),
                total_num_classes=total_num_classes,
                seed=base_seed + 7,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                momentum=cfg.momentum,
                max_epochs=cfg.max_epochs_cluster,
            )

            for cid in members:
                X_te_hat = test_map[cid]
                y_te = np.asarray(client_train_test[cid]["Y_test"]).astype(int)

                if X_te_hat.shape[0] == 0 or y_te.size == 0:
                    acc_c = float("nan")
                else:
                    X_te_std = scaler_cluster.transform(X_te_hat)
                    acc_c = float(evaluate_pytorch_model(model_cluster, X_te_std, y_te, batch_size=cfg.batch_size))

                dc_rows.append({
                    "trial": t,
                    "client_id": cid,
                    "cluster": cluster_id,
                    "dc_global": global_scores.get(cid, float("nan")),
                    "dc_cluster": acc_c,
                    "cluster_model_used": True,
                })

        if cfg.verbose:
            print(f"TRIAL {t + 1} finished.")

    # -------------------------
    # Assemble outputs
    # -------------------------
    df_local = pd.DataFrame(local_rows)
    df_dc = pd.DataFrame(dc_rows)

    if df_local.empty or df_dc.empty:
        merged = pd.DataFrame()
        per_trial_mean = pd.DataFrame()
        overall_mean = pd.Series(dtype=float)
    else:
        merged = pd.merge(df_dc, df_local, on=["trial", "client_id"], how="left")

        # Optional: if cluster model not used, you may want dc_cluster to fall back to local_acc
        # (matching your original logic for small clusters)
        mask_fallback = (merged["cluster_model_used"] == False) & merged["dc_cluster"].isna()
        if mask_fallback.any():
            merged.loc[mask_fallback, "dc_cluster"] = merged.loc[mask_fallback, "local_acc"]

        per_trial_mean = merged.groupby("trial")[["local_acc", "dc_global", "dc_cluster"]].mean().round(4)
        overall_mean = merged[["local_acc", "dc_global", "dc_cluster"]].mean().round(4)

    return {
        "local_results": df_local,
        "dc_results": df_dc,
        "merged_results": merged,
        "per_trial_mean": per_trial_mean,
        "overall_mean": overall_mean,
        "clusters": last_clusters,
        "linkage": last_linkage,
        "tv_distance_matrix": last_tv_matrix,
        "config_used": cfg,
    }
