#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_clustering_pipeline.py

End-to-end hybrid hierarchical clustering pipeline (no CLI flags).
- Expects walmart.csv in the same folder (transaction-level).
- Produces outputs_final/ with user clusters, summaries, plots.
- Combines the best ideas from previous experiments:
  * enriched features: category count shares + revenue shares, entropy, n_unique_categories
  * log1p transforms for purchase metrics
  * robust scaling, PCA by variance target
  * hybrid workflow: detect niche clusters (average+manhattan) -> remove -> recluster core (complete+manhattan) -> reattach niches

Run:
  python final_clustering_pipeline.py
"""
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# -------------------------
# Config (tweak if needed)
# -------------------------
TRANSACTIONS_CSV = "walmart.csv"          # input (transaction-level)
OUTDIR = "outputs_final"                  # final outputs
TMP_DIR = "outputs_tmp"                   # temporary outputs
# Winsorization quantiles (transaction-level, then user-level during user features winsorize)
TX_WINSORIZE_LOWER = 0.01
TX_WINSORIZE_UPPER = 0.99
USER_WINSORIZE_LOWER_INITIAL = 0.005     # for initial detection (moderate)
USER_WINSORIZE_UPPER_INITIAL = 0.995
USER_WINSORIZE_LOWER_CORE = 0.01         # for core reclustering (stronger)
USER_WINSORIZE_UPPER_CORE = 0.99
# PCA variance targets
PCA_VARIANCE_DETECT = 0.80
PCA_VARIANCE_CORE = 0.90
# thresholds to call small clusters "outliers"/niches
MIN_CLUSTER_PCT = 0.01   # 1% of total users
MIN_CLUSTER_SIZE = 50    # or absolute minimum
# Candidate ks for detection (we use k=4 to detect niches)
DETECT_K = 4
# Core recluster settings
CORE_LINKAGE = "complete"
CORE_METRIC = "manhattan"
CORE_K = 4

# -------------------------
# Helpers
# -------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def to_dense_if_needed(X):
    return X.toarray() if hasattr(X, "toarray") else X

def metric_for_scipy(metric):
    mapping = {"manhattan": "cityblock", "l1": "cityblock", "l2": "euclidean"}
    return mapping.get(metric, metric)

def compute_linkage_matrix(X_dense: np.ndarray, metric: str, linkage_method: str):
    """
    Build linkage matrix for dendrogram:
     - For ward linkage, metric must be 'euclidean' and linkage() can accept data directly.
     - Otherwise compute condensed distance matrix via scipy.spatial.distance.pdist using mapped metric.
    """
    scipy_metric = metric_for_scipy(metric)
    if linkage_method == "ward":
        if metric != "euclidean":
            raise ValueError("Ward linkage requires Euclidean metric.")
        Z = linkage(X_dense, method="ward")
    else:
        dm = pdist(X_dense, metric=scipy_metric)
        Z = linkage(dm, method=linkage_method)
    return Z

def fit_agglomerative(X: np.ndarray, linkage_method: str, metric: str, k: int):
    if linkage_method == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, metric=metric)
    return model.fit_predict(X)

def shannon_entropy(counts: np.ndarray):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

# -------------------------
# Data reading + initial cleaning
# -------------------------
def read_transactions(path=TRANSACTIONS_CSV, q_low=TX_WINSORIZE_LOWER, q_high=TX_WINSORIZE_UPPER):
    dtype_map = {
        "User_ID": "int32",
        "Product_ID": "string",
        "Gender": "category",
        "Age": "category",
        "Occupation": "int16",
        "City_Category": "category",
        "Stay_In_Current_City_Years": "string",
        "Marital_Status": "int8",
        "Product_Category": "float32",
        "Purchase": "float32",
    }
    usecols = ["User_ID","Product_ID","Gender","Age","Occupation","City_Category",
               "Stay_In_Current_City_Years","Marital_Status","Product_Category","Purchase"]
    print(f"Reading transactions from {path} ...")
    df = pd.read_csv(path, dtype=dtype_map, usecols=usecols)
    df = df.drop_duplicates(ignore_index=True)
    # Normalize and cast
    df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].astype("string").str.replace("+", "", regex=False)
    df["Stay_In_Current_City_Years"] = pd.to_numeric(df["Stay_In_Current_City_Years"], errors="coerce").astype("Int8")
    pc = pd.to_numeric(df["Product_Category"], errors="coerce").astype("Int16")
    df["Product_Category"] = pc
    df["Product_Category_str"] = df["Product_Category"].astype("Int16").astype("string")
    df["Product_Category_str"] = df["Product_Category_str"].fillna("unknown").replace({"<NA>":"unknown"})
    df["Product_Category_str"] = "cat_" + df["Product_Category_str"]
    # Missing handling
    df["Purchase"] = df["Purchase"].fillna(0.0).astype("float64")
    for c in ["Gender","Age","City_Category"]:
        if df[c].isna().any():
            df[c] = df[c].cat.add_categories(["Unknown"]).fillna("Unknown")
    for c in ["Occupation","Marital_Status","Stay_In_Current_City_Years"]:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else 0
            df[c] = df[c].fillna(fill_val).astype(df[c].dtype)
    # Winsorize Purchase
    if 0 < q_low < q_high < 1:
        low = df["Purchase"].quantile(q_low)
        high = df["Purchase"].quantile(q_high)
        df["Purchase"] = df["Purchase"].clip(lower=low, upper=high)
    print("Transactions read and cleaned.")
    return df

# -------------------------
# Feature engineering (user-level)
# -------------------------
def build_user_features(df: pd.DataFrame, top_cat: int = 10):
    print("Building user-level features (aggregations, shares, entropy, log1p)...")
    grp = df.groupby("User_ID", observed=True)
    agg_basic = pd.DataFrame({
        "n_transactions": grp.size(),
        "total_purchase": grp["Purchase"].sum(),
        "avg_purchase": grp["Purchase"].mean(),
        "median_purchase": grp["Purchase"].median(),
        "std_purchase": grp["Purchase"].std(ddof=1),
        "unique_products": grp["Product_ID"].nunique(),
    }).reset_index()
    agg_basic["std_purchase"] = agg_basic["std_purchase"].fillna(0.0)
    # modes
    def series_mode(s):
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else np.nan
    modes = grp.agg({
        "Age": series_mode,
        "Gender": series_mode,
        "City_Category": series_mode,
        "Stay_In_Current_City_Years": series_mode,
        "Marital_Status": series_mode,
        "Occupation": series_mode,
    }).reset_index().rename(columns={
        "Age":"Age_mode","Gender":"Gender_mode","City_Category":"City_Category_mode",
        "Stay_In_Current_City_Years":"Stay_mode","Marital_Status":"Marital_mode","Occupation":"Occupation_mode"
    })
    # categories (top + other)
    top_cats = df["Product_Category_str"].value_counts(dropna=False).head(top_cat).index.tolist()
    cat_col = df["Product_Category_str"].where(df["Product_Category_str"].isin(top_cats), other="cat_other")
    df_tmp = df.assign(Product_Category_top=cat_col)
    cat_pivot = df_tmp.pivot_table(index="User_ID", columns="Product_Category_top", values="Product_ID", aggfunc="count", fill_value=0, observed=True).astype("int32")
    cat_share = cat_pivot.div(cat_pivot.sum(axis=1).replace(0,1), axis=0)
    cat_share.columns = [f"cat_share__{c}" for c in cat_share.columns]
    cat_share = cat_share.reset_index()
    # revenue share
    rev_pivot = df_tmp.pivot_table(index="User_ID", columns="Product_Category_top", values="Purchase", aggfunc="sum", fill_value=0.0, observed=True).astype("float64")
    rev_share = rev_pivot.div(rev_pivot.sum(axis=1).replace(0,1), axis=0)
    rev_share.columns = [f"cat_rev_share__{c}" for c in rev_share.columns]
    rev_share = rev_share.reset_index()
    # entropy + unique categories
    cat_entropy = pd.DataFrame({
        "User_ID": cat_pivot.index,
        "cat_entropy": [shannon_entropy(row.values) for _, row in cat_pivot.iterrows()],
        "n_unique_categories": (cat_pivot > 0).sum(axis=1).astype("int16"),
    }).reset_index(drop=True)
    # merge
    features = (agg_basic.merge(modes, on="User_ID", how="left")
                       .merge(cat_share, on="User_ID", how="left")
                       .merge(rev_share, on="User_ID", how="left")
                       .merge(cat_entropy, on="User_ID", how="left"))
    # log1p for purchase metrics (keep n_transactions)
    for col in ["avg_purchase","median_purchase","std_purchase"]:
        features[f"{col}_log1p"] = np.log1p(features[col].astype("float64"))
    # NA handling for categorical modes
    for col in ["Age_mode","Gender_mode","City_Category_mode"]:
        if col in features.columns:
            if isinstance(features[col].dtype, CategoricalDtype):
                features[col] = features[col].cat.add_categories(["Unknown"]).fillna("Unknown")
            else:
                features[col] = features[col].astype("string").fillna("Unknown")
    for col, dtype in [("Stay_mode","int16"),("Marital_mode","int16"),("Occupation_mode","int16")]:
        if col in features.columns:
            features[col] = features[col].fillna(-1).astype(dtype)
    # numeric NAs -> 0
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[num_cols] = features[num_cols].fillna(0)
    print("User features built.")
    return features, cat_pivot

# -------------------------
# Transform + PCA helpers
# -------------------------
def build_transformer(features: pd.DataFrame):
    id_col = "User_ID"
    numeric_cols = ["n_transactions","unique_products","cat_entropy","n_unique_categories"]
    numeric_cols += [c for c in features.columns if c.startswith("cat_share__")]
    numeric_cols += [c for c in features.columns if c.startswith("cat_rev_share__")]
    for base in ["avg_purchase","median_purchase","std_purchase"]:
        col = f"{base}_log1p"
        if col in features.columns:
            numeric_cols.append(col)
    cat_cols = ["Age_mode","Gender_mode","City_Category_mode","Stay_mode","Marital_mode","Occupation_mode"]
    num_transformer = RobustScaler(with_centering=True, with_scaling=True, unit_variance=False, quantile_range=(25.0,75.0))
    cat_transformer = make_one_hot_encoder()
    preproc = ColumnTransformer(transformers=[("num", num_transformer, numeric_cols), ("cat", cat_transformer, cat_cols)], remainder="drop", sparse_threshold=0.3)
    return preproc, id_col, numeric_cols, cat_cols

# -------------------------
# Dendrogram + plotting
# -------------------------
def save_dendrogram(X_for_clust, linkage_method, metric, outpath, truncate_p=50):
    print(f"Computing linkage for dendrogram (linkage={linkage_method}, metric={metric}) ...")
    Z = compute_linkage_matrix(X_for_clust, metric=metric, linkage_method=linkage_method)
    plt.figure(figsize=(14,6))
    dendrogram(Z, truncate_mode="lastp", p=truncate_p, leaf_rotation=90., leaf_font_size=10., show_contracted=True, above_threshold_color="gray")
    plt.title(f"Dendrogram (linkage={linkage_method}, metric={metric}) - last {truncate_p} merges")
    plt.xlabel("Merged clusters (truncated)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_2d_plot(X_dense, labels, outpath, explained_var):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca2.fit_transform(X_dense)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels, palette="tab10", s=18, linewidth=0, alpha=0.8)
    plt.title(f"Clusters (PCA 2D) – explained var={explained_var:.2f}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(title="cluster", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_3d_plot(X_dense, labels, out_html, out_png):
    try:
        import plotly.express as px
        pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
        X3 = pca3.fit_transform(X_dense)
        df_plot = pd.DataFrame({"PC1":X3[:,0], "PC2":X3[:,1], "PC3":X3[:,2], "cluster":labels})
        fig = px.scatter_3d(df_plot, x="PC1", y="PC2", z="PC3", color="cluster", opacity=0.8)
        fig.update_layout(title="Clusters (PCA 3D)")
        fig.write_html(out_html)
    except Exception:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
        X3 = pca3.fit_transform(X_dense)
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=labels, cmap="tab10", s=10, alpha=0.85)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

# -------------------------
# Summaries
# -------------------------
def profile_clusters(user_cluster_df: pd.DataFrame, labels_col="cluster"):
    numeric_cols = user_cluster_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("User_ID", labels_col)]
    summary = user_cluster_df.groupby(labels_col)[numeric_cols].agg(["count","mean","median"])
    summary.columns = [f"{a}__{b}" for a,b in summary.columns]
    return summary

# -------------------------
# Hybrid workflow (main)
# -------------------------
def run_hybrid_pipeline():
    ensure_dir(OUTDIR); ensure_dir(TMP_DIR)
    # 1) Read transactions with transaction-level winsorize (moderate)
    tx_df = read_transactions(TRANSACTIONS_CSV, q_low=TX_WINSORIZE_LOWER, q_high=TX_WINSORIZE_UPPER)

    # 2) Build user features (initial)
    feats, cat_pivot = build_user_features(tx_df, top_cat=10)

    # 3) Winsorize user-level features (initial detection)
    def winsorize_user(feats_df, low, high):
        dfc = feats_df.copy()
        num_cols = [c for c in dfc.columns if c != "User_ID" and pd.api.types.is_numeric_dtype(dfc[c])]
        for c in num_cols:
            lower = dfc[c].quantile(low)
            upper = dfc[c].quantile(high)
            dfc[c] = dfc[c].clip(lower, upper)
        return dfc
    feats_detect = winsorize_user(feats, USER_WINSORIZE_LOWER_INITIAL, USER_WINSORIZE_UPPER_INITIAL)

    # 4) Preprocess (scale+one-hot) and PCA for detection
    preproc, id_col, numeric_cols, cat_cols = build_transformer(feats_detect)
    X_raw = feats_detect.drop(columns=[id_col])
    X_trans = preproc.fit_transform(X_raw)
    X_dense = to_dense_if_needed(X_trans)
    # PCA variance target for detection
    pca_detect = PCA(n_components=PCA_VARIANCE_DETECT, random_state=RANDOM_STATE)
    X_for_detect = pca_detect.fit_transform(X_dense)
    explained_detect = pca_detect.explained_variance_ratio_.sum()
    print(f"PCA detection explained variance: {explained_detect:.3f}")

    # 5) Detect niches/outliers using average+manhattan, k=DETECT_K
    detect_linkage, detect_metric, detect_k = "average", "manhattan", DETECT_K
    print(f"Running detection clustering: {detect_linkage}+{detect_metric}, k={detect_k}")
    labels_detect = fit_agglomerative(X_for_detect, detect_linkage, detect_metric, detect_k)
    feats_detect["detect_cluster"] = labels_detect
    out_det_dir = os.path.join(TMP_DIR, "detection")
    ensure_dir(out_det_dir)
    feats_detect.to_csv(os.path.join(out_det_dir, "user_clusters_detect.csv"), index=False)
    counts = feats_detect["detect_cluster"].value_counts().sort_index()
    print("Detection cluster sizes:", counts.to_dict())
    total_users = feats_detect.shape[0]
    outlier_clusters = [cl for cl, cnt in counts.items() if (cnt < MIN_CLUSTER_SIZE) or (cnt / total_users < MIN_CLUSTER_PCT)]
    print(f"Identified outlier clusters (threshold size<{MIN_CLUSTER_SIZE} or pct<{MIN_CLUSTER_PCT}): {outlier_clusters}")
    outlier_user_ids = feats_detect[feats_detect["detect_cluster"].isin(outlier_clusters)]["User_ID"].astype(str).unique().tolist()
    core_user_ids = feats_detect[~feats_detect["detect_cluster"].isin(outlier_clusters)]["User_ID"].astype(str).unique().tolist()
    pd.Series(core_user_ids, name="User_ID").to_csv(os.path.join(out_det_dir, "core_user_ids.csv"), index=False)
    pd.Series(outlier_user_ids, name="User_ID").to_csv(os.path.join(out_det_dir, "outlier_user_ids.csv"), index=False)
    print(f"Core users: {len(core_user_ids)}, Outliers/niches: {len(outlier_user_ids)}")

    # 6) Filter original transactions into core_transactions and outlier_transactions (streaming)
    core_tx_path = os.path.join(out_det_dir, "core_transactions.csv")
    out_tx_path = os.path.join(out_det_dir, "outlier_transactions.csv")
    if not os.path.exists(core_tx_path) or not os.path.exists(out_tx_path):
        print("Splitting transaction file into core/outlier transactions (streaming)...")
        core_set = set(core_user_ids)
        out_set = set(outlier_user_ids)
        reader = pd.read_csv(TRANSACTIONS_CSV, dtype=str, usecols=None, chunksize=200000)
        header_written_core = False
        header_written_out = False
        for chunk in reader:
            chunk["User_ID"] = chunk["User_ID"].astype(str)
            c_core = chunk[chunk["User_ID"].isin(core_set)]
            c_out = chunk[chunk["User_ID"].isin(out_set)]
            if not c_core.empty:
                c_core.to_csv(core_tx_path, mode="a", index=False, header=not header_written_core)
                header_written_core = True
            if not c_out.empty:
                c_out.to_csv(out_tx_path, mode="a", index=False, header=not header_written_out)
                header_written_out = True
        print(f"Wrote core transactions to {core_tx_path} ({'exists' if os.path.exists(core_tx_path) else 'missing'})")
        print(f"Wrote outlier transactions to {out_tx_path} ({'exists' if os.path.exists(out_tx_path) else 'missing'})")
    else:
        print("Core/outlier transaction files already exist; skipping split.")

    # 7) Recluster core with test2 settings (complete+manhattan), stronger winsorize & PCA variance
    print("Rebuilding features for core users and reclustering core...")
    core_tx_df = read_transactions(core_tx_path, q_low=TX_WINSORIZE_LOWER, q_high=TX_WINSORIZE_UPPER)
    feats_core, _ = build_user_features(core_tx_df, top_cat=10)
    feats_core = winsorize_user(feats_core, USER_WINSORIZE_LOWER_CORE, USER_WINSORIZE_UPPER_CORE)
    preproc_core, id_col_core, numeric_core, cat_core = build_transformer(feats_core)
    X_raw_core = feats_core.drop(columns=[id_col_core])
    X_trans_core = preproc_core.fit_transform(X_raw_core)
    X_dense_core = to_dense_if_needed(X_trans_core)
    pca_core = PCA(n_components=PCA_VARIANCE_CORE, random_state=RANDOM_STATE)
    X_for_core = pca_core.fit_transform(X_dense_core)
    explained_core = pca_core.explained_variance_ratio_.sum()
    print(f"PCA core explained variance: {explained_core:.3f}")
    core_labels = fit_agglomerative(X_for_core, CORE_LINKAGE, CORE_METRIC, CORE_K)
    feats_core["cluster"] = core_labels
    out_core_dir = os.path.join(TMP_DIR, "core_recluster")
    ensure_dir(out_core_dir)
    feats_core.to_csv(os.path.join(out_core_dir, "user_clusters_core.csv"), index=False)
    print(f"Core recluster sizes: {feats_core['cluster'].value_counts().to_dict()}")

    # 8) Reattach outliers: preserve their detect_cluster groups as separate segments
    print("Reattaching outliers as separate niche clusters...")
    outlier_ids_df = pd.read_csv(os.path.join(out_det_dir, "outlier_user_ids.csv"))
    outlier_ids = outlier_ids_df["User_ID"].astype(str).tolist()
    detect_map = feats_detect.set_index("User_ID")["detect_cluster"].astype(int).to_dict()
    outlier_entries = []
    for uid in outlier_ids:
        # uid in feats_detect index might be int; handle both
        try:
            detect_cl = detect_map.get(int(uid))
        except Exception:
            detect_cl = detect_map.get(uid)
        outlier_entries.append({"User_ID": int(uid) if uid.isdigit() else uid, "detect_cluster": detect_cl})
    df_outliers = pd.DataFrame(outlier_entries)
    core_df = feats_core[["User_ID","cluster"]].copy()
    core_clusters = sorted(core_df["cluster"].unique().tolist())
    core_map = {old: new for new, old in enumerate(core_clusters)}
    core_df["cluster"] = core_df["cluster"].map(core_map)
    next_label = max(core_df["cluster"].max(), -1) + 1
    detect_unique = sorted(df_outliers["detect_cluster"].unique().tolist())
    detect_map_to_final = {d: i+next_label for i,d in enumerate(detect_unique)}
    df_outliers["cluster"] = df_outliers["detect_cluster"].map(detect_map_to_final)
    final_df = pd.concat([core_df, df_outliers[["User_ID","cluster"]]], ignore_index=True, sort=False)
    final_df["User_ID"] = final_df["User_ID"].astype(int)
    ensure_dir(OUTDIR)
    final_df.to_csv(os.path.join(OUTDIR, "final_user_clusters.csv"), index=False)
    print(f"Wrote final_user_clusters.csv with {final_df.shape[0]} users and {final_df['cluster'].nunique()} clusters.")

    # 9) Save profiles and diagnostics
    print("Computing final cluster profiles and diagnostics...")
    feats_all = feats.merge(final_df, on="User_ID", how="right")
    numeric_cols_all = feats_all.select_dtypes(include=[np.number]).columns.tolist()
    feats_all[numeric_cols_all] = feats_all[numeric_cols_all].fillna(0)
    profile = profile_clusters(feats_all, labels_col="cluster")
    profile.to_csv(os.path.join(OUTDIR, "cluster_profile_summary.csv"))
    try:
        sil_core = silhouette_score(X_for_core, core_labels, metric=CORE_METRIC)
        pd.DataFrame([{"k": CORE_K, "silhouette_core": float(sil_core)}]).to_csv(os.path.join(OUTDIR, "silhouette_core.csv"), index=False)
        print(f"Core silhouette ({CORE_LINKAGE}+{CORE_METRIC}, k={CORE_K}): {sil_core:.3f}")
    except Exception as e:
        print("Could not compute core silhouette:", e)
    try:
        sil_detect = silhouette_score(X_for_detect, labels_detect, metric=detect_metric)
        pd.DataFrame([{"k": DETECT_K, "silhouette_detect": float(sil_detect)}]).to_csv(os.path.join(OUTDIR, "silhouette_detect.csv"), index=False)
        print(f"Detect silhouette ({detect_linkage}+{detect_metric}, k={DETECT_K}): {sil_detect:.3f}")
    except Exception as e:
        print("Could not compute detect silhouette:", e)

    # Dendrograms and plots - NOTE: supply linkage first, metric second
    save_dendrogram(X_for_core, CORE_LINKAGE, CORE_METRIC, os.path.join(OUTDIR, "dendrogram_core.png"))
    save_2d_plot(X_dense_core, core_labels, os.path.join(OUTDIR, "clusters_core_2d.png"), explained_core)
    save_3d_plot(X_dense_core, core_labels, os.path.join(OUTDIR, "clusters_core_3d.html"), os.path.join(OUTDIR, "clusters_core_3d.png"))

    final_df.head(5000).to_csv(os.path.join(OUTDIR, "final_user_clusters_sample.csv"), index=False)

    print("Hybrid pipeline completed. Outputs in:", OUTDIR)
    print("Key files:")
    for f in ["final_user_clusters.csv", "cluster_profile_summary.csv", "silhouette_core.csv", "silhouette_detect.csv", "dendrogram_core.png", "clusters_core_2d.png", "clusters_core_3d.html"]:
        print(" -", os.path.join(OUTDIR, f))

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    print("Starting final clustering pipeline (hybrid).")
    run_hybrid_pipeline()
    print("Done.")