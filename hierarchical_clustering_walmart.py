#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hierarchical clustering of Walmart users.

Melhorias fixas em TODAS as variantes:
- Limpeza:
  * drop_duplicates nas linhas
  * Winsorização de Purchase por quantis (nível linha) --winsorize-lower/upper
  * Winsorização das features numéricas agregadas (nível usuário) --user-winsorize-lower/upper
- Features (iguais em todas as variantes):
  * Shares por contagem (cat_share__*) e por receita (cat_rev_share__*) por categoria
  * Entropia de categorias (cat_entropy) e número de categorias únicas (n_unique_categories)
  * log1p para avg_purchase, median_purchase e std_purchase
  * Redução de colinearidade: não usamos total_purchase_log1p e removemos colunas brutas redundantes; mantemos n_transactions
- Pré-processamento:
  * RobustScaler
  * PCA por alvo de variância explicada (--pca-variance-target, padrão 0.80)

Variantes (mutuamente exclusivas):
--original: ward + euclidean; k = argmax silhouette em [--k-min, --k-max]; outdir default: outputs
--test1   : average + manhattan; k em {4,5} (escolhe melhor silhouette); outdir default: outputs_test1
--test2   : complete + manhattan; k=4; outdir default: outputs_test2
--test3   : ward + euclidean; k=3; outdir default: outputs_test3

Arquivos gerados em cada outdir:
- user_clusters.csv
- cluster_profile_summary.csv
- silhouette_scores.csv
- dendrogram.png
- clusters_2d.png
- clusters_3d.html (ou clusters_3d.png se Plotly indisponível)
"""

import argparse
import os
import warnings
from typing import Tuple

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

RANDOM_STATE = 42

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical clustering on Walmart users with multiple variants.")
    p.add_argument("--csv", required=True, help="Path to walmart.csv")
    p.add_argument("--outdir", default="outputs", help="Output directory (ajustada pela variante se ficar como 'outputs').")
    p.add_argument("--use-pca", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=True, help="Executa PCA antes do clustering.")
    p.add_argument("--pca-variance-target", type=float, default=0.80, help="Alvo de variância explicada para PCA (0<val<=1).")
    p.add_argument("--top-cat", type=int, default=10, help="Top N categorias por frequência para features de categoria.")
    p.add_argument("--truncate-dendro", type=int, default=50, help="Dendrogram: mostrar últimas p junções.")

    # Limpeza
    p.add_argument("--winsorize-lower", type=float, default=0.01, help="Quantil inferior para winsorizar Purchase (nível linha).")
    p.add_argument("--winsorize-upper", type=float, default=0.995, help="Quantil superior para winsorizar Purchase (nível linha).")
    p.add_argument("--user-winsorize-lower", type=float, default=0.005, help="Quantil inferior para winsorizar numéricos (nível usuário).")
    p.add_argument("--user-winsorize-upper", type=float, default=0.995, help="Quantil superior para winsorizar numéricos (nível usuário).")

    # Busca k (apenas --original usa estes)
    p.add_argument("--k-min", type=int, default=2, help="k mínimo para busca por silhouette (apenas --original).")
    p.add_argument("--k-max", type=int, default=10, help="k máximo para busca por silhouette (apenas --original).")

    # Variantes (mutuamente exclusivas)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--original", action="store_true", help="Ward+Euclidean, busca k em [k-min, k-max].")
    g.add_argument("--test1", action="store_true", help="Average+Manhattan, k em {4,5}.")
    g.add_argument("--test2", action="store_true", help="Complete+Manhattan, k=4.")
    g.add_argument("--test3", action="store_true", help="Ward+Euclidean, k=3.")

    return p.parse_args()

# -------------------------
# Utils
# -------------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def default_outdir_for_variant(args) -> str:
    # Se o usuário não alterou e manteve 'outputs', escolhemos por variante
    if args.outdir == "outputs":
        if args.test1: return "outputs_test1"
        if args.test2: return "outputs_test2"
        if args.test3: return "outputs_test3"
        return "outputs"  # original
    return args.outdir

def winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower, upper)

def to_dense_if_needed(X):
    return X.toarray() if hasattr(X, "toarray") else X

def compute_linkage_matrix(X_dense: np.ndarray, metric: str, linkage_method: str):
    """
    Para SciPy, 'manhattan' deve ser 'cityblock'.
    """
    metric_map_for_scipy = {
        "manhattan": "cityblock",
        "l1": "cityblock",
        "l2": "euclidean",
    }
    scipy_metric = metric_map_for_scipy.get(metric, metric)

    if linkage_method == "ward":
        if metric != "euclidean":
            raise ValueError("Ward linkage requires Euclidean metric.")
        Z = linkage(X_dense, method="ward")
    else:
        try:
            dm = pdist(X_dense, metric=scipy_metric)
        except ValueError as e:
            if "Unknown Distance Metric" in str(e) and metric in ("manhattan","l1"):
                dm = pdist(X_dense, metric="cityblock")
            else:
                raise
        Z = linkage(dm, method=linkage_method)
    return Z

def fit_agglomerative(X: np.ndarray, linkage_method: str, metric: str, k: int) -> np.ndarray:
    if linkage_method == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, metric=metric)
    return model.fit_predict(X)

# -------------------------
# Data prep
# -------------------------
def read_and_preprocess_rows(csv_path: str, q_low: float, q_high: float) -> pd.DataFrame:
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
    usecols = [
        "User_ID","Product_ID","Gender","Age","Occupation","City_Category",
        "Stay_In_Current_City_Years","Marital_Status","Product_Category","Purchase"
    ]
    df = pd.read_csv(csv_path, dtype=dtype_map, usecols=usecols)

    # Limpeza linha
    df = df.drop_duplicates(ignore_index=True)

    # Normalizar Stay
    df["Stay_In_Current_City_Years"] = (
        df["Stay_In_Current_City_Years"].str.replace("+", "", regex=False).astype("Int8")
    )

    # Product_Category buckets
    pc = pd.to_numeric(df["Product_Category"], errors="coerce").astype("Int16")
    df["Product_Category"] = pc
    df["Product_Category_str"] = df["Product_Category"].astype("Int16").astype("string")
    df["Product_Category_str"] = df["Product_Category_str"].fillna("unknown").replace({"<NA>": "unknown"})
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

    # Winsorizar Purchase (nível linha)
    if 0 < q_low < q_high < 1:
        low = df["Purchase"].quantile(q_low)
        high = df["Purchase"].quantile(q_high)
        df["Purchase"] = df["Purchase"].clip(lower=low, upper=high)

    return df

def series_mode(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

def shannon_entropy(counts: np.ndarray):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def build_user_features(df: pd.DataFrame, top_cat: int = 10) -> pd.DataFrame:
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

    top_cats = df["Product_Category_str"].value_counts(dropna=False).head(top_cat).index.tolist()
    cat_col = df["Product_Category_str"].where(df["Product_Category_str"].isin(top_cats), other="cat_other")
    df_tmp = df.assign(Product_Category_top=cat_col)

    # Shares por contagem
    cat_pivot = (
        df_tmp.pivot_table(
            index="User_ID",
            columns="Product_Category_top",
            values="Product_ID",
            aggfunc="count",
            fill_value=0,
            observed=True,
        ).astype("int32")
    )
    cat_share = cat_pivot.div(cat_pivot.sum(axis=1).replace(0, 1), axis=0)
    cat_share.columns = [f"cat_share__{c}" for c in cat_share.columns]
    cat_share = cat_share.reset_index()

    # Shares por receita
    rev_pivot = (
        df_tmp.pivot_table(
            index="User_ID",
            columns="Product_Category_top",
            values="Purchase",
            aggfunc="sum",
            fill_value=0.0,
            observed=True,
        ).astype("float64")
    )
    rev_share = rev_pivot.div(rev_pivot.sum(axis=1).replace(0, 1), axis=0)
    rev_share.columns = [f"cat_rev_share__{c}" for c in rev_share.columns]
    rev_share = rev_share.reset_index()

    # Entropia e n categorias
    cat_entropy = pd.DataFrame({
        "User_ID": cat_pivot.index,
        "cat_entropy": [shannon_entropy(row.values) for _, row in cat_pivot.iterrows()],
        "n_unique_categories": (cat_pivot > 0).sum(axis=1).astype("int16"),
    }).reset_index(drop=True)

    features = (
        agg_basic
        .merge(modes, on="User_ID", how="left")
        .merge(cat_share, on="User_ID", how="left")
        .merge(rev_share, on="User_ID", how="left")
        .merge(cat_entropy, on="User_ID", how="left")
    )

    # log1p das métricas de gasto (mantemos n_transactions bruto)
    for col in ["avg_purchase","median_purchase","std_purchase"]:
        features[f"{col}_log1p"] = np.log1p(features[col].astype("float64"))

    # NA handling
    for col in ["Age_mode","Gender_mode","City_Category_mode"]:
        if col in features.columns:
            if isinstance(features[col].dtype, CategoricalDtype):
                features[col] = features[col].cat.add_categories(["Unknown"]).fillna("Unknown")
            else:
                features[col] = features[col].astype("string").fillna("Unknown")
    for col, dtype in [("Stay_mode","int16"), ("Marital_mode","int16"), ("Occupation_mode","int16")]:
        if col in features.columns:
            features[col] = features[col].fillna(-1).astype(dtype)

    # Numéricos: preencher NA
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    features[num_cols] = features[num_cols].fillna(0)

    return features

def winsorize_user_features(feats: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    if not (0 < q_low < q_high < 1):
        return feats
    feats = feats.copy()
    num_cols = [c for c in feats.columns if c != "User_ID" and pd.api.types.is_numeric_dtype(feats[c])]
    for col in num_cols:
        feats[col] = winsorize_series(feats[col].astype("float64"), q_low, q_high)
    return feats

# -------------------------
# Transformer
# -------------------------
def make_one_hot_encoder():
    # scikit-learn >= 1.2: sparse_output; < 1.2: sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def build_transformer(features: pd.DataFrame):
    id_col = "User_ID"

    # Numéricos base
    numeric_cols = ["n_transactions","unique_products","cat_entropy","n_unique_categories"]

    # Shares por contagem e por receita
    numeric_cols += [c for c in features.columns if c.startswith("cat_share__")]
    numeric_cols += [c for c in features.columns if c.startswith("cat_rev_share__")]

    # log1p das métricas (sem as versões brutas; total_purchase_log1p não é usado)
    for base in ["avg_purchase","median_purchase","std_purchase"]:
        col = f"{base}_log1p"
        if col in features.columns:
            numeric_cols.append(col)

    cat_cols = ["Age_mode","Gender_mode","City_Category_mode","Stay_mode","Marital_mode","Occupation_mode"]

    num_transformer = RobustScaler(with_centering=True, with_scaling=True, unit_variance=False, quantile_range=(25.0, 75.0))
    cat_transformer = make_one_hot_encoder()

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preproc, id_col, numeric_cols, cat_cols

# -------------------------
# Variant logic
# -------------------------
def choose_k_by_silhouette(X: np.ndarray, linkage_method: str, metric: str, k_min: int, k_max: int, outdir: str) -> int:
    scores = []
    for k in range(k_min, k_max + 1):
        try:
            labels = fit_agglomerative(X, linkage_method, metric, k)
            if len(np.unique(labels)) < 2:
                continue
            sil = silhouette_score(X, labels, metric="euclidean" if linkage_method=="ward" else metric)
            scores.append((k, sil))
        except Exception as e:
            warnings.warn(f"Silhouette failed for k={k}: {e}")
    if scores:
        df = pd.DataFrame(scores, columns=["k","silhouette"]).sort_values("k")
        df.to_csv(os.path.join(outdir, "silhouette_scores.csv"), index=False)
        best_k = int(df.sort_values("silhouette", ascending=False).iloc[0]["k"])
        return best_k
    return max(2, k_min)

def run_variant(X_for_clust: np.ndarray, variant: str, outdir: str, k_min: int, k_max: int) -> Tuple[np.ndarray, str, str, int]:
    """
    Retorna: labels, linkage, metric, k
    """
    if variant == "original":
        linkage_method, metric = "ward", "euclidean"
        k = choose_k_by_silhouette(X_for_clust, linkage_method, metric, k_min, k_max, outdir)
    elif variant == "test1":
        linkage_method, metric = "average", "manhattan"
        candidates = [4, 5]
        scores = []
        for k in candidates:
            lbl = fit_agglomerative(X_for_clust, linkage_method, metric, k)
            sil = silhouette_score(X_for_clust, lbl, metric="manhattan")
            scores.append((k, sil))
        df = pd.DataFrame(scores, columns=["k","silhouette"]).sort_values("k")
        df.to_csv(os.path.join(outdir, "silhouette_scores.csv"), index=False)
        k = int(df.sort_values("silhouette", ascending=False).iloc[0]["k"])
    elif variant == "test2":
        linkage_method, metric, k = "complete", "manhattan", 4
        lbl = fit_agglomerative(X_for_clust, linkage_method, metric, k)
        pd.DataFrame([{"k":k, "silhouette": silhouette_score(X_for_clust, lbl, metric="manhattan")}]).to_csv(
            os.path.join(outdir, "silhouette_scores.csv"), index=False
        )
    elif variant == "test3":
        linkage_method, metric, k = "ward", "euclidean", 3
        lbl = fit_agglomerative(X_for_clust, linkage_method, metric, k)
        pd.DataFrame([{"k":k, "silhouette": silhouette_score(X_for_clust, lbl, metric="euclidean")}]).to_csv(
            os.path.join(outdir, "silhouette_scores.csv"), index=False
        )
    else:
        raise ValueError("Variante desconhecida.")

    if variant in ("original","test1"):
        lbl = fit_agglomerative(X_for_clust, linkage_method, metric, k)

    return lbl, linkage_method, metric, k

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # Determinar variante
    variant = "original"
    if args.test1: variant = "test1"
    elif args.test2: variant = "test2"
    elif args.test3: variant = "test3"

    # Outdir default por variante (se usuário não customizou)
    outdir = default_outdir_for_variant(args)
    ensure_outdir(outdir)

    # Carregar e limpar linhas
    print("Reading CSV and preprocessing rows (dedupe + winsorize Purchase)...")
    df = read_and_preprocess_rows(args.csv, q_low=args.winsorize_lower, q_high=args.winsorize_upper)

    # Features por usuário
    print("Building user-level features (shares, revenue shares, entropy, log1p)...")
    feats = build_user_features(df, top_cat=args.top_cat)

    # Winsorizar features numéricas (nível usuário)
    print("Winsorizing user-level numeric features...")
    feats = winsorize_user_features(feats, q_low=args.user_winsorize_lower, q_high=args.user_winsorize_upper)

    print(f"Users: {feats.shape[0]}, Feature columns: {feats.shape[1]-1}")

    # Transformer
    print("Building preprocessing pipeline (RobustScaler + OneHot)...")
    preproc, id_col, numeric_cols, cat_cols = build_transformer(feats)

    X_raw = feats.drop(columns=[id_col])
    user_ids = feats[id_col].values

    print("Fitting transformer...")
    X_trans = preproc.fit_transform(X_raw)
    X_dense = to_dense_if_needed(X_trans)

    # PCA
    X_for_clust = X_dense
    if args.use_pca:
        n_comp = args.pca_variance_target
        print(f"Applying PCA for clustering (variance target={n_comp:.2f})...")
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_for_clust = pca.fit_transform(X_dense)
        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA (clustering) explained variance ratio sum: {explained:.3f}")
    else:
        pca = None

    # Executar variante
    print(f"Running variant: {variant}")
    labels, best_linkage, best_metric, best_k = run_variant(
        X_for_clust, variant, outdir, k_min=args.k_min, k_max=args.k_max
    )

    # Dendrograma
    print(f"Computing linkage matrix for dendrogram (linkage={best_linkage}, metric={best_metric})...")
    Z = compute_linkage_matrix(X_for_clust, metric=best_metric, linkage_method=best_linkage)
    plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=args.truncate_dendro,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
        above_threshold_color="gray",
    )
    plt.title(f"Dendrogram (linkage={best_linkage}, metric={best_metric}) - last {args.truncate_dendro} merges")
    plt.xlabel("Merged clusters (truncated)")
    plt.ylabel("Distance")
    dendro_path = os.path.join(outdir, "dendrogram.png")
    plt.tight_layout()
    plt.savefig(dendro_path, dpi=150)
    plt.close()
    print(f"Dendrogram saved to {dendro_path}")

    # Salvar clusters
    user_cluster_df = feats.copy()
    user_cluster_df["cluster"] = labels.astype(int)
    out_csv = os.path.join(outdir, "user_clusters.csv")
    user_cluster_df.to_csv(out_csv, index=False)
    print(f"User clusters saved to {out_csv}")

    # PCA 2D
    print("Creating 2D PCA visualization...")
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca2.fit_transform(X_dense)
    var2 = pca2.explained_variance_ratio_.sum()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels, palette="tab10", s=16, linewidth=0, alpha=0.8)
    plt.title(f"Clusters (PCA 2D) – explained var={var2:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_2d = os.path.join(outdir, "clusters_2d.png")
    plt.savefig(out_2d, dpi=150)
    plt.close()
    print(f"2D cluster plot saved to {out_2d}")

    # 3D (opcional, plotly)
    print("Creating 3D PCA visualization...")
    out_3d_html = os.path.join(outdir, "clusters_3d.html")
    out_3d_png = os.path.join(outdir, "clusters_3d.png")
    try:
        import plotly.express as px
        pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
        X3 = pca3.fit_transform(X_dense)
        df_plot = pd.DataFrame({"PC1":X3[:,0], "PC2":X3[:,1], "PC3":X3[:,2], "cluster":labels})
        fig = px.scatter_3d(df_plot, x="PC1", y="PC2", z="PC3", color="cluster", opacity=0.8)
        fig.update_layout(title="Clusters (PCA 3D)")
        fig.write_html(out_3d_html)
        print(f"3D interactive plot saved to {out_3d_html}")
    except Exception as e:
        warnings.warn(f"Plotly not available or failed ({e}); falling back to Matplotlib.")
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
        X3 = pca3.fit_transform(X_dense)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=labels, cmap="tab10", s=10, alpha=0.85)
        ax.set_title("Clusters (PCA 3D)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.savefig(out_3d_png, dpi=150)
        plt.close()
        print(f"3D cluster plot saved to {out_3d_png}")

    # Resumo de perfis
    print("Summarizing cluster profiles...")
    numeric_cols = user_cluster_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("User_ID","cluster")]
    summary = user_cluster_df.groupby("cluster")[numeric_cols].agg(["count","mean","median"])
    summary.columns = [f"{a}__{b}" for a,b in summary.columns]
    out_profile = os.path.join(outdir, "cluster_profile_summary.csv")
    summary.to_csv(out_profile)
    print(f"Cluster profile summary saved to {out_profile}")

    print("Done.")

if __name__ == "__main__":
    main()