#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
association_rules_pipeline_colab.py — perfil PRODUÇÃO

Camadas:
- Categorias (geral) — pair mode (1→1), thresholds mais fortes.
- Categorias (por cluster) — pair mode com suporte adaptativo por cluster (mín. 40 usuários).
- Produtos (geral) — pair mode com top_n=40, thresholds mais fortes (lift/conf maiores).

Se quiser cumprir o requisito de “usar FP-Growth/Apriori”, ative FP-Growth só para categorias (flag USE_FPGROWTH_CATEGORIES=True);
para produtos mantenha pares por performance/operabilidade.

Rodar:
  !pip install mlxtend seaborn networkx
  from association_rules_pipeline_colab import main
  main()
"""

import os
import math
import json
import time
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------- Flags Principais -------------------
INPUT_CSV = "walmart.csv"
CLUSTERS_CSV = os.path.join("outputs_final", "final_user_clusters.csv")
OUTDIR = "outputs_assoc"
os.makedirs(OUTDIR, exist_ok=True)

# Perfis (produção)
PAIR_MODE = True              # Mantemos pares 1→1 como padrão (rápido e prático)
USE_FPGROWTH_CATEGORIES = False  # Se quiser demonstrar algoritmo, mude para True (só categorias, max_len=3)
RUN_PRODUCT_LEVEL = True
RUN_BY_CLUSTER = True

SAVE_TOP_N = 200

# Parâmetros (categoria — geral)
CATEGORY_MIN_SUPPORT = 0.04   # 4% dos usuários
CATEGORY_MIN_CONF = 0.35
CATEGORY_MIN_LIFT = 1.25
CATEGORY_MAX_LEN = 2          # pares

# Parâmetros (categoria — por cluster)
CLUSTER_MIN_CONF = 0.35
CLUSTER_MIN_LIFT = 1.25
CLUSTER_MAX_LEN = 2
CLUSTER_MIN_USERS_ABS = 40    # suporte mínimo absoluto por cluster
CLUSTER_MIN_SUPPORT_FRAC = 0.05  # ou 5% no cluster (usamos o maior entre abs e frac)

# Parâmetros (produto — geral; para cluster, reaplicamos com mesmo conjunto)
PRODUCT_MIN_SUPPORT = 0.02
PRODUCT_MIN_CONF = 0.35
PRODUCT_MIN_LIFT = 1.30
PRODUCT_MAX_LEN = 2
PRODUCT_TOP_N_ITEMS = 40

LIMITE_ITEMSETS = 20000
SOFT_TIMEOUT_SECONDS = 90

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

def read_transactions(path=INPUT_CSV):
    print(f"[{ts()}] Lendo {path} ...")
    dtype_map = {
        "User_ID": "int32",
        "Product_ID": "string",
        "Product_Category": "float32",
        "Purchase": "float32",
    }
    usecols = ["User_ID","Product_ID","Product_Category","Purchase"]
    df = pd.read_csv(path, dtype=dtype_map, usecols=usecols)
    df = df.drop_duplicates(ignore_index=True)

    pc = pd.to_numeric(df["Product_Category"], errors="coerce").astype("Int16")
    df["Product_Category"] = pc
    df["Product_Category_str"] = df["Product_Category"].astype("Int16").astype("string")
    df["Product_Category_str"] = df["Product_Category_str"].fillna("unknown").replace({"<NA>":"unknown"})
    df["Product_Category_str"] = "cat_" + df["Product_Category_str"]
    return df

def build_basket(df, item_col, min_support_frac, top_n=None, min_users_abs=None, cluster_users=None):
    sub = df[["User_ID", item_col]].dropna().drop_duplicates()
    n_users = sub["User_ID"].nunique()
    item_user_counts = sub.groupby(item_col)["User_ID"].nunique().sort_values(ascending=False)

    # Suporte adaptativo: usa o maior entre fração e absoluto (quando cluster)
    min_users_frac = math.ceil((min_support_frac or 0) * (cluster_users or n_users))
    threshold = max(1, min_users_frac, (min_users_abs or 0))
    keep = item_user_counts[item_user_counts >= threshold]
    if top_n and keep.shape[0] > top_n:
        keep = keep.head(top_n)

    kept = set(keep.index)
    sub = sub[sub[item_col].isin(kept)]
    basket = (
        sub.assign(val=1)
           .pivot_table(index="User_ID", columns=item_col, values="val", aggfunc="max", fill_value=0)
           .astype("int8")
    )
    keep_df = keep.to_frame(name="user_count")
    return basket, keep_df, n_users

def dedupe_directions(df_rules):
    # Remove duplicatas A→B e B→A mantendo a direção com maior confiança
    if df_rules.empty: 
        return df_rules
    key = df_rules.apply(lambda r: tuple(sorted([r["antecedent"], r["consequent"]])), axis=1)
    df_rules = df_rules.assign(_pair_key=key)
    df_rules = df_rules.sort_values(["_pair_key","confidence","lift","support"], ascending=[True, False, False, False])
    dedup = df_rules.drop_duplicates(subset="_pair_key", keep="first").drop(columns=["_pair_key"])
    return dedup.reset_index(drop=True)

def run_pair_mode(basket, min_conf, min_lift, outdir, level_key):
    print(f"[{ts()}] Pair mode — {level_key} ...")
    users = basket.shape[0]
    names = basket.columns.to_list()
    mat = basket.values.astype(bool)
    item_count = mat.sum(axis=0)  # abs
    item_support = item_count / users

    rows = []
    for i, A in enumerate(names):
        col_i = mat[:, i]
        if item_count[i] == 0: 
            continue
        for j, B in enumerate(names):
            if i == j: 
                continue
            col_j = mat[:, j]
            co = np.sum(col_i & col_j)
            if co == 0:
                continue
            sup_pair = co / users
            conf = co / item_count[i]
            lift = conf / item_support[j] if item_support[j] > 0 else 0
            if conf >= min_conf and lift >= min_lift:
                rows.append((A,B,sup_pair,conf,lift))

    cols = ["antecedent","consequent","support","confidence","lift"]
    df_rules = pd.DataFrame(rows, columns=cols).sort_values(["lift","confidence","support"], ascending=[False,False,False]).reset_index(drop=True)
    # Dedupe direções simétricas
    df_rules = dedupe_directions(df_rules)

    df_rules.to_csv(os.path.join(outdir, f"{level_key}_pair_rules_all.csv"), index=False)
    df_rules.head(SAVE_TOP_N).to_csv(os.path.join(outdir, f"{level_key}_pair_rules_top{SAVE_TOP_N}.csv"), index=False)

    # Scatter
    if not df_rules.empty:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        plt.scatter(df_rules["support"], df_rules["confidence"], c=df_rules["lift"], cmap="viridis", s=22, alpha=0.85)
        cbar = plt.colorbar(); cbar.set_label("lift")
        plt.xlabel("support"); plt.ylabel("confidence"); plt.title(f"Regras 1→1 — {level_key}")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{level_key}_pair_rules_scatter.png"), dpi=140); plt.close()
    print(f"[{ts()}] Regras (dedup) geradas={df_rules.shape[0]} — {level_key}")
    return df_rules.shape[0]

def run_fp_growth_categories(basket, min_support, min_conf, min_lift, outdir, level_key):
    # Opcional: cumprir requisito Apriori/FP-Growth
    from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
    start = time.time()
    try:
        fi = fpgrowth(basket.astype(bool), min_support=min_support, use_colnames=True, max_len=3)
    except Exception as e:
        print(f"[WARN] FP-Growth falhou ({e}); tentando Apriori.")
        fi = apriori(basket.astype(bool), min_support=min_support, use_colnames=True, max_len=3, low_memory=True)
    dur = time.time() - start
    print(f"[{ts()}] Itemsets categorias: {fi.shape[0]} em {dur:.1f}s")
    if fi.empty:
        return 0
    rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return 0
    rules = rules[rules["lift"] >= min_lift].copy().reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules.sort_values(["lift","confidence","support"], ascending=[False,False,False])\
         .to_csv(os.path.join(outdir, f"{level_key}_association_rules_all.csv"), index=False)
    print(f"[{ts()}] Regras FP/Apriori (categorias)={rules.shape[0]}")
    return rules.shape[0]

def process_level(df, level_key, params, cluster_context=None):
    outdir = os.path.join(OUTDIR, level_key)
    os.makedirs(outdir, exist_ok=True)

    basket, item_stats, n_users = build_basket(
        df,
        item_col=params["col_name"],
        min_support_frac=params["min_support"],
        top_n=params.get("top_n"),
        min_users_abs=params.get("min_users_abs"),
        cluster_users=cluster_context["n_users"] if cluster_context else None
    )
    item_stats.to_csv(os.path.join(outdir, f"{level_key}_item_support.csv"))
    print(f"[INFO] {level_key}: users={n_users}, items={basket.shape[1]}")

    if PAIR_MODE:
        n_rules = run_pair_mode(
            basket,
            min_conf=params["min_conf"],
            min_lift=params["min_lift"],
            outdir=outdir,
            level_key=level_key
        )
    else:
        if params.get("fp_only_categories"):
            n_rules = run_fp_growth_categories(
                basket,
                min_support=params["min_support"],
                min_conf=params["min_conf"],
                min_lift=params["min_lift"],
                outdir=outdir,
                level_key=level_key
            )
        else:
            # Poderíamos adicionar Apriori/FP-Growth genérico aqui
            n_rules = run_pair_mode(basket, params["min_conf"], params["min_lift"], outdir, level_key)

    return {
        "level": level_key,
        "n_users": n_users,
        "n_items": basket.shape[1],
        "n_rules": n_rules,
        "mode": "PAIR" if PAIR_MODE else "FP/Apriori"
    }

def write_report(summaries, path):
    lines = []
    lines.append(f"# Relatório Regras Associativas (perfil PRODUÇÃO)")
    lines.append(f"Gerado em: {ts()}")
    lines.append("")
    lines.append("## Sumário")
    for s in summaries:
        lines.append(f"- {s['level']}: users={s['n_users']} itens={s['n_items']} regras={s['n_rules']} modo={s['mode']}")
    lines.append("")
    lines.append("## Parâmetros principais")
    lines.append(f"- Categoria (geral): support>={CATEGORY_MIN_SUPPORT}, conf>={CATEGORY_MIN_CONF}, lift>={CATEGORY_MIN_LIFT}, max_len={CATEGORY_MAX_LEN}")
    lines.append(f"- Categoria (cluster): support>=max({CLUSTER_MIN_SUPPORT_FRAC:.0%}, {CLUSTER_MIN_USERS_ABS} users), conf>={CLUSTER_MIN_CONF}, lift>={CLUSTER_MIN_LIFT}")
    if RUN_PRODUCT_LEVEL:
        lines.append(f"- Produto: top_n={PRODUCT_TOP_N_ITEMS}, support>={PRODUCT_MIN_SUPPORT}, conf>={PRODUCT_MIN_CONF}, lift>={PRODUCT_MIN_LIFT}, max_len={PRODUCT_MAX_LEN}")
    lines.append("")
    lines.append("## Observações")
    lines.append("- A deduplicação mantém a direção com maior confiança para cada par {A,B}.")
    lines.append("- Se vierem regras demais (>120), eleve lift/conf em +0.05; se vierem poucas (<30), reduza em −0.05 (mín. conf 0.25, lift 1.15).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Relatório salvo em {path}")

def main():
    df = read_transactions()
    summaries = []

    # Categorias (geral)
    summaries.append(process_level(df, "category", {
        "col_name": "Product_Category_str",
        "min_support": CATEGORY_MIN_SUPPORT,
        "min_conf": CATEGORY_MIN_CONF,
        "min_lift": CATEGORY_MIN_LIFT,
        "max_len": CATEGORY_MAX_LEN,
        "fp_only_categories": USE_FPGROWTH_CATEGORIES
    }))

    # Produtos (geral)
    if RUN_PRODUCT_LEVEL:
        summaries.append(process_level(df, "product", {
            "col_name": "Product_ID",
            "min_support": PRODUCT_MIN_SUPPORT,
            "min_conf": PRODUCT_MIN_CONF,
            "min_lift": PRODUCT_MIN_LIFT,
            "max_len": PRODUCT_MAX_LEN,
            "top_n": PRODUCT_TOP_N_ITEMS
        }))
    else:
        print("[INFO] RUN_PRODUCT_LEVEL=False — nível produto pulado.")

    # Por cluster (categorias)
    if RUN_BY_CLUSTER and os.path.exists(CLUSTERS_CSV):
        clu = pd.read_csv(CLUSTERS_CSV, dtype={"User_ID":"int32"})
        merged = df.merge(clu, on="User_ID", how="inner")
        for cl_id, sub in merged.groupby("cluster"):
            n_users_c = sub["User_ID"].nunique()
            summaries.append(process_level(sub, f"cluster_{cl_id}_category", {
                "col_name": "Product_Category_str",
                "min_support": CLUSTER_MIN_SUPPORT_FRAC,
                "min_conf": CLUSTER_MIN_CONF,
                "min_lift": CLUSTER_MIN_LIFT,
                "max_len": CLUSTER_MAX_LEN,
                "min_users_abs": CLUSTER_MIN_USERS_ABS
            }, cluster_context={"n_users": n_users_c}))
    elif RUN_BY_CLUSTER:
        print("[WARN] outputs_final/final_user_clusters.csv não encontrado — por cluster pulado.")

    write_report(summaries, os.path.join(OUTDIR, "association_rules_report.md"))

if __name__ == "__main__":
    main()
