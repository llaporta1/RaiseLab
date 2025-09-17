import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1) Load & clean
# UCI Adult Dataset: adult.data + adult.test
cols = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

def load_adult():
    base = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    # original UCI CSV endpoints:
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    df_tr = pd.read_csv(train_url, names=cols, na_values=" ?", skipinitialspace=True)
    df_te = pd.read_csv(test_url,  names=cols, na_values=" ?", skipinitialspace=True, comment='|')
    df = pd.concat([df_tr, df_te], ignore_index=True)

    # Normalize income labels: '>50K', '<=50K'
    df['income'] = df['income'].astype(str).str.replace('.', '', regex=False).str.strip()
    df = df.dropna().reset_index(drop=True)
    return df

df = load_adult()

# 2) Define intersectional groups (e.x., race x education)
group_keys = ["race", "education"]
df["group_key"] = df[group_keys].astype(str).agg(" | ".join, axis=1)

# 3) Create binned copies for DP histogram/suppression checks
def bin_age(x):
    # 10-year bins
    bins = [0,20,30,40,50,60,70,80,120]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)

def bin_hours(x):
    bins = [0,20,30,40,50,60,80,120]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)

def bin_capital(val):
    return (
        "zero" if val==0 else
        "low"  if val>0 and val<=df['capital-gain'].replace(0,np.nan).median(skipna=True) else
        "high"
    )

df["age_bin"]   = bin_age(df["age"])
df["hrs_bin"]   = bin_hours(df["hours-per-week"])
df["cg_bin"]    = df["capital-gain"].apply(bin_capital)
df["cl_bin"]    = df["capital-loss"].apply(lambda v: "zero" if v==0 else "nonzero")

# Columns used for suppression histograms
bin_cols = ["age_bin","hrs_bin","cg_bin","cl_bin","workclass","occupation","native-country","sex","marital-status","relationship"]

# 4) Histogram noise + cell suppression 
rng = np.random.default_rng(42)

def laplace_noise(counts, eps=1.0):
    scale = 1.0/eps
    noise = rng.laplace(0.0, scale, size=len(counts))
    noisy = counts.astype(float) + noise
    noisy = np.clip(noisy, 0, None)  
    return noisy

def suppression_threshold(counts, pct):
    # percentile-based threshold on nonzero counts only
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return np.inf
    return np.percentile(nonzero, pct)

def apply_suppression_flags(df_group, percentiles=(5,10,2)):
    """
    For each bin_col, compute counts per level.
    - Publish a Laplace-noisy histogram (returned for logging)
    - Suppress cells whose count < percentile threshold (use the MAX of provided percentiles for stricter suppression or do multiple passes)
    - Do NOT touch true zeros (they are not in the data, so nothing to mask)
    Returns:
      - noisy_hist: dict[col] = DataFrame(level, true_count, noisy_count)
      - mask_df: boolean DataFrame: True where that (row, col) should be suppressed
    """
    noisy_hist = {}
    masks = {}

    # choose percentile setting (example: take MAX, the most conservative)
    pct = max(percentiles)

    for col in bin_cols:
        counts = df_group[col].value_counts(dropna=False).sort_index()
        noisy = laplace_noise(counts.values, eps=1.0)
        noisy_hist[col] = pd.DataFrame({
            "level": counts.index.astype(str),
            "true_count": counts.values,
            "noisy_count": noisy
        })

        # suppression threshold (ignore zeros)
        thr = suppression_threshold(counts.values, pct)
        low_levels = set(counts[counts.values < thr].index.tolist())
        masks[col] = df_group[col].isin(low_levels)

    mask_df = pd.DataFrame({col: masks[col].values for col in masks}, index=df_group.index)
    return noisy_hist, mask_df


# 5) Modeling utilities
target_col = "income"
y = df[target_col].eq(">50K").astype(int)

base_cats = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
base_nums = ["age","hours-per-week","capital-gain","capital-loss","fnlwgt","education-num"]

# Add DP-suppressed categorical copies for columns that can be masked
# Replace the model’s categorical columns with their DP-aware versions:
def build_features(df_src, suppression_mask=None):
    X = df_src.copy()
    if suppression_mask is not None:

        X_model = X.copy()
        for col in bin_cols:
            sup_flag = suppression_mask[col]
            new_col = f"{col}_dp"
            X_model[new_col] = X[col].astype(str)
            X_model.loc[sup_flag, new_col] = f"SUPPRESSED_{col}"

        cat_cols = [f"{c}_dp" for c in bin_cols] + ["education","race","sex","marital-status","relationship","workclass","occupation","native-country"]
    else:
        X_model = X.copy()
        cat_cols = base_cats

    num_cols = base_nums

    # Drop columns so not duplicated 
    drop_cols = list(set(["group_key"] + bin_cols))
    X_model = X_model.drop(columns=[c for c in drop_cols if c in X_model.columns], errors="ignore")

    return X_model, cat_cols, num_cols

def make_pipeline(cat_cols, num_cols):
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )
    pipe = Pipeline(steps=[
        ("prep", ct),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])
    return pipe

def find_min_depth_for_threshold(Xg, yg, acc_threshold=0.85, max_cap=40, test_size=0.2):
    if len(np.unique(yg)) < 2 or len(yg) < 50:
        return {"reached": False, "depth": None, "accuracy": None, "n": len(yg)}
    X_tr, X_te, y_tr, y_te = train_test_split(Xg, yg, test_size=test_size, stratify=yg, random_state=42)
    pipe = make_pipeline(
        cat_cols=[c for c in Xg.columns if Xg[c].dtype=='object'],
        num_cols=[c for c in Xg.columns if Xg[c].dtype!='object']
    )

    best = {"reached": False, "depth": None, "accuracy": None, "n": len(yg)}
    for depth in range(1, max_cap+1):
        pipe.set_params(clf__max_depth=depth)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        if acc >= acc_threshold:
            best.update({"reached": True, "depth": pipe.named_steps["clf"].get_depth(), "accuracy": acc})
            break
    if not best["reached"]:
        best.update({"depth": pipe.named_steps["clf"].get_depth(), "accuracy": acc})
    return best


# 6) Run experiment

results = []
# Decide majority vs minority by total group size (top 30% by size = majority)
group_counts = df.groupby("group_key").size().sort_values(ascending=False)
cut = int(np.ceil(0.30 * len(group_counts)))
majority_groups = set(group_counts.index[:cut])

for gkey, df_g in df.groupby("group_key"):
    y_g = (df_g[target_col].eq(">50K")).astype(int)

    # A) Baseline (no DP)
    X_base, cat_cols_base, num_cols_base = build_features(df_g, suppression_mask=None)
    base_85 = find_min_depth_for_threshold(X_base, y_g, acc_threshold=0.85)
    base_90 = find_min_depth_for_threshold(X_base, y_g, acc_threshold=0.90)

    # B) DP: histograms with bins noise + cell suppression
    noisy_hist, mask_df = apply_suppression_flags(df_g, percentiles=(5,10,2))  # we use max percentile internally
    X_dp, cat_cols_dp, num_cols_dp = build_features(df_g, suppression_mask=mask_df)

    dp_85 = find_min_depth_for_threshold(X_dp, y_g, acc_threshold=0.85)
    dp_90 = find_min_depth_for_threshold(X_dp, y_g, acc_threshold=0.90)

    results.append({
        "group": gkey,
        "size": len(df_g),
        "majority": gkey in majority_groups,

        "baseline85_reached": base_85["reached"],
        "baseline85_depth": base_85["depth"],
        "baseline85_acc": base_85["accuracy"],

        "baseline90_reached": base_90["reached"],
        "baseline90_depth": base_90["depth"],
        "baseline90_acc": base_90["accuracy"],

        "dp85_reached": dp_85["reached"],
        "dp85_depth": dp_85["depth"],
        "dp85_acc": dp_85["accuracy"],

        "dp90_reached": dp_90["reached"],
        "dp90_depth": dp_90["depth"],
        "dp90_acc": dp_90["accuracy"],
    })

res = pd.DataFrame(results).sort_values(by="size", ascending=False)

# Aggregate majority vs minority average comparisons
def agg_side_by_side(dfres, flag):
    sub = dfres[dfres["majority"]==flag]
    return pd.DataFrame({
        "n_groups": [len(sub)],
        "avg_size": [sub["size"].mean()],
        "baseline85_depth_avg": [sub["baseline85_depth"].dropna().mean()],
        "dp85_depth_avg": [sub["dp85_depth"].dropna().mean()],
        "baseline90_depth_avg": [sub["baseline90_depth"].dropna().mean()],
        "dp90_depth_avg": [sub["dp90_depth"].dropna().mean()],
        "baseline85_acc_avg": [sub["baseline85_acc"].dropna().mean()],
        "dp85_acc_avg": [sub["dp85_acc"].dropna().mean()],
        "baseline90_acc_avg": [sub["baseline90_acc"].dropna().mean()],
        "dp90_acc_avg": [sub["dp90_acc"].dropna().mean()],
    })

majority_agg = agg_side_by_side(res, True)
minority_agg = agg_side_by_side(res, False)

print("=== Per-group results (head) ===")
print(res.head(12).to_string(index=False))

print("\n=== Majority groups aggregate ===")
print(majority_agg.to_string(index=False))

print("\n=== Minority groups aggregate ===")
print(minority_agg.to_string(index=False))
