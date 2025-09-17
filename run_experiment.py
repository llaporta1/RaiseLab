import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1) Load & clean UCI Adult data
cols = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

def load_adult():
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


# 2) Intersectional groups
group_keys = ["race", "education"]
df["group_key"] = df[group_keys].astype(str).agg(" | ".join, axis=1)

# 3) Binning (for DP hist/suppression)
def bin_age(x):
    bins = [0,20,30,40,50,60,70,80,120]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)

def bin_hours(x):
    bins = [0,20,30,40,50,60,80,120]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)

def bin_capital_gain(val, med):
    return ("zero" if val == 0 else ("low" if 0 < val <= med else "high"))

df["age_bin"] = bin_age(df["age"])
df["hrs_bin"] = bin_hours(df["hours-per-week"])
cg_median_nonzero = df['capital-gain'].replace(0, np.nan).median(skipna=True)
df["cg_bin"] = df["capital-gain"].apply(lambda v: bin_capital_gain(v, cg_median_nonzero))
df["cl_bin"] = df["capital-loss"].apply(lambda v: "zero" if v == 0 else "nonzero")

# Columns used for suppression histograms (categorical)
bin_cols = [
    "age_bin","hrs_bin","cg_bin","cl_bin",
    "workclass","occupation","native-country","sex","marital-status","relationship"
]

# 4) Histogram noise + cell suppression
rng = np.random.default_rng(42)

def laplace_noise(counts, eps=1.0):
    scale = 1.0 / eps
    noise = rng.laplace(0.0, scale, size=len(counts))
    noisy = counts.astype(float) + noise
    noisy = np.clip(noisy, 0, None)  
    return noisy

def suppression_threshold(counts, pct):
    # Percentile threshold on nonzero counts 
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return np.inf
    return np.percentile(nonzero, pct)

def apply_suppression_flags(df_group, percentiles=(5,10,2), eps=1.0):
    """
    For each bin_col in this group:
      - Publish Laplace-noisy histogram (for reporting only)
      - Compute percentile threshold on TRUE counts (ignores zeros)
      - Mask rows where the bin level's count is below threshold
      - Do NOT touch true zero bins (no rows exist to mask)
    Returns:
      noisy_hist: dict[col] -> DataFrame(level,true_count,noisy_count)
      mask_df: DataFrame of booleans: True where row should be SUPPRESSED for that col
    """
    noisy_hist = {}
    masks = {}
    pct = max(percentiles) 

    for col in bin_cols:
        counts = df_group[col].value_counts(dropna=False).sort_index()
        noisy = laplace_noise(counts.values, eps=eps)
        noisy_hist[col] = pd.DataFrame({
            "level": counts.index.astype(str),
            "true_count": counts.values,
            "noisy_count": noisy
        })

        thr = suppression_threshold(counts.values, pct)
        low_levels = set(counts[counts.values < thr].index.tolist())
        masks[col] = df_group[col].isin(low_levels)

    mask_df = pd.DataFrame({col: masks[col].values for col in masks}, index=df_group.index)
    return noisy_hist, mask_df


# 5) Modeling utilities
target_col = "income"
base_cats_other = ["education","race","sex","marital-status","relationship","workclass","occupation","native-country"]
base_nums = ["age","hours-per-week","capital-gain","capital-loss","fnlwgt","education-num"]

def build_features(df_src, suppression_mask=None):
    """
    Build the SAME feature space for baseline and DP:
      - Always include dp copies for binned columns: <bin>_dp
      - If suppression_mask is None: use the original bin labels (no suppression)
      - If suppression_mask provided: replace low-count rows with 'SUPPRESSED_<col>'
      - Always drop target & helper columns from features
    """
    X = df_src.copy()

    # Drop target + helper columns to prevent leakage
    X = X.drop(columns=[c for c in ["income","group_key"] if c in X.columns], errors="ignore")

    # Create *_dp columns for BOTH baseline and DP
    X_model = X.copy()
    for col in bin_cols:
        new_col = f"{col}_dp"
        X_model[new_col] = X[col].astype(str)
        if suppression_mask is not None:
            sup_flag = suppression_mask[col]
            X_model.loc[sup_flag, new_col] = f"SUPPRESSED_{col}"

    # Categorical features = dp copies of binned columns + other categoricals
    cat_cols = [f"{c}_dp" for c in bin_cols] + base_cats_other

    # Numeric features unchanged
    num_cols = base_nums

    # Drop original bin columns (use *_dp instead) and any residual helper columns
    drop_cols = list(set(bin_cols))
    X_model = X_model.drop(columns=[c for c in drop_cols if c in X_model.columns], errors="ignore")

    # ensure target not present
    X_model = X_model.drop(columns=["income"], errors="ignore")

    assert "income" not in X_model.columns, "Target leakage: 'income' found in features."
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

    used_cats = [c for c in Xg.columns if Xg[c].dtype == 'object']
    used_nums = [c for c in Xg.columns if Xg[c].dtype != 'object']
    assert "income" not in used_cats and "income" not in used_nums, "Target leakage in features."

    pipe = make_pipeline(cat_cols=used_cats, num_cols=used_nums)

    best = {"reached": False, "depth": None, "accuracy": None, "n": len(yg)}
    last_acc = None

    for depth in range(1, max_cap + 1):
        pipe.set_params(clf__max_depth=depth)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        last_acc = acc
        if acc >= acc_threshold:
            best.update({
                "reached": True,
                "depth": pipe.named_steps["clf"].get_depth(),
                "accuracy": acc
            })
            break

    if not best["reached"]:
        best.update({
            "depth": pipe.named_steps["clf"].get_depth(),
            "accuracy": last_acc
        })
    return best

# 6) Run experiment
target_series = df[target_col].eq(">50K").astype(int)

results = []
# Majority vs minority by total group size (top 30% by size => majority)
group_counts = df.groupby("group_key").size().sort_values(ascending=False)
cut = int(np.ceil(0.30 * len(group_counts)))
majority_groups = set(group_counts.index[:cut])

for gkey, df_g in df.groupby("group_key"):
    y_g = (df_g[target_col].eq(">50K")).astype(int)

    # A) Baseline (no DP)
    X_base, _, _ = build_features(df_g, suppression_mask=None)
    base_85 = find_min_depth_for_threshold(X_base, y_g, acc_threshold=0.85)
    base_90 = find_min_depth_for_threshold(X_base, y_g, acc_threshold=0.90)

    # B) DP: histograms with noise + cell suppression (affects *_dp features)
    noisy_hist, mask_df = apply_suppression_flags(df_g, percentiles=(5,10,2), eps=1.0)
    X_dp, _, _ = build_features(df_g, suppression_mask=mask_df)

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

# 7) Aggregates + disparity deltas
def agg_side_by_side(dfres, is_majority):
    sub = dfres[dfres["majority"] == is_majority]
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
        # Depth deltas (DP - Baseline)
        "delta85_depth_avg": [(sub["dp85_depth"] - sub["baseline85_depth"]).dropna().mean()],
        "delta90_depth_avg": [(sub["dp90_depth"] - sub["baseline90_depth"]).dropna().mean()],
        # Accuracy deltas
        "delta85_acc_avg": [(sub["dp85_acc"] - sub["baseline85_acc"]).dropna().mean()],
        "delta90_acc_avg": [(sub["dp90_acc"] - sub["baseline90_acc"]).dropna().mean()],
    })

majority_agg = agg_side_by_side(res, True)
minority_agg = agg_side_by_side(res, False)

print("=== Per-group results (head) ===")
print(res.head(12).to_string(index=False))

print("\n=== Majority groups aggregate ===")
print(majority_agg.to_string(index=False))

print("\n=== Minority groups aggregate ===")
print(minority_agg.to_string(index=False))

def verdict(delta_majority, delta_minority, threshold=0.1):
    diff = delta_minority - delta_majority
    if pd.isna(diff):
        return "insufficient data"
    if diff > threshold:
        return "disparity widened (minority depth increased more)"
    if diff < -threshold:
        return "disparity narrowed (minority depth increased less)"
    return "no material change"

maj_d85 = majority_agg["delta85_depth_avg"].iloc[0]
min_d85 = minority_agg["delta85_depth_avg"].iloc[0]
maj_d90 = majority_agg["delta90_depth_avg"].iloc[0]
min_d90 = minority_agg["delta90_depth_avg"].iloc[0]

print("\n=== Disparity verdicts ===")
print(f"85% threshold: {verdict(maj_d85, min_d85)}  (majority Δdepth={maj_d85:.3f}, minority Δdepth={min_d85:.3f})")
print(f"90% threshold: {verdict(maj_d90, min_d90)}  (majority Δdepth={maj_d90:.3f}, minority Δdepth={min_d90:.3f})")
