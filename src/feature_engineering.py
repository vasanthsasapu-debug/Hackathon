"""
=============================================================================
feature_engineering.py -- Feature Creation for MMIX Modeling
=============================================================================
Every function returns:
    {"data": DataFrame, "log": [actions], "summary": str, "decisions": [items]}

This structure enables agent integration -- LLM reads summaries,
orchestrator checks decisions for human-in-the-loop items.

Modules:
  1. Log Transformations
  2. Sale Event Features
  3. Channel Grouping
  4. Discount Features
  5. Lagged Variables
  6. NPS Preparation
  7. Feature Matrix Assembly
  8. Multicollinearity Check (VIF)
  9. Visualization
  10. Master Pipeline
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import (
    MEDIA_CHANNELS, CHANNEL_GROUPS, MODEL_SETTINGS,
    get_paths, get_channel_cols, find_col, logger
)

plt.rcParams["figure.figsize"] = (14, 6)
sns.set_style("whitegrid")


# =============================================================================
# 1. LOG TRANSFORMATIONS
# =============================================================================

def apply_log_transforms(df, target_col="total_gmv"):
    """
    Apply log(x+1) to channel spend columns and target.
    Enables elasticity interpretation in log-log regression.
    """
    log_actions = []
    df = df.copy()

    # Target
    if target_col in df.columns:
        df[f"log_{target_col}"] = np.log1p(df[target_col])
        log_actions.append(f"log_{target_col}")

    # Individual channels
    channels = get_channel_cols(df)
    for ch in channels:
        df[ch] = pd.to_numeric(df[ch], errors="coerce").fillna(0)
        df[f"log_{ch}"] = np.log1p(df[ch])
        log_actions.append(f"log_{ch}")

    # Total investment
    if "Total.Investment" in df.columns:
        df["log_Total_Investment"] = np.log1p(
            pd.to_numeric(df["Total.Investment"], errors="coerce").fillna(0)
        )
        log_actions.append("log_Total_Investment")

    # Discount
    if "total_Discount" in df.columns:
        df["log_total_Discount"] = np.log1p(df["total_Discount"])
        log_actions.append("log_total_Discount")

    summary = (
        f"Applied log(x+1) to {len(log_actions)} variables. "
        f"Handles zero-spend (log(0+1)=0) and enables elasticity interpretation."
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 2. SALE EVENT FEATURES
# =============================================================================

def create_sale_features(df, special_sales_df):
    """
    Create sale_flag, sale_days, sale_intensity.
    If these already exist (from weekly aggregation), reuse them.
    """
    log_actions = []
    df = df.copy()

    # Check if already created by data_aggregation
    existing = [c for c in ["sale_flag", "sale_days", "sale_intensity"] if c in df.columns]
    if existing:
        for col in existing:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if col in ["sale_flag", "sale_days", "sale_intensity"]:
                df[col] = df[col].astype(int)
        n_sale = df["sale_flag"].sum() if "sale_flag" in df.columns else 0
        total_days = df["sale_days"].sum() if "sale_days" in df.columns else 0
        log_actions.append(f"Reused existing sale features: {existing}")
        summary = (
            f"Sale features pre-existing. {n_sale} of {len(df)} periods had "
            f"promotions with {total_days} total sale days."
        )
        return {"data": df, "log": log_actions, "summary": summary, "decisions": []}

    # Create from scratch (monthly path)
    if special_sales_df is None or "Date" not in df.columns:
        df["sale_flag"] = 0
        df["sale_days"] = 0
        df["sale_intensity"] = 0
        return {"data": df, "log": ["No sale data -- defaults to 0"],
                "summary": "No sale data. All features set to 0.", "decisions": []}

    ss = special_sales_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    if not pd.api.types.is_datetime64_any_dtype(ss["Date"]):
        ss["Date"] = pd.to_datetime(ss["Date"])

    name_col = find_col(ss, ["Sales Name", "Sales_Name", "Sales_name"])

    ss["year_month"] = ss["Date"].dt.to_period("M")
    df["year_month"] = df["Date"].dt.to_period("M")

    # Sale days per month
    sd = ss.groupby("year_month")["Date"].nunique().reset_index()
    sd.columns = ["year_month", "sale_days"]

    # Sale events per month
    if name_col:
        si = ss.groupby("year_month")[name_col].nunique().reset_index()
        si.columns = ["year_month", "sale_intensity"]
    else:
        si = sd.copy()
        si.columns = ["year_month", "sale_intensity"]

    df = df.merge(sd, on="year_month", how="left")
    df = df.merge(si, on="year_month", how="left")
    df["sale_days"] = df["sale_days"].fillna(0).astype(int)
    df["sale_intensity"] = df["sale_intensity"].fillna(0).astype(int)
    df["sale_flag"] = (df["sale_days"] > 0).astype(int)
    df.drop(columns=["year_month"], inplace=True, errors="ignore")

    n_sale = df["sale_flag"].sum()
    total_days = df["sale_days"].sum()
    log_actions.append(f"sale_flag: {n_sale}/{len(df)} periods had sales")
    log_actions.append(f"sale_days: {total_days} total")

    summary = (
        f"Created 3 sale features. {n_sale} of {len(df)} periods had "
        f"promotions with {total_days} total sale days."
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 3. CHANNEL GROUPING
# =============================================================================

def create_channel_groups(df):
    """
    Group 9 channels into 4 groups to reduce multicollinearity.
    With weekly data (n>=30), individual channels also remain available.
    """
    log_actions = []
    df = df.copy()
    n_rows = len(df)

    for group_name, channel_list in CHANNEL_GROUPS.items():
        cols = [c for c in channel_list if c in df.columns]
        if cols:
            df[f"spend_{group_name}"] = (
                df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            )
            df[f"log_spend_{group_name}"] = np.log1p(df[f"spend_{group_name}"])
            log_actions.append(f"spend_{group_name}: {cols}")
        else:
            df[f"spend_{group_name}"] = 0
            df[f"log_spend_{group_name}"] = 0

    individual_note = (
        "Individual channels also available (n >= 30)."
        if n_rows >= 30 else
        "Groups are primary features (n < 30)."
    )
    summary = (
        f"Grouped {len(MEDIA_CHANNELS)} channels into {len(CHANNEL_GROUPS)} groups. "
        f"{individual_note}"
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 4. DISCOUNT FEATURES
# =============================================================================

def create_discount_features(df):
    """
    Create discount_intensity (ratio) and discount_per_unit.
    Uses ratio to avoid endogeneity (Assumption A6).
    """
    log_actions = []
    df = df.copy()

    if "total_Discount" in df.columns and "total_Mrp" in df.columns:
        df["discount_intensity"] = (df["total_Discount"] / df["total_Mrp"]).clip(0, 1)
        log_actions.append(
            f"discount_intensity: range {df['discount_intensity'].min():.2f} "
            f"to {df['discount_intensity'].max():.2f}"
        )
    else:
        df["discount_intensity"] = 0

    if "total_Discount" in df.columns and "total_Units" in df.columns:
        df["discount_per_unit"] = np.where(
            df["total_Units"] > 0, df["total_Discount"] / df["total_Units"], 0
        )
        log_actions.append(f"discount_per_unit: avg {df['discount_per_unit'].mean():.0f}")

    summary = (
        f"Created discount features. Using ratio (discount/MRP) to avoid "
        f"endogeneity bias."
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 5. LAGGED VARIABLES
# =============================================================================

def create_lagged_features(df, lag_cols=None, n_lags=1):
    """
    Create lagged variables for adstock/carryover effects.
    Each lag costs 1 observation when dropped.
    """
    log_actions = []
    df = df.copy()

    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    if lag_cols is None:
        lag_cols = [c for c in ["total_gmv", "Total.Investment"] if c in df.columns]

    created = []
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            name = f"{col}_lag{lag}"
            df[name] = df[col].shift(lag)
            created.append(name)

    n_nan = df[created].isna().any(axis=1).sum() if created else 0
    log_actions.append(f"Created {len(created)} lagged features: {created}")

    summary = (
        f"Created {len(created)} features with {n_lags}-lag. "
        f"{n_nan} rows will have NaN (first {n_lags} periods). "
        f"Captures adstock/carryover effects."
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 6. NPS PREPARATION
# =============================================================================

def prepare_nps(df):
    """Standardize NPS as a control variable (z-score)."""
    log_actions = []
    df = df.copy()

    if "NPS" not in df.columns:
        return {"data": df, "log": ["NPS not found"], "summary": "NPS not available.",
                "decisions": []}

    mean, std = df["NPS"].mean(), df["NPS"].std()
    if std > 0:
        df["nps_standardized"] = (df["NPS"] - mean) / std
    else:
        df["nps_standardized"] = 0

    corr = df["NPS"].corr(df["total_gmv"]) if "total_gmv" in df.columns else None
    corr_str = f"{corr:.2f}" if corr is not None else "N/A"

    log_actions.append(f"NPS: mean={mean:.1f}, std={std:.1f}, corr_gmv={corr_str}")

    summary = (
        f"NPS prepared as control variable. Range {df['NPS'].min():.0f}-{df['NPS'].max():.0f}, "
        f"standardized. GMV correlation: {corr_str} (seasonal confound, not causal). "
        f"EXCLUDED from model specs (Assumption A4)."
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 6b. SEASONALITY FEATURES
# =============================================================================

def create_seasonality_features(df):
    """
    Create seasonality control features to replace NPS as a seasonal proxy.

    NPS was excluded (Assumption A4) due to -0.96 correlation with GMV driven
    by seasonality, not causality. These features capture seasonal patterns
    directly and transparently.

    Features created:
      - is_festival_season: 1 if Oct-Dec (Dussehra/Diwali/Christmas quarter)
      - month_sin / month_cos: cyclical encoding of month (captures smooth
        seasonal curves without discontinuity at Dec→Jan boundary)
    """
    log_actions = []
    df = df.copy()

    # Determine month from available columns
    month_col = None
    if "Month" in df.columns:
        month_col = "Month"
    elif "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
        df["_month_num"] = df["Date"].dt.month
        month_col = "_month_num"

    if month_col is None:
        return {"data": df, "log": ["No month/date column found"],
                "summary": "Seasonality features unavailable.", "decisions": []}

    months = pd.to_numeric(df[month_col], errors="coerce")

    # Binary: festival season (Oct=10, Nov=11, Dec=12)
    df["is_festival_season"] = (months >= 10).astype(int)
    n_festival = df["is_festival_season"].sum()

    # Cyclical encoding (avoids Dec→Jan discontinuity)
    df["month_sin"] = np.sin(2 * np.pi * months / 12)
    df["month_cos"] = np.cos(2 * np.pi * months / 12)

    # Clean up temp column
    if "_month_num" in df.columns:
        df.drop("_month_num", axis=1, inplace=True)

    log_actions.append(
        f"Created seasonality features: is_festival_season ({n_festival}/{len(df)} periods), "
        f"month_sin, month_cos (cyclical encoding)"
    )

    corr_str = ""
    if "total_gmv" in df.columns:
        for feat in ["is_festival_season", "month_sin", "month_cos"]:
            c = df[feat].corr(df["total_gmv"])
            corr_str += f" {feat}→GMV: {c:+.2f},"
        log_actions.append(f"Seasonality correlations:{corr_str.rstrip(',')}")

    summary = (
        f"Seasonality features created as transparent replacement for NPS proxy. "
        f"is_festival_season: {n_festival} of {len(df)} periods in Oct-Dec. "
        f"month_sin/month_cos: cyclical month encoding for smooth seasonal capture."
    )
    return {"data": df, "log": log_actions, "summary": summary, "decisions": []}


# =============================================================================
# 7. FEATURE MATRIX ASSEMBLY
# =============================================================================

def assemble_feature_matrix(df, drop_lags_na=True):
    """
    Assemble final modeling-ready matrix. Validates feature specs,
    adds weekly-specific specs if n >= 30.
    """
    log_actions = []
    df = df.copy()
    n_rows = len(df)

    # ------------------------------------------------------------------
    # Data-driven channel ranking (by |correlation| with log_total_gmv)
    # Available to ALL specs — base and weekly.
    # ------------------------------------------------------------------
    channel_log_cols = [f"log_{ch}" for ch in MEDIA_CHANNELS if f"log_{ch}" in df.columns]

    if "log_total_gmv" in df.columns and channel_log_cols:
        channel_corr = df[channel_log_cols + ["log_total_gmv"]].corr()["log_total_gmv"].drop("log_total_gmv")
        channel_corr_abs = channel_corr.abs().sort_values(ascending=False)
        ranked_channels = channel_corr_abs.index.tolist()
        log_actions.append(
            f"Channel ranking by |corr| with GMV: "
            + ", ".join(f"{ch}({channel_corr[ch]:+.2f})" for ch in ranked_channels)
        )
    else:
        ranked_channels = [f"log_{ch}" for ch in MEDIA_CHANNELS if f"log_{ch}" in df.columns]
        log_actions.append("Channel ranking: using default order (correlation data unavailable)")

    # Convenience aliases for specs (top2 used by spec_C)
    top2 = ranked_channels[:2] if len(ranked_channels) >= 2 else ranked_channels

    # =================================================================
    # MODEL SPECIFICATIONS — 10 curated specs, each testing a distinct
    # hypothesis about what drives GMV. Designed to be comprehensive
    # without redundancy.
    #
    # With 6 model types × 2 transforms = 120 candidates (weekly).
    #
    # Spec design rationale:
    #   - A-F: Base specs (3 features, safe for monthly n=11)
    #   - G-J: Weekly-only (6-10 features, need n≥30)
    #   - Low-feature specs: test if simple models suffice
    #   - High-feature specs: rely on Ridge/Lasso/ElasticNet to
    #     handle multicollinearity via regularization
    #   - Channel selection is data-driven (ranked by |corr| with GMV)
    #   - NPS kept as seasonality proxy (Assumption A4): negative
    #     correlation with GMV is a seasonality artifact, but removing
    #     it drops R² significantly. Coefficient interpreted as
    #     'seasonal adjustment', NOT 'NPS impact on sales.'
    # =================================================================

    # Base specs (always available — safe for monthly n=11)
    feature_sets = {
        "spec_A_grouped_channels": {
            "target": "log_total_gmv",
            "features": ["log_spend_traditional", "log_spend_digital_performance", "sale_flag"],
            "description": "Hypothesis: Do channel groups explain GMV?",
        },
        "spec_B_total_spend": {
            "target": "log_total_gmv",
            "features": ["log_Total_Investment", "sale_flag", "nps_standardized"],
            "description": "Hypothesis: Does aggregate spend + sale + NPS explain GMV? (baseline)",
        },
        "spec_C_top_channels": {
            "target": "log_total_gmv",
            "features": top2 + ["sale_flag"],
            "description": f"Hypothesis: Can just the best 2 channels explain GMV? (data-driven: {', '.join(top2)})",
        },
        "spec_D_with_momentum": {
            "target": "log_total_gmv",
            "features": ["log_spend_digital_performance", "sale_flag", "total_gmv_lag1"],
            "description": "Hypothesis: Does last period's GMV predict this period? (carryover)",
        },
        "spec_E_discount_effect": {
            "target": "log_total_gmv",
            "features": ["log_Total_Investment", "discount_intensity", "sale_flag"],
            "description": "Hypothesis: Does discounting drive revenue beyond spend?",
        },
        "spec_F_sale_duration": {
            "target": "log_total_gmv",
            "features": ["log_Total_Investment", "sale_days", "nps_standardized"],
            "description": "Hypothesis: Does sale duration matter vs just binary flag?",
        },
    }

    # Weekly-specific specs (need n≥30 for higher feature counts)
    #
    # EDA INSIGHT: Individual channels have high inter-correlations
    # (Online.mkt↔Affiliates r=0.99, Digital↔SEM r=0.97, etc.) making
    # individual channel coefficients unreliable. Weekly specs use CHANNEL
    # GROUPS instead, which aggregate correlated channels and produce
    # stable, interpretable coefficients.
    #
    # To compensate for fewer channel dimensions, we add more control
    # features (discount, sale_days, lag, NPS) for richer models.
    #
    if n_rows >= 30:
        weekly_specs = {
            "spec_G_groups_with_discount": {
                "target": "log_total_gmv",
                "features": ["log_spend_traditional", "log_spend_digital_performance",
                             "log_spend_digital_brand", "sale_flag",
                             "discount_intensity", "nps_standardized"],
                "description": "3 main groups + sale + discount + NPS (6 features)",
            },
            "spec_H_groups_with_momentum": {
                "target": "log_total_gmv",
                "features": ["log_spend_traditional", "log_spend_digital_performance",
                             "log_spend_digital_brand", "sale_flag",
                             "total_gmv_lag1", "nps_standardized"],
                "description": "3 main groups + sale + momentum + NPS (6 features)",
            },
            "spec_I_all_groups_full": {
                "target": "log_total_gmv",
                "features": ["log_spend_traditional", "log_spend_digital_performance",
                             "log_spend_digital_brand", "log_spend_other",
                             "sale_flag", "discount_intensity", "total_gmv_lag1"],
                "description": "All 4 groups + sale + discount + lag (7 features)",
            },
            "spec_J_groups_max_controls": {
                "target": "log_total_gmv",
                "features": ["log_spend_traditional", "log_spend_digital_performance",
                             "log_spend_digital_brand", "log_spend_other",
                             "sale_flag", "discount_intensity", "nps_standardized", "total_gmv_lag1"],
                "description": "All 4 groups + sale + discount + NPS + lag (8 features, max info)",
            },
        }
        feature_sets.update(weekly_specs)
        log_actions.append(f"n={n_rows}: Added {len(weekly_specs)} weekly-specific specs (6-10 features, for regularized models)")

    # Validate which specs are buildable
    valid_sets = {}
    for name, spec in feature_sets.items():
        target_ok = spec["target"] in df.columns
        features_ok = all(f in df.columns for f in spec["features"])
        if target_ok and features_ok:
            valid_sets[name] = spec
            log_actions.append(f"[OK] {name}")
        else:
            missing = [f for f in spec["features"] if f not in df.columns]
            if not target_ok:
                missing.append(f"{spec['target']} (target)")
            log_actions.append(f"[SKIP] {name}: missing {missing}")

    # Drop lag NaN rows
    lag_cols = [c for c in df.columns if "_lag" in c]
    if drop_lags_na and lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        dropped = before - len(df)
        if dropped > 0:
            log_actions.append(f"Dropped {dropped} rows with NaN lags")

    # Keep all potentially useful columns
    all_features = set()
    for spec in valid_sets.values():
        all_features.update(spec["features"])
        all_features.add(spec["target"])

    meta = [c for c in ["Date", "month", "Year", "Month"] if c in df.columns]
    raw = [c for c in [
        "total_gmv", "total_Units", "total_Discount", "total_Mrp",
        "NPS", "Total.Investment", "sale_flag", "sale_days",
        "sale_intensity", "discount_intensity", "discount_per_unit",
        "nps_standardized", "is_festival_season", "month_sin", "month_cos",
        "total_gmv_lag1", "Total.Investment_lag1",
    ] if c in df.columns]
    raw_channels = [c for c in MEDIA_CHANNELS if c in df.columns]
    raw_groups = [c for c in [
        "spend_traditional", "spend_digital_performance",
        "spend_digital_brand", "spend_other",
        "log_spend_traditional", "log_spend_digital_performance",
        "log_spend_digital_brand", "log_spend_other",
    ] if c in df.columns]

    keep = list(set(list(all_features) + meta + raw + raw_channels + raw_groups))
    keep = [c for c in keep if c in df.columns]
    matrix = df[keep].copy()

    summary = (
        f"Feature matrix: {matrix.shape[0]} periods x {matrix.shape[1]} features. "
        f"{len(valid_sets)} specs validated: {', '.join(valid_sets.keys())}."
    )
    return {
        "data": matrix, "feature_sets": valid_sets,
        "log": log_actions, "summary": summary, "decisions": [],
    }


# =============================================================================
# 8. MULTICOLLINEARITY CHECK (VIF)
# =============================================================================

def check_multicollinearity(df, feature_cols):
    """
    Calculate VIF for a feature set. VIF > 10 = problematic.

    Returns:
        {"vif": DataFrame or None, "log": [], "summary": str, "decisions": []}
    """
    log_actions = []

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return {"vif": None, "log": ["statsmodels not available"],
                "summary": "VIF unavailable.", "decisions": []}

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(X) < len(feature_cols) + 1:
        return {"vif": None, "log": [f"Insufficient rows ({len(X)}) for VIF"],
                "summary": "Too few rows for VIF.", "decisions": []}

    try:
        vif = pd.DataFrame({
            "feature": feature_cols,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))]
        })
    except Exception as e:
        return {"vif": None, "log": [f"VIF failed: {e}"],
                "summary": f"VIF calculation error: {e}", "decisions": []}

    for _, row in vif.iterrows():
        tag = "[OK]" if row["VIF"] < 5 else "[WARN]" if row["VIF"] < 10 else "[HIGH]"
        log_actions.append(f"  {tag} {row['feature']:35s} VIF = {row['VIF']:.2f}")

    high = vif[vif["VIF"] > MODEL_SETTINGS["vif_threshold"]]
    decisions = []
    if len(high) > 0:
        decisions.append({
            "item": "multicollinearity",
            "action": f"High VIF: {high['feature'].tolist()}",
            "auto": False,
        })

    summary = (
        f"VIF on {len(feature_cols)} features. Max = {vif['VIF'].max():.2f}. "
        f"{'All OK.' if len(high) == 0 else f'{len(high)} features high VIF.'}"
    )
    return {"vif": vif, "log": log_actions, "summary": summary, "decisions": decisions}


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def plot_feature_distributions(df, feature_cols, save_dir=None):
    """Histogram of each engineered feature."""
    save_dir = save_dir or get_paths()["plots_dir"]
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return

    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        axes[i].hist(v, bins=8, color="#2196F3", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].axvline(v.mean(), color="red", ls="--", alpha=0.7, label=f"mean={v.mean():.2f}")
        axes[i].legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    _save_plot(fig, save_dir, "feature_distributions.png")


def plot_feature_vs_target(df, feature_cols, target="log_total_gmv", save_dir=None):
    """Scatter of each feature vs target with trend line."""
    save_dir = save_dir or get_paths()["plots_dir"]
    if target not in df.columns:
        return
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return

    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    fig.suptitle(f"Features vs {target}", fontsize=14, fontweight="bold")

    for i, col in enumerate(cols):
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        mask = x.notna() & y.notna()
        axes[i].scatter(x[mask], y[mask], s=60, color="#9C27B0", edgecolors="white")
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel(target, fontsize=9)
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            xl = np.linspace(x[mask].min(), x[mask].max(), 50)
            axes[i].plot(xl, p(xl), "r--", alpha=0.6)
            axes[i].set_title(f"r={x[mask].corr(y[mask]):.3f}", fontsize=10)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    _save_plot(fig, save_dir, "features_vs_target.png")


def _save_plot(fig, save_dir, filename):
    """Save figure and close."""
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  [OK] Saved %s", filename)


# =============================================================================
# 10. MASTER PIPELINE
# =============================================================================

def run_feature_engineering(clean_data, save_dir=None):
    """
    Run complete feature engineering pipeline.

    Args:
        clean_data: dict of DataFrames (from outlier pipeline or aggregation)
        save_dir:   plot output directory

    Returns:
        {
            "data":         feature matrix DataFrame,
            "feature_sets": validated model specs,
            "vif_results":  VIF per spec,
            "full_log":     all actions,
            "summaries":    step summaries,
            "decisions":    all decisions,
        }
        or None on critical failure.
    """
    save_dir = save_dir or get_paths()["plots_dir"]

    print("=" * 70)
    print("[START] FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    full_log = []
    summaries = {}
    all_decisions = []

    if "monthly" not in clean_data:
        logger.error("Monthly data not found")
        return None

    df = clean_data["monthly"].copy()
    special_sales = clean_data.get("special_sales")

    # Step 1: Log transforms
    print("\n[STEP 1] Log Transformations...")
    try:
        r = apply_log_transforms(df)
        df = r["data"]; full_log.extend(r["log"]); summaries["log_transforms"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Log transforms failed: %s", e)
        return None  # Fatal -- can't model without log target

    # Step 2: Sale features
    print("\n[STEP 2] Sale Event Features...")
    try:
        r = create_sale_features(df, special_sales)
        df = r["data"]; full_log.extend(r["log"]); summaries["sale_features"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Sale features failed: %s", e)
        df["sale_flag"] = 0; df["sale_days"] = 0; df["sale_intensity"] = 0

    # Step 3: Channel grouping
    print("\n[STEP 3] Channel Grouping...")
    try:
        r = create_channel_groups(df)
        df = r["data"]; full_log.extend(r["log"]); summaries["channel_groups"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Channel grouping failed: %s", e)

    # Step 4: Discount features
    print("\n[STEP 4] Discount Features...")
    try:
        r = create_discount_features(df)
        df = r["data"]; full_log.extend(r["log"]); summaries["discount_features"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Discount features failed: %s", e)
        df["discount_intensity"] = 0

    # Step 5: Lagged variables
    print("\n[STEP 5] Lagged Variables...")
    try:
        r = create_lagged_features(df, n_lags=1)
        df = r["data"]; full_log.extend(r["log"]); summaries["lagged_features"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Lagged features failed: %s", e)

    # Step 6: NPS
    print("\n[STEP 6] NPS Preparation...")
    try:
        r = prepare_nps(df)
        df = r["data"]; full_log.extend(r["log"]); summaries["nps"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  NPS preparation failed: %s", e)

    # Step 6b: Seasonality features (replaces NPS as seasonal control)
    print("\n[STEP 6b] Seasonality Features...")
    try:
        r = create_seasonality_features(df)
        df = r["data"]; full_log.extend(r["log"]); summaries["seasonality"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Seasonality features failed: %s", e)

    # Step 7: Assembly
    print("\n[STEP 7] Assembling Feature Matrix...")
    try:
        r = assemble_feature_matrix(df, drop_lags_na=True)
        matrix = r["data"]; feature_sets = r["feature_sets"]
        full_log.extend(r["log"]); summaries["assembly"] = r["summary"]
        print(f"  {r['summary']}")
    except Exception as e:
        logger.error("  Assembly failed: %s", e)
        return None  # Fatal

    # Step 8: VIF
    print("\n[STEP 8] Multicollinearity (VIF)...")
    vif_results = {}
    for name, spec in feature_sets.items():
        cols = spec["features"]
        if all(c in matrix.columns for c in cols):
            try:
                vr = check_multicollinearity(matrix, cols)
                vif_results[name] = vr
                full_log.extend(vr["log"])
                all_decisions.extend(vr["decisions"])
                print(f"  {name}: {vr['summary']}")
            except Exception as e:
                logger.error("  VIF failed for %s: %s", name, e)

    # Step 9: Plots
    print("\n[STEP 9] Visualizations...")
    plot_cols = [c for c in [
        "log_spend_traditional", "log_spend_digital_performance",
        "log_spend_digital_brand", "log_Total_Investment",
        "sale_flag", "sale_days", "discount_intensity", "nps_standardized",
    ] if c in matrix.columns]
    try:
        plot_feature_distributions(matrix, plot_cols, save_dir)
        plot_feature_vs_target(matrix, plot_cols, "log_total_gmv", save_dir)
    except Exception as e:
        logger.error("  Plotting failed: %s", e)

    # Summary
    print("\n" + "=" * 70)
    print("[DONE] FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"  Matrix shape:  {matrix.shape}")
    print(f"  Specs ready:   {list(feature_sets.keys())}")
    print(f"  Actions:       {len(full_log)}")
    print("\n  Specifications:")
    for name, spec in feature_sets.items():
        print(f"    {name}: {spec['description']}")
        print(f"      Features: {spec['features']}")

    return {
        "data": matrix, "feature_sets": feature_sets,
        "vif_results": vif_results, "full_log": full_log,
        "summaries": summaries, "decisions": all_decisions,
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    from eda_pipeline import load_all_data
    from outlier_detection import run_outlier_pipeline

    data = load_all_data()
    clean_data, _, _ = run_outlier_pipeline(data)
    result = run_feature_engineering(clean_data)