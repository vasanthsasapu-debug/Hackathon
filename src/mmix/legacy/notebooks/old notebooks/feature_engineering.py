"""
=============================================================================
FEATURE ENGINEERING -- E-Commerce MMIX
=============================================================================
Designed for agent integration. Every function returns:
    {
        "data": modified DataFrame or dict,
        "log": [list of actions taken],
        "summary": "plain text summary for narrative generation",
        "decisions": [{"item": ..., "action": ..., "auto": True/False}]
    }

Modules:
  1. Log Transformations (channel spend, GMV)
  2. Sale Event Features (flag, intensity, days count)
  3. Channel Grouping (reduce dimensionality, handle multicollinearity)
  4. Discount Features (sale_intensity ratio)
  5. Lagged Variables (adstock proxy)
  6. NPS Preparation
  7. Final Feature Matrix Assembly
  8. Multicollinearity Check (VIF)
  9. Master Pipeline
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style("whitegrid")

# =============================================================================
# CONSTANTS
# =============================================================================
MEDIA_CHANNELS = ['TV', 'Digital', 'Sponsorship', 'Content.Marketing',
                  'Online.marketing', 'Affiliates', 'SEM', 'Radio', 'Other']

# Channel groupings based on correlation analysis:
# - Radio/Other: spurious negative correlation, sparse data -> merge into residual
# - Online.marketing/Affiliates: highly correlated (0.87/0.87) -> combine
# - Content.Marketing: too small (0.8% share) -> merge with Digital
CHANNEL_GROUPS = {
    'traditional': ['TV', 'Sponsorship'],
    'digital_performance': ['Online.marketing', 'Affiliates', 'SEM'],
    'digital_brand': ['Digital', 'Content.Marketing'],
    'other_spend': ['Radio', 'Other']
}


# =============================================================================
# 1. LOG TRANSFORMATIONS
# =============================================================================

def apply_log_transforms(monthly_df, target_col='total_gmv'):
    """
    Apply log(x+1) transforms to channel spend and target variable.
    This gives us elasticity interpretation in log-log regression.
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    # Log-transform target
    if target_col in df.columns:
        df[f'log_{target_col}'] = np.log1p(df[target_col])
        log.append(f"Created log_{target_col} using log(x+1)")

    # Log-transform individual channels
    channels_present = [c for c in MEDIA_CHANNELS if c in df.columns]
    transformed = []
    for ch in channels_present:
        df[ch] = pd.to_numeric(df[ch], errors='coerce').fillna(0)
        col_name = f'log_{ch}'
        df[col_name] = np.log1p(df[ch])
        transformed.append(col_name)

    log.append(f"Log-transformed {len(transformed)} channel columns: {transformed}")

    # Log-transform total investment
    if 'Total.Investment' in df.columns:
        df['log_Total_Investment'] = np.log1p(
            pd.to_numeric(df['Total.Investment'], errors='coerce').fillna(0)
        )
        log.append("Created log_Total_Investment")

    # Log-transform discount
    if 'total_Discount' in df.columns:
        df['log_total_Discount'] = np.log1p(df['total_Discount'])
        log.append("Created log_total_Discount")

    n_features = len(transformed) + 2  # channels + total investment + discount
    summary = (
        f"Applied log(x+1) to {n_features} variables. "
        f"This handles zero-spend months cleanly (log(0+1)=0) and enables "
        f"elasticity interpretation in regression (1% change in spend -> beta% change in GMV)."
    )

    return {
        "data": df,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 2. SALE EVENT FEATURES
# =============================================================================

def create_sale_features(monthly_df, special_sales_df):
    """
    Create sale-related features from SpecialSale.csv:
    - sale_flag: binary (was there a sale this month?)
    - sale_days: count of sale days in the month
    - sale_intensity: number of distinct sale events in the month
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    if special_sales_df is None or 'Date' not in df.columns:
        df['sale_flag'] = 0
        df['sale_days'] = 0
        df['sale_intensity'] = 0
        log.append("No special sales data available -- all sale features set to 0")
        return {
            "data": df, "log": log,
            "summary": "No sale data available. All sale features defaulted to 0.",
            "decisions": decisions
        }

    ss = special_sales_df.copy()

    # Ensure dates are datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    if not pd.api.types.is_datetime64_any_dtype(ss['Date']):
        ss['Date'] = pd.to_datetime(ss['Date'])

    # Detect sale name column
    sale_name_col = _find_col(ss, ['Sales Name', 'Sales_Name', 'Sales_name'])

    # Build monthly aggregation of sale events
    ss['year_month'] = ss['Date'].dt.to_period('M')
    df['year_month'] = df['Date'].dt.to_period('M')

    # Sale days per month
    sale_days = ss.groupby('year_month')['Date'].nunique().reset_index()
    sale_days.columns = ['year_month', 'sale_days']

    # Distinct sale events per month
    if sale_name_col:
        sale_events = ss.groupby('year_month')[sale_name_col].nunique().reset_index()
        sale_events.columns = ['year_month', 'sale_intensity']
    else:
        sale_events = sale_days.copy()
        sale_events.columns = ['year_month', 'sale_intensity']

    # Merge into monthly data
    df = df.merge(sale_days, on='year_month', how='left')
    df = df.merge(sale_events, on='year_month', how='left')

    df['sale_days'] = df['sale_days'].fillna(0).astype(int)
    df['sale_intensity'] = df['sale_intensity'].fillna(0).astype(int)
    df['sale_flag'] = (df['sale_days'] > 0).astype(int)

    # Clean up
    df.drop(columns=['year_month'], inplace=True)

    # Log what we found
    months_with_sales = df['sale_flag'].sum()
    total_sale_days = df['sale_days'].sum()
    log.append(f"sale_flag: {months_with_sales} of {len(df)} months had sales")
    log.append(f"sale_days: {total_sale_days} total sale days across all months")
    log.append(f"sale_intensity: max {df['sale_intensity'].max()} distinct events in a month")

    summary = (
        f"Created 3 sale features. {months_with_sales} of {len(df)} months had "
        f"active promotions with {total_sale_days} total sale days. "
        f"Sale lift was ~116% in EDA, confirming sales are a major GMV driver."
    )

    return {
        "data": df,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 3. CHANNEL GROUPING
# =============================================================================

def create_channel_groups(monthly_df):
    """
    Group channels to reduce dimensionality and handle multicollinearity.

    Groups (based on correlation analysis):
    - traditional: TV + Sponsorship (brand channels, positive correlation with GMV)
    - digital_performance: Online.marketing + Affiliates + SEM (high GMV correlation)
    - digital_brand: Digital + Content.Marketing (moderate GMV correlation)
    - other_spend: Radio + Other (spurious negative correlation -- A9)
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    for group_name, channel_list in CHANNEL_GROUPS.items():
        cols_present = [c for c in channel_list if c in df.columns]
        if cols_present:
            # Raw sum
            df[f'spend_{group_name}'] = (
                df[cols_present].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
            )
            # Log transform of grouped spend
            df[f'log_spend_{group_name}'] = np.log1p(df[f'spend_{group_name}'])

            log.append(
                f"Created spend_{group_name}: {cols_present} -> "
                f"avg {df[f'spend_{group_name}'].mean()/1e7:.1f} Cr/month"
            )
        else:
            df[f'spend_{group_name}'] = 0
            df[f'log_spend_{group_name}'] = 0
            log.append(f"spend_{group_name}: no source columns found, set to 0")

    decisions.append({
        "item": "channel_grouping",
        "action": (
            "Grouped 9 channels into 4 groups to reduce multicollinearity. "
            "Radio/Other merged into other_spend due to spurious negative "
            "correlation (Assumption A9). Online.marketing/Affiliates/SEM "
            "combined as digital_performance (all strongly correlated with GMV)."
        ),
        "auto": True
    })

    summary = (
        f"Grouped {len(MEDIA_CHANNELS)} channels into {len(CHANNEL_GROUPS)} groups. "
        f"This reduces predictor count from 9 to 4, critical with only 11 data points. "
        f"Grouping is based on: channel type (traditional vs digital), "
        f"correlation patterns with GMV, and spend share analysis."
    )

    return {
        "data": df,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 4. DISCOUNT FEATURES
# =============================================================================

def create_discount_features(monthly_df):
    """
    Create discount-related features.
    Uses ratio approach to avoid endogeneity (Assumption A6).
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    # Discount intensity: discount as proportion of MRP
    if 'total_Discount' in df.columns and 'total_Mrp' in df.columns:
        df['discount_intensity'] = df['total_Discount'] / df['total_Mrp']
        df['discount_intensity'] = df['discount_intensity'].clip(0, 1)
        log.append(
            f"Created discount_intensity (discount/MRP): "
            f"range {df['discount_intensity'].min():.2f} to {df['discount_intensity'].max():.2f}"
        )
    else:
        df['discount_intensity'] = 0
        log.append("discount_intensity: source columns missing, set to 0")

    # Discount per unit (average discount given per item sold)
    if 'total_Discount' in df.columns and 'total_Units' in df.columns:
        df['discount_per_unit'] = np.where(
            df['total_Units'] > 0,
            df['total_Discount'] / df['total_Units'],
            0
        )
        log.append(
            f"Created discount_per_unit: "
            f"avg {df['discount_per_unit'].mean():.0f} per unit"
        )

    decisions.append({
        "item": "discount_treatment",
        "action": (
            "Using discount_intensity (ratio) instead of raw discount as feature. "
            "Raw discount is endogenous (Assumption A6) -- company discounts more "
            "when sales are low, creating reverse causality."
        ),
        "auto": True
    })

    summary = (
        f"Created 2 discount features. Using discount_intensity (ratio of discount "
        f"to MRP) as the primary feature instead of raw discount value, to avoid "
        f"endogeneity bias in elasticity estimates."
    )

    return {
        "data": df,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 5. LAGGED VARIABLES
# =============================================================================

def create_lagged_features(monthly_df, lag_cols=None, n_lags=1):
    """
    Create lagged variables for adstock/carryover effects.

    With only 11 months, we limit to 1 lag (loses 1 row per lag).
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    # Ensure sorted by date
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    if lag_cols is None:
        lag_cols = ['total_gmv', 'Total.Investment']
        lag_cols = [c for c in lag_cols if c in df.columns]

    created = []
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            lag_name = f'{col}_lag{lag}'
            df[lag_name] = df[col].shift(lag)
            created.append(lag_name)

    rows_before = len(df)
    # Don't drop NaN rows here -- let the model handle it or drop at assembly
    rows_with_nan = df[created].isna().any(axis=1).sum() if created else 0

    log.append(f"Created {len(created)} lagged features: {created}")
    log.append(f"{rows_with_nan} rows will have NaN in lagged columns (first {n_lags} months)")

    decisions.append({
        "item": "lag_structure",
        "action": (
            f"Created {n_lags}-month lags for {lag_cols}. "
            f"With n=11, using more than 1 lag would lose too many rows. "
            f"Lag-1 captures immediate carryover/adstock effect."
        ),
        "auto": True
    })

    summary = (
        f"Created {len(created)} lagged features with {n_lags}-month lag. "
        f"This captures adstock effects (how last month's spend carries over to "
        f"this month's revenue). Limited to 1 lag due to small sample size -- "
        f"each lag costs 1 observation."
    )

    return {
        "data": df,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 6. NPS PREPARATION
# =============================================================================

def prepare_nps(monthly_df):
    """
    Prepare NPS as a control variable.
    NPS has -0.96 correlation with GMV (confounded by seasonality, not causal).
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    if 'NPS' not in df.columns:
        log.append("NPS column not found -- skipping")
        return {
            "data": df, "log": log,
            "summary": "NPS not available in data.",
            "decisions": decisions
        }

    nps_stats = {
        'mean': df['NPS'].mean(),
        'std': df['NPS'].std(),
        'min': df['NPS'].min(),
        'max': df['NPS'].max(),
        'corr_gmv': df['NPS'].corr(df['total_gmv']) if 'total_gmv' in df.columns else None
    }

    # Standardize NPS (z-score) for comparability in regression
    df['nps_standardized'] = (df['NPS'] - nps_stats['mean']) / nps_stats['std']

    log.append(
        f"NPS stats: mean={nps_stats['mean']:.1f}, "
        f"std={nps_stats['std']:.1f}, range=[{nps_stats['min']:.1f}, {nps_stats['max']:.1f}]"
    )
    if nps_stats['corr_gmv'] is not None:
        log.append(f"NPS-GMV correlation: {nps_stats['corr_gmv']:.3f} (confounded by seasonality)")

    decisions.append({
        "item": "nps_usage",
        "action": (
            "NPS included as CONTROL variable only, not as a causal driver. "
            "The -0.96 correlation with GMV is confounded -- high-demand months "
            "have lower NPS due to delivery pressure, not because NPS hurts sales."
        ),
        "auto": True
    })

    summary = (
        f"NPS prepared as control variable. Range {nps_stats['min']:.0f}-{nps_stats['max']:.0f}, "
        f"standardized for regression. Strong negative correlation with GMV ({nps_stats['corr_gmv']:.2f}) "
        f"is a seasonal confound, not causal."
    )

    return {
        "data": df,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 7. FINAL FEATURE MATRIX ASSEMBLY
# =============================================================================

def assemble_feature_matrix(monthly_df, drop_lags_na=True):
    """
    Assemble the final modeling-ready feature matrix.
    Selects relevant columns and optionally drops rows with NaN from lags.
    """
    log = []
    decisions = []
    df = monthly_df.copy()

    # Define feature sets for different model specifications
    feature_sets = {
        "model_A_grouped": {
            "target": "log_total_gmv",
            "features": [
                "log_spend_traditional",
                "log_spend_digital_performance",
                "sale_flag"
            ],
            "description": "Grouped channels (traditional + digital_performance) + sale flag"
        },
        "model_B_total": {
            "target": "log_total_gmv",
            "features": [
                "log_Total_Investment",
                "sale_flag",
                "nps_standardized"
            ],
            "description": "Total investment + sale flag + NPS control"
        },
        "model_C_top_channels": {
            "target": "log_total_gmv",
            "features": [
                "log_Online.marketing",
                "log_Sponsorship",
                "sale_flag"
            ],
            "description": "Top 2 correlated channels + sale flag"
        },
        "model_D_with_lag": {
            "target": "log_total_gmv",
            "features": [
                "log_spend_digital_performance",
                "sale_flag",
                "total_gmv_lag1"
            ],
            "description": "Digital performance + sale flag + lagged GMV (momentum)"
        },
        "model_E_discount": {
            "target": "log_total_gmv",
            "features": [
                "log_Total_Investment",
                "discount_intensity",
                "sale_flag"
            ],
            "description": "Total investment + discount intensity + sale flag"
        }
    }

    # Validate which feature sets are actually buildable
    valid_sets = {}
    for name, spec in feature_sets.items():
        target_ok = spec['target'] in df.columns
        features_ok = all(f in df.columns for f in spec['features'])
        if target_ok and features_ok:
            valid_sets[name] = spec
            log.append(f"[OK] {name}: all features available -- {spec['description']}")
        else:
            missing = [f for f in spec['features'] if f not in df.columns]
            if not target_ok:
                missing.append(spec['target'] + ' (target)')
            log.append(f"[SKIP] {name}: missing columns {missing}")

    # Drop rows with NaN in lag columns if requested
    lag_cols = [c for c in df.columns if '_lag' in c]
    rows_before = len(df)
    if drop_lags_na and lag_cols:
        df = df.dropna(subset=lag_cols)
        rows_dropped = rows_before - len(df)
        if rows_dropped > 0:
            log.append(f"Dropped {rows_dropped} rows with NaN in lag columns")

    # Collect all possible feature columns
    all_features = set()
    for spec in valid_sets.values():
        all_features.update(spec['features'])
        all_features.add(spec['target'])

    # Add metadata columns
    meta_cols = ['Date', 'month', 'Year', 'Month']
    meta_present = [c for c in meta_cols if c in df.columns]

    # Build final matrix with all potential features
    keep_cols = list(all_features) + meta_present
    # Also keep raw columns for reference
    raw_cols = ['total_gmv', 'total_Units', 'total_Discount', 'total_Mrp',
                'NPS', 'Total.Investment', 'sale_flag', 'sale_days',
                'sale_intensity', 'discount_intensity']
    raw_present = [c for c in raw_cols if c in df.columns]
    keep_cols = list(set(keep_cols + raw_present))
    keep_cols = [c for c in keep_cols if c in df.columns]

    feature_matrix = df[keep_cols].copy()

    decisions.append({
        "item": "feature_matrix",
        "action": (
            f"Assembled feature matrix: {feature_matrix.shape[0]} rows x "
            f"{feature_matrix.shape[1]} columns. "
            f"{len(valid_sets)} model specifications validated and ready."
        ),
        "auto": True
    })

    summary = (
        f"Final feature matrix: {feature_matrix.shape[0]} months x "
        f"{feature_matrix.shape[1]} features. "
        f"{len(valid_sets)} model specifications ready for testing: "
        f"{', '.join(valid_sets.keys())}. "
        f"Each model uses max 3 predictors to respect the n=11 sample size constraint."
    )

    return {
        "data": feature_matrix,
        "feature_sets": valid_sets,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 8. MULTICOLLINEARITY CHECK (VIF)
# =============================================================================

def check_multicollinearity(df, feature_cols):
    """
    Calculate Variance Inflation Factor for a set of features.
    VIF > 10 indicates problematic multicollinearity.
    VIF > 5 is a warning.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    log = []
    decisions = []

    # Prepare numeric matrix
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').dropna()

    if len(X) < len(feature_cols) + 1:
        log.append(
            f"Not enough observations ({len(X)}) for VIF with "
            f"{len(feature_cols)} features -- skipping"
        )
        return {
            "vif": None, "log": log,
            "summary": "Insufficient data for VIF calculation.",
            "decisions": decisions
        }

    vif_data = pd.DataFrame()
    vif_data['feature'] = feature_cols
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) for i in range(len(feature_cols))
    ]

    log.append("VIF Results:")
    for _, row in vif_data.iterrows():
        status = "[OK]" if row['VIF'] < 5 else "[WARN]" if row['VIF'] < 10 else "[HIGH]"
        log.append(f"  {status} {row['feature']:35s} VIF = {row['VIF']:.2f}")

    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        decisions.append({
            "item": "multicollinearity",
            "action": (
                f"High VIF detected for: {high_vif['feature'].tolist()}. "
                f"Consider removing or combining these features."
            ),
            "auto": False  # Needs human review
        })

    summary = (
        f"VIF check on {len(feature_cols)} features. "
        f"Max VIF = {vif_data['VIF'].max():.2f}. "
        f"{'All features below threshold (VIF < 10).' if len(high_vif) == 0 else f'{len(high_vif)} features with high VIF -- review needed.'}"
    )

    return {
        "vif": vif_data,
        "log": log,
        "summary": summary,
        "decisions": decisions
    }


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def plot_feature_distributions(df, feature_cols, save_dir=None):
    """Plot distributions of engineered features."""
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        print("  [WARN] No feature columns to plot")
        return

    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle('Engineered Feature Distributions', fontsize=14, fontweight='bold')

    for i, col in enumerate(cols):
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        axes[i].hist(vals, bins=8, color='#2196F3', edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=10)
        axes[i].axvline(vals.mean(), color='red', ls='--', alpha=0.7, label=f'mean={vals.mean():.2f}')
        axes[i].legend(fontsize=7)

    # Hide unused axes
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [OK] Feature distributions saved to {save_dir}")


def plot_feature_vs_target(df, feature_cols, target='log_total_gmv', save_dir=None):
    """Scatter plots of each feature vs target."""
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    if target not in df.columns:
        print(f"  [WARN] Target '{target}' not found")
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

    fig.suptitle(f'Features vs {target}', fontsize=14, fontweight='bold')

    for i, col in enumerate(cols):
        x = pd.to_numeric(df[col], errors='coerce')
        y = pd.to_numeric(df[target], errors='coerce')
        mask = x.notna() & y.notna()
        axes[i].scatter(x[mask], y[mask], s=60, color='#9C27B0', edgecolors='white')
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel(target, fontsize=9)
        # Trend line
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
            axes[i].plot(x_line, p(x_line), 'r--', alpha=0.6)
            corr = x[mask].corr(y[mask])
            axes[i].set_title(f'r = {corr:.3f}', fontsize=10)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'features_vs_target.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [OK] Feature vs target plots saved to {save_dir}")


# =============================================================================
# 10. MASTER FEATURE ENGINEERING PIPELINE
# =============================================================================

def run_feature_engineering(clean_data, save_dir=None):
    """
    Run the complete feature engineering pipeline.

    Args:
        clean_data: dict of cleaned DataFrames (output of outlier pipeline)
        save_dir: where to save plots

    Returns:
        {
            "data": feature_matrix DataFrame,
            "feature_sets": dict of model specifications,
            "full_log": list of all actions,
            "summaries": dict of step summaries,
            "decisions": list of all decisions
        }
    """
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    print("=" * 70)
    print("[START] FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    full_log = []
    summaries = {}
    all_decisions = []

    if 'monthly' not in clean_data:
        print("[ERROR] Monthly data not found in clean_data")
        return None

    df = clean_data['monthly'].copy()
    special_sales = clean_data.get('special_sales', None)

    # Step 1: Log transforms
    print("\n[STEP 1] Log Transformations...")
    result = apply_log_transforms(df)
    df = result['data']
    full_log.extend(result['log'])
    summaries['log_transforms'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 2: Sale features
    print("\n[STEP 2] Sale Event Features...")
    result = create_sale_features(df, special_sales)
    df = result['data']
    full_log.extend(result['log'])
    summaries['sale_features'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 3: Channel grouping
    print("\n[STEP 3] Channel Grouping...")
    result = create_channel_groups(df)
    df = result['data']
    full_log.extend(result['log'])
    summaries['channel_groups'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 4: Discount features
    print("\n[STEP 4] Discount Features...")
    result = create_discount_features(df)
    df = result['data']
    full_log.extend(result['log'])
    summaries['discount_features'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 5: Lagged variables
    print("\n[STEP 5] Lagged Variables...")
    result = create_lagged_features(df, n_lags=1)
    df = result['data']
    full_log.extend(result['log'])
    summaries['lagged_features'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 6: NPS preparation
    print("\n[STEP 6] NPS Preparation...")
    result = prepare_nps(df)
    df = result['data']
    full_log.extend(result['log'])
    summaries['nps'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 7: Assemble feature matrix
    print("\n[STEP 7] Assembling Feature Matrix...")
    result = assemble_feature_matrix(df, drop_lags_na=True)
    feature_matrix = result['data']
    feature_sets = result['feature_sets']
    full_log.extend(result['log'])
    summaries['assembly'] = result['summary']
    all_decisions.extend(result['decisions'])
    print(f"  {result['summary']}")

    # Step 8: VIF check for each valid feature set
    print("\n[STEP 8] Multicollinearity Check (VIF)...")
    vif_results = {}
    for name, spec in feature_sets.items():
        cols = spec['features']
        available = [c for c in cols if c in feature_matrix.columns]
        if len(available) == len(cols):
            vif_result = check_multicollinearity(feature_matrix, cols)
            vif_results[name] = vif_result
            full_log.extend(vif_result['log'])
            all_decisions.extend(vif_result['decisions'])
            print(f"  {name}: {vif_result['summary']}")

    # Step 9: Visualize
    print("\n[STEP 9] Visualizing Features...")
    plot_cols = [
        'log_spend_traditional', 'log_spend_digital_performance',
        'log_spend_digital_brand', 'log_Total_Investment',
        'sale_flag', 'sale_days', 'discount_intensity', 'nps_standardized'
    ]
    plot_cols = [c for c in plot_cols if c in feature_matrix.columns]
    plot_feature_distributions(feature_matrix, plot_cols, save_dir)
    plot_feature_vs_target(feature_matrix, plot_cols, 'log_total_gmv', save_dir)

    # Summary
    print("\n" + "=" * 70)
    print("[DONE] FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"\n  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Model specs ready:    {list(feature_sets.keys())}")
    print(f"  Total actions:        {len(full_log)}")

    print("\n  Model Specifications:")
    for name, spec in feature_sets.items():
        print(f"    {name}: {spec['description']}")
        print(f"      Features: {spec['features']}")

    return {
        "data": feature_matrix,
        "feature_sets": feature_sets,
        "vif_results": vif_results,
        "full_log": full_log,
        "summaries": summaries,
        "decisions": all_decisions
    }


# =============================================================================
# HELPERS
# =============================================================================

def _find_col(df, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    from eda_pipeline import load_all_data
    from outlier_detection import run_outlier_pipeline

    data = load_all_data()
    clean_data, outlier_log, assumptions = run_outlier_pipeline(data)
    fe_result = run_feature_engineering(clean_data)
