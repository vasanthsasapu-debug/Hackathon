"""
=============================================================================
OUTLIER DETECTION & REMOVAL -- E-Commerce MMIX
=============================================================================
Approach: Conservative, assumption-driven.
Philosophy: Business behavior is NOT an outlier. Only flag what's
            statistically AND logically unjustifiable.

Modules:
  1. Assumptions Registry (documented for submission)
  2. Transaction-Level Cleaning (firstfile.csv, Sales.csv)
  3. Monthly Data Validation (SecondFile.csv)
  4. Statistical Outlier Detection (IQR + Z-score)
  5. Business-Context Outlier Review
  6. Reconciliation Checks
  7. Visualization
  8. Master Pipeline
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


# =============================================================================
# 1. ASSUMPTIONS REGISTRY
# =============================================================================

ASSUMPTIONS = {
    "A1_AUG2015": {
        "description": (
            "Aug 2015 shows extremely low GMV (<5% of median). "
            "Platform likely launched Jul 2015; Aug is ramp-up period."
        ),
        "decision": "EXCLUDE from modeling (insufficient data collection period)",
        "reasoning": (
            "The platform was new. Including a partial ramp-up month would "
            "distort elasticity estimates. Losing 1 of 12 months is costly, "
            "but keeping an anomalous point is worse for model validity."
        )
    },
    "A2_SALE_DISCOUNTS": {
        "description": (
            "Discount spikes (30-40% of MRP) occur during known sale events "
            "(Diwali, Eid, Independence Day, etc.)."
        ),
        "decision": "KEEP all data. Create sale_intensity feature instead of removing.",
        "reasoning": (
            "Discounting during sales is standard e-commerce behavior, not a "
            "data quality issue. Removing these would eliminate the very signal "
            "we need to measure promotional elasticity."
        )
    },
    "A3_CHANNEL_ZEROS": {
        "description": "Radio, Other, Content.Marketing show zeros in several months.",
        "decision": "KEEP zeros as legitimate 'no spend' periods. Use log(x+1) transform.",
        "reasoning": (
            "Not all channels are active every month. Zeros represent real "
            "budget decisions, not missing data. Log(x+1) handles these "
            "cleanly in regression without artificial imputation."
        )
    },
    "A4_NPS_VARIATION": {
        "description": "NPS ranges 44-60 across 12 months.",
        "decision": "KEEP raw NPS values. Not outliers.",
        "reasoning": (
            "This range is normal for a growing e-commerce platform. "
            "NPS shows strong negative correlation with GMV (-0.96), but "
            "this is confounded by seasonality -- high-demand months have "
            "lower NPS due to delivery pressure. NPS is a control variable, "
            "not a causal driver in this context."
        )
    },
    "A5_SEASONAL_PEAKS": {
        "description": (
            "Oct 2015, Sep 2015 show high GMV. Coincides with festival season "
            "(Dussehra, Diwali) and known sale events."
        ),
        "decision": "KEEP as normal seasonality. Sale_flag captures the effect.",
        "reasoning": (
            "Festival-season peaks are the core business pattern we are "
            "modeling. Removing them would eliminate the signal entirely."
        )
    },
    "A6_DISCOUNT_ENDOGENEITY": {
        "description": "Discount may be endogenous (company discounts more when sales are low).",
        "decision": (
            "Use discount as CONTROL variable, not as primary MMIX lever. "
            "Create sale_intensity ratio instead."
        ),
        "reasoning": (
            "If the company increases discounts in response to low sales, "
            "the discount coefficient will be biased. Using it as a control "
            "rather than a lever avoids misinterpreting reverse causality."
        )
    },
    "A7_SAMPLE_SIZE": {
        "description": (
            "Only 12 months of data (Jul 2015 -- Jun 2016). "
            "MMIX typically needs 60+ months for robust elasticities."
        ),
        "decision": (
            "Proceed with simpler models (max 3-4 predictors). "
            "No seasonal dummies (would consume all degrees of freedom). "
            "Use quarterly/half-year indicators instead if needed."
        ),
        "reasoning": (
            "With n=12 (or 11 after Aug exclusion), degrees of freedom are "
            "severely limited. Rule of thumb: n/3 to n/5 predictors max. "
            "Simpler models with fewer predictors will be more stable."
        )
    },
    "A8_SALES_GMV_DTYPE": {
        "description": (
            "Sales.csv GMV column is dtype 'object' -- contains whitespace or "
            "non-numeric entries. Needs coercion to float."
        ),
        "decision": "Coerce to numeric; NaN rows logged and removed.",
        "reasoning": (
            "Object dtype in a numeric field indicates data quality issues "
            "at source. Coercing to numeric and dropping failures is the "
            "standard safe approach."
        )
    },
    "A9_RADIO_OTHER_SPURIOUS": {
        "description": (
            "Radio and Other show strong negative correlation (-0.95) with GMV."
        ),
        "decision": "Exclude from individual channel modeling or merge into residual bucket.",
        "reasoning": (
            "These channels have data only in a few months. The months they "
            "are active happen to be lower-GMV months, creating a false "
            "negative correlation. This is not causal -- Radio does not hurt "
            "sales. The pattern is an artifact of sparse, non-overlapping "
            "activity periods."
        )
    }
}


def print_assumptions():
    """Print all documented assumptions."""
    print("=" * 70)
    print("DOCUMENTED ASSUMPTIONS FOR MMIX PIPELINE")
    print("=" * 70)
    for key, a in ASSUMPTIONS.items():
        print(f"\n  [{key}]")
        print(f"   Description: {a['description']}")
        print(f"   Decision:    {a['decision']}")
        print(f"   Reasoning:   {a['reasoning']}")
    print("=" * 70)


# =============================================================================
# 2. TRANSACTION-LEVEL CLEANING
# =============================================================================

def clean_transactions(df, dataset_name="transactions"):
    """
    Clean transaction-level data (firstfile.csv or Sales.csv).
    Returns: cleaned df, removal log
    """
    log = []
    original_len = len(df)
    print(f"\n[CLEAN] {dataset_name} ({original_len:,} rows)")

    gmv_col = _find_col(df, ['gmv_new', 'GMV', 'gmv'])
    units_col = _find_col(df, ['units', 'Units_sold', 'Units'])
    mrp_col = _find_col(df, ['product_mrp', 'MRP', 'mrp'])
    discount_col = _find_col(df, ['discount', 'Discount'])

    # Coerce GMV to numeric (handles Sales.csv object dtype)
    if gmv_col:
        before_nan = df[gmv_col].isna().sum()
        df[gmv_col] = pd.to_numeric(df[gmv_col], errors='coerce')
        after_nan = df[gmv_col].isna().sum()
        coerced_nan = after_nan - before_nan
        if coerced_nan > 0:
            msg = f"Coerced {coerced_nan} non-numeric GMV values to NaN"
            log.append({"step": "gmv_coerce", "rows_affected": coerced_nan, "reason": msg})
            print(f"  [INFO] {msg}")

    # Remove negative GMV -- impossible value
    if gmv_col:
        neg_gmv = (df[gmv_col] < 0).sum()
        if neg_gmv > 0:
            df = df[df[gmv_col] >= 0]
            msg = f"Removed {neg_gmv} rows with negative GMV (impossible value)"
            log.append({"step": "negative_gmv", "rows_removed": neg_gmv, "reason": msg})
            print(f"  [OK] {msg}")

    # Remove GMV=0 but units>0 -- data error (sold items but no revenue recorded)
    if gmv_col and units_col:
        df[units_col] = pd.to_numeric(df[units_col], errors='coerce')
        bad_rows = ((df[gmv_col] == 0) & (df[units_col] > 0)).sum()
        if bad_rows > 0:
            df = df[~((df[gmv_col] == 0) & (df[units_col] > 0))]
            msg = f"Removed {bad_rows} rows with GMV=0 but units>0 (recording error)"
            log.append({"step": "zero_gmv_with_units", "rows_removed": bad_rows, "reason": msg})
            print(f"  [OK] {msg}")

    # Remove negative units -- impossible value
    if units_col:
        neg_units = (df[units_col] < 0).sum()
        if neg_units > 0:
            df = df[df[units_col] >= 0]
            msg = f"Removed {neg_units} rows with negative units (impossible value)"
            log.append({"step": "negative_units", "rows_removed": neg_units, "reason": msg})
            print(f"  [OK] {msg}")

    # Flag impossible discounts (discount > MRP) -- keep but flag
    if discount_col and mrp_col:
        df[discount_col] = pd.to_numeric(df[discount_col], errors='coerce')
        df[mrp_col] = pd.to_numeric(df[mrp_col], errors='coerce')
        impossible = ((df[discount_col] > df[mrp_col]) & df[mrp_col].notna()).sum()
        if impossible > 0:
            df['flag_impossible_discount'] = (
                (df[discount_col] > df[mrp_col]) & df[mrp_col].notna()
            ).astype(int)
            msg = (f"Flagged {impossible} rows where discount > MRP "
                   f"(kept for review, not removed)")
            log.append({"step": "impossible_discount_flag", "rows_flagged": impossible, "reason": msg})
            print(f"  [WARN] {msg}")

    # Remove negative discounts -- impossible value
    if discount_col:
        neg_disc = (df[discount_col] < 0).sum()
        if neg_disc > 0:
            df = df[df[discount_col] >= 0]
            msg = f"Removed {neg_disc} rows with negative discount (impossible value)"
            log.append({"step": "negative_discount", "rows_removed": neg_disc, "reason": msg})
            print(f"  [OK] {msg}")

    # Drop NaN GMV rows from coercion
    if gmv_col:
        nan_gmv = df[gmv_col].isna().sum()
        if nan_gmv > 0:
            df = df.dropna(subset=[gmv_col])
            msg = f"Removed {nan_gmv} rows with NaN GMV (non-parseable values)"
            log.append({"step": "nan_gmv", "rows_removed": nan_gmv, "reason": msg})
            print(f"  [OK] {msg}")

    final_len = len(df)
    total_removed = original_len - final_len
    pct = (total_removed / original_len * 100) if original_len > 0 else 0
    print(f"  [SUMMARY] {dataset_name}: {original_len:,} -> {final_len:,} "
          f"({total_removed:,} removed, {pct:.2f}%)")

    return df, log


# =============================================================================
# 3. MONTHLY DATA VALIDATION & CLEANING
# =============================================================================

def clean_monthly(df):
    """
    Validate and clean the monthly aggregated dataset (SecondFile.csv).
    Returns: cleaned df, removal log, validation results
    """
    log = []
    validation = {}
    print(f"\n[CLEAN] Monthly Data ({len(df)} rows)")

    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Duplicate date check
    if 'Date' in df.columns:
        dup_dates = df['Date'].duplicated().sum()
        validation['duplicate_dates'] = dup_dates
        if dup_dates > 0:
            print(f"  [WARN] {dup_dates} duplicate dates found -- removing")
            df = df.drop_duplicates(subset='Date', keep='first')
            log.append({"step": "duplicate_dates", "rows_removed": dup_dates,
                        "reason": "Duplicate monthly records removed (kept first occurrence)"})
        else:
            print("  [OK] No duplicate dates")

    # Aug 2015 exclusion (Assumption A1)
    ratio = None
    aug_excluded = False
    if 'Date' in df.columns and 'total_gmv' in df.columns:
        aug_2015 = df[(df['Date'].dt.year == 2015) & (df['Date'].dt.month == 8)]
        if len(aug_2015) > 0:
            median_gmv = df['total_gmv'].median()
            aug_gmv = aug_2015['total_gmv'].values[0]
            ratio = aug_gmv / median_gmv if median_gmv > 0 else 0

            if ratio < 0.05:
                df = df[~((df['Date'].dt.year == 2015) & (df['Date'].dt.month == 8))]
                msg = (f"EXCLUDED Aug 2015: GMV={aug_gmv/1e7:.2f}Cr is {ratio*100:.1f}% "
                       f"of median {median_gmv/1e7:.1f}Cr "
                       f"(Assumption A1: platform ramp-up period)")
                log.append({"step": "aug_2015_exclusion", "rows_removed": 1, "reason": msg})
                print(f"  [EXCLUDE] {msg}")
                aug_excluded = True
            else:
                print(f"  [INFO] Aug 2015 GMV ratio = {ratio*100:.1f}% -- keeping")
    validation['aug_2015_excluded'] = aug_excluded

    # Negative value check -- impossible for revenue/spend metrics
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"  [WARN] {col}: {neg_count} negative values -- setting to 0")
            df.loc[df[col] < 0, col] = 0
            log.append({"step": f"negative_{col}", "rows_affected": neg_count,
                        "reason": f"Negative values in {col} set to 0 (impossible for this metric)"})

    # Channel spend validation -- coerce and fill zeros (Assumption A3)
    channels = _get_channel_cols(df)
    for ch in channels:
        df[ch] = pd.to_numeric(df[ch], errors='coerce').fillna(0)
    channel_zeros = {ch: (df[ch] == 0).sum() for ch in channels}
    validation['channel_zero_months'] = channel_zeros

    has_zeros = {ch: z for ch, z in channel_zeros.items() if z > 0}
    if has_zeros:
        print(f"  [INFO] Channel zero-months (Assumption A3 -- legitimate 'no spend'):")
        for ch, z in has_zeros.items():
            print(f"         {ch:25s} -> {z} months with zero spend (kept)")
    else:
        print("  [OK] All channels have non-zero spend in all months")

    # NPS validation
    if 'NPS' in df.columns:
        nps_range = (df['NPS'].min(), df['NPS'].max())
        validation['nps_range'] = nps_range
        if nps_range[0] < -100 or nps_range[1] > 100:
            print(f"  [WARN] NPS out of valid range: {nps_range}")
        else:
            print(f"  [OK] NPS range valid: {nps_range[0]:.1f} to {nps_range[1]:.1f}")

    # Discount sanity check
    if 'total_Discount' in df.columns and 'total_Mrp' in df.columns:
        df['discount_ratio'] = df['total_Discount'] / df['total_Mrp']
        invalid = (df['discount_ratio'] > 1).sum()
        if invalid > 0:
            print(f"  [WARN] {invalid} months where discount > MRP (capping at MRP)")
            df.loc[df['discount_ratio'] > 1, 'total_Discount'] = (
                df.loc[df['discount_ratio'] > 1, 'total_Mrp']
            )
        else:
            print("  [OK] All discounts <= MRP")
        validation['discount_valid'] = invalid == 0

    print(f"  [SUMMARY] Monthly: {len(df)} months remaining")
    if 'Date' in df.columns:
        print(f"            Range: {df['Date'].min().strftime('%b-%Y')} "
              f"to {df['Date'].max().strftime('%b-%Y')}")

    return df, log, validation


# =============================================================================
# 4. STATISTICAL OUTLIER DETECTION
# =============================================================================

def detect_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Detect outliers using IQR method.
    Flags only -- does NOT auto-remove. Business review needed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n[OUTLIER] IQR Detection (multiplier={multiplier})...")
    outlier_flags = pd.DataFrame(index=df.index)
    summary = {}

    for col in columns:
        vals = pd.to_numeric(df[col], errors='coerce')
        if vals.isna().all():
            continue

        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        is_outlier = (vals < lower) | (vals > upper)
        outlier_flags[f'{col}_outlier'] = is_outlier.astype(int)

        n_outliers = is_outlier.sum()
        summary[col] = {
            'q1': q1, 'q3': q3, 'iqr': iqr,
            'lower_bound': lower, 'upper_bound': upper,
            'n_outliers': n_outliers,
            'pct_outliers': (n_outliers / len(df) * 100) if len(df) > 0 else 0
        }

        if n_outliers > 0:
            outlier_vals = vals[is_outlier].tolist()
            print(f"  [FLAG] {col}: {n_outliers} outliers "
                  f"(bounds: [{lower:.2f}, {upper:.2f}])")
            print(f"         Values: {outlier_vals}")
        else:
            print(f"  [OK]   {col}: No outliers")

    return outlier_flags, summary


def detect_outliers_zscore(df, columns=None, threshold=2.5):
    """
    Detect outliers using Z-score method.
    Threshold=2.5 is relaxed for small samples (n=12).
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n[OUTLIER] Z-score Detection (threshold={threshold})...")
    outlier_flags = pd.DataFrame(index=df.index)
    summary = {}

    for col in columns:
        vals = pd.to_numeric(df[col], errors='coerce')
        if vals.isna().all():
            continue

        mean = vals.mean()
        std = vals.std()
        if std == 0:
            print(f"  [SKIP] {col}: zero variance")
            continue

        z_scores = np.abs((vals - mean) / std)
        is_outlier = z_scores > threshold
        outlier_flags[f'{col}_zscore_outlier'] = is_outlier.astype(int)

        n_outliers = is_outlier.sum()
        summary[col] = {
            'mean': mean, 'std': std, 'threshold': threshold,
            'n_outliers': n_outliers
        }

        if n_outliers > 0:
            print(f"  [FLAG] {col}: {n_outliers} outliers (|z| > {threshold})")
        else:
            print(f"  [OK]   {col}: No outliers")

    return outlier_flags, summary


# =============================================================================
# 5. BUSINESS-CONTEXT OUTLIER REVIEW
# =============================================================================

def business_context_review(df, special_sales_df=None):
    """
    Review flagged outliers against business context.
    Decides: keep (with explanation) or exclude (with justification).
    """
    print("\n[REVIEW] Business Context Outlier Review...")
    decisions = []

    if 'Date' not in df.columns or 'total_gmv' not in df.columns:
        print("  [WARN] Date or total_gmv column missing -- skipping")
        return decisions

    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    median_gmv = df['total_gmv'].median()

    # Build set of sale months
    sale_months = set()
    if special_sales_df is not None:
        ss = special_sales_df.copy()
        if 'Date' in ss.columns:
            if not pd.api.types.is_datetime64_any_dtype(ss['Date']):
                ss['Date'] = pd.to_datetime(ss['Date'])
            sale_months = set(ss['Date'].dt.to_period('M'))

    for idx, row in df.iterrows():
        month_label = row['Date'].strftime('%b-%Y')
        gmv = row['total_gmv']
        ratio = gmv / median_gmv if median_gmv > 0 else 0
        current_period = row['Date'].to_period('M')
        is_sale_month = current_period in sale_months

        if ratio < 0.10:
            decision = {
                'month': month_label, 'gmv_ratio': ratio,
                'is_sale_month': is_sale_month,
                'action': 'EXCLUDE',
                'reason': f'GMV at {ratio*100:.1f}% of median -- likely data issue or ramp-up'
            }
        elif ratio > 2.0 and is_sale_month:
            decision = {
                'month': month_label, 'gmv_ratio': ratio,
                'is_sale_month': True,
                'action': 'KEEP',
                'reason': f'High GMV ({ratio*100:.0f}% of median) explained by sale events'
            }
        elif ratio > 2.0 and not is_sale_month:
            decision = {
                'month': month_label, 'gmv_ratio': ratio,
                'is_sale_month': False,
                'action': 'FLAG_REVIEW',
                'reason': f'High GMV ({ratio*100:.0f}% of median) without known sale'
            }
        else:
            continue  # Normal range, no action needed

        decisions.append(decision)
        tag = {"EXCLUDE": "[EXCLUDE]", "KEEP": "[KEEP]", "FLAG_REVIEW": "[FLAG]"}
        print(f"  {tag.get(decision['action'], '[?]')} {month_label}: {decision['reason']}")

    if not decisions:
        print("  [OK] All months within normal business range")

    return decisions


# =============================================================================
# 6. RECONCILIATION CHECKS
# =============================================================================

def reconciliation_checks(monthly_df, investment_df=None, tolerance=0.05):
    """Validate roll-ups and cross-dataset consistency."""
    print("\n[RECON] Reconciliation Checks...")
    results = {}
    m = monthly_df.copy()

    # Revenue roll-up: sum of category revenues vs total_gmv
    rev_cols = [c for c in m.columns if c.startswith('Revenue_')]
    if rev_cols and 'total_gmv' in m.columns:
        sum_rev = m[rev_cols].sum(axis=1)
        diff_pct = abs(sum_rev - m['total_gmv']) / m['total_gmv'] * 100
        max_diff = diff_pct.max()
        results['revenue_rollup_max_diff_pct'] = max_diff
        if max_diff > tolerance * 100:
            print(f"  [WARN] Revenue roll-up: Max diff = {max_diff:.2f}% "
                  f"(tolerance: {tolerance*100}%)")
        else:
            print(f"  [OK] Revenue roll-up valid (max diff: {max_diff:.2f}%)")

    # Units roll-up
    unit_cols = [c for c in m.columns if c.startswith('Units_') and c != 'total_Units']
    if unit_cols and 'total_Units' in m.columns:
        sum_units = m[unit_cols].sum(axis=1)
        diff_pct = abs(sum_units - m['total_Units']) / m['total_Units'] * 100
        max_diff = diff_pct.max()
        results['units_rollup_max_diff_pct'] = max_diff
        if max_diff > tolerance * 100:
            print(f"  [WARN] Units roll-up: Max diff = {max_diff:.2f}%")
        else:
            print(f"  [OK] Units roll-up valid (max diff: {max_diff:.2f}%)")

    # Investment roll-up: sum of channels vs Total.Investment
    channels = _get_channel_cols(m)
    if channels and 'Total.Investment' in m.columns:
        ch_sum = m[channels].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        total_inv = pd.to_numeric(m['Total.Investment'], errors='coerce')
        diff_pct = abs(ch_sum - total_inv) / total_inv * 100
        max_diff = diff_pct.max()
        results['investment_rollup_max_diff_pct'] = max_diff
        if max_diff > tolerance * 100:
            print(f"  [WARN] Investment roll-up: Max diff = {max_diff:.2f}%")
            bad_mask = diff_pct > tolerance * 100
            if 'Date' in m.columns:
                for idx in m[bad_mask].index:
                    print(f"         {m.loc[idx, 'Date'].strftime('%b-%Y')}: "
                          f"{diff_pct.loc[idx]:.1f}% off")
        else:
            print(f"  [OK] Investment roll-up valid (max diff: {max_diff:.2f}%)")

    # Cross-validate with MediaInvestment.csv (redundancy check)
    if investment_df is not None:
        inv_total_col = _find_col(investment_df, ['Total Investment', 'Total.Investment'])
        if inv_total_col and 'Total.Investment' in m.columns:
            print("  [INFO] Cross-validating with MediaInvestment.csv...")
            inv_total = pd.to_numeric(investment_df[inv_total_col], errors='coerce').sum()
            monthly_total = pd.to_numeric(m['Total.Investment'], errors='coerce').sum()
            if monthly_total > 0:
                cross_diff = abs(inv_total - monthly_total) / monthly_total * 100
                results['cross_dataset_diff_pct'] = cross_diff
                if cross_diff > tolerance * 100:
                    print(f"  [WARN] Cross-dataset diff: {cross_diff:.1f}%")
                else:
                    print(f"  [OK] Cross-dataset valid ({cross_diff:.1f}% diff)")

    return results


# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def plot_outlier_summary(df, outlier_flags, key_cols=None, save_dir=None):
    """Visualize outlier detection results for monthly data."""
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    if key_cols is None:
        key_cols = ['total_gmv', 'Total.Investment', 'NPS']
        key_cols = [c for c in key_cols if c in df.columns]

    if not key_cols:
        print("  [WARN] No key columns to plot")
        return

    n = len(key_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle('Outlier Detection Summary (Monthly)', fontsize=14, fontweight='bold')

    for ax, col in zip(axes, key_cols):
        vals = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        flag_col = f'{col}_outlier'
        if flag_col in outlier_flags.columns:
            colors = ['red' if f else '#2196F3' for f in outlier_flags[flag_col]]
        else:
            colors = ['#2196F3'] * len(vals)

        ax.scatter(range(len(vals)), vals, c=colors, s=80, edgecolors='white', zorder=3)
        ax.axhline(y=upper, color='red', ls='--', alpha=0.5, label=f'Upper: {upper:.0f}')
        ax.axhline(y=lower, color='red', ls='--', alpha=0.5, label=f'Lower: {lower:.0f}')
        ax.axhline(y=vals.median(), color='green', ls='-', alpha=0.5, label='Median')
        ax.set_title(col)
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'outlier_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  [OK] Outlier summary plot saved to {save_dir}")


# =============================================================================
# 8. MASTER OUTLIER PIPELINE
# =============================================================================

def run_outlier_pipeline(data, save_dir=None):
    """
    Run the complete outlier detection and removal pipeline.

    Args:
        data: dict of DataFrames from load_all_data()
        save_dir: where to save plots (auto-detects if None)

    Returns:
        clean_data: dict of cleaned DataFrames
        full_log: list of all removal/flag decisions
        assumptions: documented assumptions dict
    """
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    print("=" * 70)
    print("[START] OUTLIER DETECTION & REMOVAL PIPELINE")
    print("=" * 70)

    full_log = []
    clean_data = {}

    print_assumptions()

    # Step 1: Clean transactions
    if 'transactions' in data:
        clean_tx, tx_log = clean_transactions(
            data['transactions'].copy(), "transactions (firstfile.csv)"
        )
        clean_data['transactions'] = clean_tx
        full_log.extend(tx_log)

    if 'sales' in data:
        clean_sales, sales_log = clean_transactions(
            data['sales'].copy(), "sales (Sales.csv)"
        )
        clean_data['sales'] = clean_sales
        full_log.extend(sales_log)

    # Step 2: Clean monthly data
    if 'monthly' in data:
        clean_monthly_df, monthly_log, validation = clean_monthly(
            data['monthly'].copy()
        )
        clean_data['monthly'] = clean_monthly_df
        full_log.extend(monthly_log)

        # Step 3: Statistical outlier detection (monthly)
        key_metric_cols = ['total_gmv', 'total_Units', 'total_Discount', 'NPS']
        key_metric_cols = [c for c in key_metric_cols if c in clean_monthly_df.columns]
        channels = _get_channel_cols(clean_monthly_df)
        detect_cols = key_metric_cols + channels

        iqr_flags, iqr_summary = detect_outliers_iqr(
            clean_monthly_df, columns=detect_cols
        )
        zscore_flags, zscore_summary = detect_outliers_zscore(
            clean_monthly_df, columns=detect_cols, threshold=2.5
        )

        # Step 4: Business context review
        special_sales = data.get('special_sales', None)
        decisions = business_context_review(clean_monthly_df, special_sales)

        # Step 5: Reconciliation
        investment = data.get('investment', None)
        recon = reconciliation_checks(clean_monthly_df, investment)

        # Step 6: Visualize
        plot_outlier_summary(clean_monthly_df, iqr_flags, save_dir=save_dir)

    # Pass through other datasets unchanged
    for key in ['special_sales', 'nps', 'products', 'investment']:
        if key in data and key not in clean_data:
            clean_data[key] = data[key]

    # Summary
    print("\n" + "=" * 70)
    print("[DONE] OUTLIER PIPELINE COMPLETE")
    print("=" * 70)

    total_actions = len(full_log)
    print(f"\n  Total cleaning actions: {total_actions}")
    print(f"  Datasets cleaned:       {list(clean_data.keys())}")

    if full_log:
        print("\n  Cleaning Log:")
        for entry in full_log:
            removed = entry.get('rows_removed', entry.get('rows_affected', entry.get('rows_flagged', 0)))
            print(f"    - [{entry['step']}] {entry['reason']} ({removed} rows)")

    return clean_data, full_log, ASSUMPTIONS


# =============================================================================
# HELPERS
# =============================================================================

def _find_col(df, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_channel_cols(df):
    """Get media channel columns present in the dataframe."""
    return [c for c in MEDIA_CHANNELS if c in df.columns]


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    from eda_pipeline import load_all_data

    data = load_all_data()
    clean_data, log, assumptions = run_outlier_pipeline(data)
