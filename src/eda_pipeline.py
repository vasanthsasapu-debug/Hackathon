"""
=============================================================================
eda_pipeline.py -- Data Loading & Exploratory Data Analysis
=============================================================================
Modules:
  1. Data Loading (all 7 datasets with format-specific handling)
  2. Auto Column Classification
  3. Data Validation
  4. Data Quality Reports
  5. National-Level Trends
  6. Category-Level Breakdown
  7. Media Investment Analysis
  8. Correlation Analysis
  9. Special Sale Impact
  10. Channel Overlap (Traditional vs Digital)
  11. NPS vs Revenue
  12. Master EDA Runner
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_FILES, MEDIA_CHANNELS, get_paths, get_channel_cols, find_col, logger
)

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

CRORE = 1e7
LAKH = 1e5
THOUSAND = 1e3


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_all_data(data_dir=None):
    """
    Load all datasets defined in config.DATA_FILES.

    Args:
        data_dir: path to data folder. Auto-detects if None.

    Returns:
        dict of {dataset_name: DataFrame}
    """
    if data_dir is None:
        data_dir = get_paths()["data_dir"]

    data = {}
    logger.info("Loading datasets from %s", data_dir)

    for key, meta in DATA_FILES.items():
        fpath = os.path.join(data_dir, meta["filename"])
        if not os.path.exists(fpath):
            logger.warning("File not found: %s -- skipping '%s'", fpath, key)
            continue
        try:
            kwargs = {"sep": meta["sep"]}
            if meta.get("date_cols"):
                kwargs["parse_dates"] = meta["date_cols"]
            if meta.get("date_format"):
                kwargs["date_format"] = meta["date_format"]

            df = pd.read_csv(fpath, **kwargs)

            # --- Post-load fixes per dataset ---
            df = _post_load_fix(key, df)

            # Drop auto-generated index columns
            for col in ["Unnamed: 0"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            data[key] = df
            logger.info(
                "  [OK] %-20s -> %8s rows x %3d cols  (%s)",
                key, f"{len(df):,}", df.shape[1], meta["filename"],
            )
        except Exception as e:
            logger.error("  Failed to load %s: %s", meta["filename"], e)

    return data


def _post_load_fix(key, df):
    """Apply dataset-specific fixes after loading."""
    if key == "sales":
        # GMV is object dtype -- coerce
        if "GMV" in df.columns:
            df["GMV"] = pd.to_numeric(
                df["GMV"].astype(str).str.strip(), errors="coerce"
            )

    elif key == "investment":
        # ' Affiliates' has leading space
        df.columns = df.columns.str.strip()

    elif key == "products":
        # Frequency has commas (e.g. "1,648,824")
        if "Frequency" in df.columns:
            df["Frequency"] = pd.to_numeric(
                df["Frequency"].astype(str).str.replace(",", "").str.strip(),
                errors="coerce",
            )
    return df


# =============================================================================
# 2. AUTO COLUMN CLASSIFICATION
# =============================================================================

def auto_classify_columns(df, dataset_name=""):
    """
    Classify each column by its likely MMIX role.

    Returns:
        dict of {role: [column_names]}
    """
    roles = {
        "id": [], "date": [], "metric_revenue": [], "metric_units": [],
        "metric_price": [], "metric_discount": [], "metric_investment": [],
        "metric_score": [], "dimension_category": [], "dimension_promotion": [],
        "other": [],
    }

    for col in df.columns:
        cl = col.lower().strip()
        dtype = df[col].dtype

        if "date" in cl or dtype == "datetime64[ns]":
            roles["date"].append(col)
        elif "id" in cl:
            roles["id"].append(col)
        elif any(k in cl for k in ["gmv", "revenue"]):
            roles["metric_revenue"].append(col)
        elif "sales" in cl and "name" in cl:
            roles["dimension_promotion"].append(col)
        elif "unit" in cl:
            roles["metric_units"].append(col)
        elif "mrp" in cl or "price" in cl:
            roles["metric_price"].append(col)
        elif "discount" in cl:
            roles["metric_discount"].append(col)
        elif "nps" in cl:
            roles["metric_score"].append(col)
        elif any(k in cl for k in ["tv", "digital", "sponsor", "content",
                                     "online", "affiliate", "sem", "radio",
                                     "other", "investment"]):
            roles["metric_investment"].append(col)
        elif any(k in cl for k in ["category", "subcategory", "vertical",
                                     "analytic", "product", "sub_category"]):
            roles["dimension_category"].append(col)
        elif cl in ["year", "month"]:
            roles["date"].append(col)
        else:
            roles["other"].append(col)

    roles = {k: v for k, v in roles.items() if v}

    logger.info("  Auto-Classification for '%s':", dataset_name)
    for role, cols in roles.items():
        logger.info("    %-25s -> %s", role, cols)

    return roles


# =============================================================================
# 3. DATA VALIDATION
# =============================================================================

def validate_mmix_data(data):
    """
    Run validation checks across all datasets.

    Returns:
        list of issue strings
    """
    issues = []
    logger.info("Validating data...")

    if "monthly" in data:
        m = data["monthly"]

        # Required columns
        for group, cols in {
            "revenue": ["total_gmv"], "units": ["total_Units"],
            "investment": ["Total.Investment"], "date": ["Date"], "score": ["NPS"],
        }.items():
            for c in cols:
                if c not in m.columns:
                    msg = f"Monthly: Missing '{c}' ({group})"
                    issues.append(msg)
                    logger.error("  %s", msg)

        # Duplicate dates
        if "Date" in m.columns:
            dups = m["Date"].duplicated().sum()
            if dups > 0:
                issues.append(f"Monthly: {dups} duplicate dates")
                logger.warning("  %d duplicate dates in monthly", dups)
            else:
                logger.info("  [OK] No duplicate dates in monthly")

        # Negative values
        for c in m.select_dtypes(include=[np.number]).columns:
            neg = (m[c] < 0).sum()
            if neg > 0:
                issues.append(f"Monthly: {neg} negative in '{c}'")
                logger.warning("  %d negative values in %s", neg, c)

        # Sample size
        n = len(m)
        if n < 24:
            issues.append(f"Monthly: Only {n} months (need 60+ ideally)")
            logger.warning("  Only %d months of data", n)
        logger.info("  Sample size: %d months", n)

        # Channel nulls
        for c in get_channel_cols(m):
            nulls = pd.to_numeric(m[c], errors="coerce").isnull().sum()
            if nulls > 0:
                issues.append(f"Monthly: {nulls} nulls in '{c}'")
                logger.warning("  %d nulls in channel %s", nulls, c)

        # Anomaly check
        if "Date" in m.columns and "total_gmv" in m.columns:
            median = m["total_gmv"].median()
            low = m[m["total_gmv"] < median * 0.05]
            if len(low) > 0:
                dates = low["Date"].dt.strftime("%b-%Y").tolist()
                issues.append(f"Monthly: Anomalous low GMV in {dates}")
                logger.warning("  Anomalous GMV in %s", dates)

    # Transaction checks
    if "transactions" in data:
        tx = data["transactions"]
        gmv_col = find_col(tx, ["gmv_new", "GMV", "gmv"])
        if gmv_col:
            neg = (pd.to_numeric(tx[gmv_col], errors="coerce") < 0).sum()
            if neg > 0:
                issues.append(f"Transactions: {neg} negative GMV")
                logger.warning("  %d negative GMV in transactions", neg)

    # Special sales checks
    if "special_sales" in data:
        ss = data["special_sales"]
        for c in ["Date", "Sales Name"]:
            if c not in ss.columns:
                alt = [col for col in ss.columns
                       if c.lower().replace(" ", "") in col.lower().replace(" ", "")]
                if alt:
                    logger.info("  SpecialSale: '%s' not found, using '%s'", c, alt[0])
                else:
                    issues.append(f"SpecialSale: Missing '{c}'")
                    logger.error("  SpecialSale missing '%s'", c)

    if not issues:
        logger.info("  [OK] All validations passed")
    else:
        logger.info("  %d issues found", len(issues))

    return issues


# =============================================================================
# 4. DATA QUALITY REPORT
# =============================================================================

def data_quality_report(df, name=""):
    """Print data quality summary for a DataFrame."""
    print(f"\n{'='*60}")
    print(f"  DATA QUALITY: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    quality = pd.DataFrame({
        "dtype": df.dtypes, "non_null": df.count(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
    })
    print(quality.to_string())

    num = df.select_dtypes(include=[np.number]).columns
    if len(num) > 0:
        print(f"\n  Numeric Summary:")
        print(df[num].describe().round(2).to_string())

    return quality


# =============================================================================
# 5. NATIONAL-LEVEL TRENDS
# =============================================================================

def national_trends(monthly_df, save_dir=None):
    """Plot GMV, Units, Discount trends over time."""
    save_dir = save_dir or get_paths()["plots_dir"]
    df = _prepare_monthly(monthly_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("National-Level Trends", fontsize=16, fontweight="bold")

    if "total_gmv" in df.columns:
        axes[0, 0].plot(df["Date"], df["total_gmv"] / CRORE, marker="o", color="#2196F3", lw=2)
        axes[0, 0].set_title("Total GMV (Cr)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    if "total_Units" in df.columns:
        axes[0, 1].plot(df["Date"], df["total_Units"] / THOUSAND, marker="s", color="#4CAF50", lw=2)
        axes[0, 1].set_title("Total Units (K)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    if "total_Discount" in df.columns:
        axes[1, 0].plot(df["Date"], df["total_Discount"] / CRORE, marker="^", color="#FF5722", lw=2)
        axes[1, 0].set_title("Total Discount (Cr)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    if "total_Discount" in df.columns and "total_Mrp" in df.columns:
        pct = df["total_Discount"] / df["total_Mrp"] * 100
        axes[1, 1].bar(df["Date"], pct, color="#FF9800", width=20)
        axes[1, 1].set_title("Discount as % of MRP")
    axes[1, 1].tick_params(axis="x", rotation=45)

    _save_plot(fig, save_dir, "national_trends.png")


# =============================================================================
# 6. CATEGORY-LEVEL BREAKDOWN
# =============================================================================

def category_breakdown(monthly_df, save_dir=None):
    """Revenue by product category over time + share pie chart."""
    save_dir = save_dir or get_paths()["plots_dir"]
    df = _prepare_monthly(monthly_df)

    rev_cols = [c for c in df.columns if c.startswith("Revenue_")]
    if not rev_cols:
        logger.warning("No Revenue_ columns found -- skipping category breakdown")
        return None

    categories = [c.replace("Revenue_", "") for c in rev_cols]
    unit_cols = [c for c in df.columns if c.startswith("Units_") and c != "total_Units"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Category-Level Analysis", fontsize=16, fontweight="bold")

    for col, cat in zip(rev_cols, categories):
        axes[0].plot(df["Date"], df[col] / CRORE, marker="o", label=cat, lw=2)
    axes[0].set_title("Revenue by Category (Cr)")
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis="x", rotation=45)

    latest = df.iloc[-1]
    vals = [latest[c] for c in rev_cols if pd.notna(latest[c])]
    cats = [cat for c, cat in zip(rev_cols, categories) if pd.notna(latest[c])]
    axes[1].pie(vals, labels=cats, autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("Set2", len(cats)))
    axes[1].set_title(f"Revenue Share ({latest['Date'].strftime('%b-%Y')})")

    _save_plot(fig, save_dir, "category_breakdown.png")

    summary = pd.DataFrame({
        "Category": categories,
        "Avg Revenue (Cr)": [df[c].mean() / CRORE for c in rev_cols],
        "Avg Units": [df[c].mean() for c in unit_cols] if len(unit_cols) == len(rev_cols) else [None] * len(rev_cols),
        "Std Dev (Cr)": [df[c].std() / CRORE for c in rev_cols],
    }).round(2)
    print("\n  Category Summary:")
    print(summary.to_string(index=False))
    return summary


# =============================================================================
# 7. MEDIA INVESTMENT ANALYSIS
# =============================================================================

def media_investment_analysis(monthly_df, save_dir=None):
    """Channel spend trends and share of spend."""
    save_dir = save_dir or get_paths()["plots_dir"]
    df = _prepare_monthly(monthly_df)
    channels = get_channel_cols(df)
    if not channels:
        logger.warning("No channel columns found -- skipping")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Media Investment Analysis", fontsize=16, fontweight="bold")

    for ch in channels:
        v = pd.to_numeric(df[ch], errors="coerce") / CRORE
        axes[0].plot(df["Date"], v, marker="o", label=ch, lw=2)
    axes[0].set_title("Channel Spend Over Time (Cr)")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=45)

    avg = {ch: pd.to_numeric(df[ch], errors="coerce").mean()
           for ch in channels}
    avg = {k: v for k, v in avg.items() if not np.isnan(v)}
    axes[1].barh(list(avg.keys()), [v / CRORE for v in avg.values()],
                 color=sns.color_palette("viridis", len(avg)))
    axes[1].set_title("Avg Monthly Spend (Cr)")

    _save_plot(fig, save_dir, "media_investment.png")

    total = sum(avg.values())
    share = pd.DataFrame({
        "Channel": list(avg.keys()),
        "Avg Spend (Cr)": [f"{v / CRORE:.2f}" for v in avg.values()],
        "Share %": [f"{v / total * 100:.1f}" for v in avg.values()],
    })
    print("\n  Share of Spend:")
    print(share.to_string(index=False))
    return share


# =============================================================================
# 8. CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(monthly_df, save_dir=None):
    """Correlation matrix: media channels vs GMV."""
    save_dir = save_dir or get_paths()["plots_dir"]
    df = monthly_df.copy()
    channels = get_channel_cols(df)
    target = "total_gmv"
    extras = ["total_Discount", "NPS", "Total.Investment"]

    if target not in df.columns:
        logger.warning("Target '%s' not found -- skipping correlation", target)
        return None

    all_vars = [target] + channels + [v for v in extras if v in df.columns]
    corr = df[all_vars].apply(pd.to_numeric, errors="coerce").corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Correlation Analysis", fontsize=16, fontweight="bold")

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Full Correlation Matrix")

    if channels:
        ch_corr = corr.loc[channels, target].sort_values(ascending=True)
        colors = ["#4CAF50" if v > 0 else "#F44336" for v in ch_corr.values]
        axes[1].barh(ch_corr.index, ch_corr.values, color=colors)
        axes[1].set_title(f"Channel Correlation with {target}")
        axes[1].axvline(x=0, color="black", lw=0.8)
        axes[1].set_xlim(-1, 1)

        print(f"\n  Channel Correlations with {target}:")
        for ch, val in ch_corr.items():
            strength = "strong" if abs(val) > 0.6 else "moderate" if abs(val) > 0.3 else "weak"
            direction = "positive" if val > 0 else "negative"
            print(f"    {ch:25s} -> {val:+.3f} ({strength}, {direction})")

    _save_plot(fig, save_dir, "correlation_analysis.png")
    return corr


# =============================================================================
# 9. SPECIAL SALE IMPACT
# =============================================================================

def special_sale_impact(transactions_df, special_sales_df, save_dir=None):
    """Compare daily GMV on sale vs non-sale days."""
    save_dir = save_dir or get_paths()["plots_dir"]
    tx = transactions_df.copy()
    ss = special_sales_df.copy()

    if "Date" in tx.columns and not pd.api.types.is_datetime64_any_dtype(tx["Date"]):
        tx["Date"] = pd.to_datetime(tx["Date"])
    if "Date" in ss.columns and not pd.api.types.is_datetime64_any_dtype(ss["Date"]):
        ss["Date"] = pd.to_datetime(ss["Date"])

    sale_name_col = find_col(ss, ["Sales Name", "Sales_Name", "Sales_name"])
    gmv_col = find_col(tx, ["gmv_new", "GMV", "gmv"])

    if not sale_name_col or not gmv_col:
        logger.warning("Required columns missing -- skipping sale impact")
        return

    sale_dates = set(ss["Date"].dt.date)
    tx["is_sale"] = tx["Date"].dt.date.apply(lambda d: d in sale_dates)

    daily = tx.groupby(["Date", "is_sale"]).agg(
        daily_gmv=(gmv_col, "sum"), daily_units=("units", "sum"),
    ).reset_index()

    sale_gmv = daily[daily["is_sale"]]["daily_gmv"].values
    nonsale_gmv = daily[~daily["is_sale"]]["daily_gmv"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Special Sale Impact", fontsize=16, fontweight="bold")

    box_data = [nonsale_gmv / LAKH]
    box_labels = ["Non-Sale"]
    if len(sale_gmv) > 0:
        box_data.append(sale_gmv / LAKH)
        box_labels.append("Sale")
    axes[0].boxplot(box_data, labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor="#E3F2FD"))
    axes[0].set_title("Daily GMV (Lakhs)")

    ss_sum = ss.groupby(sale_name_col).size().reset_index(name="Days")
    axes[1].barh(ss_sum[sale_name_col], ss_sum["Days"],
                 color=sns.color_palette("Set2", len(ss_sum)))
    axes[1].set_title("Sale Duration (Days)")

    _save_plot(fig, save_dir, "special_sale_impact.png")

    print("\n  Sale vs Non-Sale:")
    print(f"    Non-Sale -> Avg: {nonsale_gmv.mean() / LAKH:.1f}L, "
          f"Median: {np.median(nonsale_gmv) / LAKH:.1f}L")
    if len(sale_gmv) > 0:
        print(f"    Sale     -> Avg: {sale_gmv.mean() / LAKH:.1f}L, "
              f"Median: {np.median(sale_gmv) / LAKH:.1f}L")
        lift = (sale_gmv.mean() / nonsale_gmv.mean() - 1) * 100
        print(f"    Lift: {lift:+.1f}%")


# =============================================================================
# 10. CHANNEL OVERLAP (Traditional vs Digital)
# =============================================================================

def channel_overlap_analysis(monthly_df, save_dir=None):
    """Compare traditional vs digital spend and their GMV relationship."""
    save_dir = save_dir or get_paths()["plots_dir"]
    df = _prepare_monthly(monthly_df)

    trad = [c for c in ["TV", "Radio", "Sponsorship"] if c in df.columns]
    digi = [c for c in ["Digital", "Online.marketing", "Affiliates", "SEM",
                         "Content.Marketing"] if c in df.columns]
    if not trad and not digi:
        logger.warning("No channel columns -- skipping overlap")
        return

    df["trad_spend"] = df[trad].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    df["digi_spend"] = df[digi].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    df["total_ch"] = df["trad_spend"] + df["digi_spend"]
    df["digi_share"] = np.where(df["total_ch"] > 0, df["digi_spend"] / df["total_ch"] * 100, 0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Traditional vs Digital", fontsize=16, fontweight="bold")

    axes[0].bar(df["Date"], df["trad_spend"] / CRORE, label="Traditional", color="#3F51B5")
    axes[0].bar(df["Date"], df["digi_spend"] / CRORE, bottom=df["trad_spend"] / CRORE,
                label="Digital", color="#00BCD4")
    axes[0].set_title("Spend (Cr)"); axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(df["Date"], df["digi_share"], "o-", color="#00BCD4", lw=2)
    axes[1].set_title("Digital Share %")
    axes[1].axhline(y=50, color="gray", ls="--", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=45)

    if "total_gmv" in df.columns:
        axes[2].scatter(df["trad_spend"] / CRORE, df["total_gmv"] / CRORE,
                        label="Traditional", s=100, color="#3F51B5", edgecolors="white")
        axes[2].scatter(df["digi_spend"] / CRORE, df["total_gmv"] / CRORE,
                        label="Digital", s=100, color="#00BCD4", edgecolors="white")
        axes[2].set_title("Spend vs GMV"); axes[2].legend()

    _save_plot(fig, save_dir, "channel_overlap.png")

    print("\n  Traditional vs Digital:")
    print(f"    Avg Traditional: {df['trad_spend'].mean() / CRORE:.1f} Cr/period")
    print(f"    Avg Digital:     {df['digi_spend'].mean() / CRORE:.1f} Cr/period")
    print(f"    Avg Digital Share: {df['digi_share'].mean():.1f}%")


# =============================================================================
# 11. NPS vs REVENUE
# =============================================================================

def nps_revenue_analysis(monthly_df, save_dir=None):
    """NPS trend alongside GMV + scatter."""
    save_dir = save_dir or get_paths()["plots_dir"]
    df = _prepare_monthly(monthly_df)

    if "NPS" not in df.columns or "total_gmv" not in df.columns:
        logger.warning("NPS or total_gmv missing -- skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("NPS vs Revenue", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.plot(df["Date"], df["total_gmv"] / CRORE, "b-o", label="GMV", lw=2)
    ax2.plot(df["Date"], df["NPS"], "r-s", label="NPS", lw=2)
    ax1.set_ylabel("GMV (Cr)", color="blue")
    ax2.set_ylabel("NPS", color="red")
    ax1.set_title("GMV & NPS Trend")
    ax1.tick_params(axis="x", rotation=45)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left")

    clean = df[["NPS", "total_gmv"]].dropna()
    axes[1].scatter(clean["NPS"], clean["total_gmv"] / CRORE, s=120, c="#9C27B0", edgecolors="white")
    if len(clean) > 2:
        z = np.polyfit(clean["NPS"], clean["total_gmv"] / CRORE, 1)
        p = np.poly1d(z)
        x_line = np.linspace(clean["NPS"].min(), clean["NPS"].max(), 50)
        axes[1].plot(x_line, p(x_line), "r--", alpha=0.7)
    axes[1].set_title("NPS vs GMV")
    axes[1].set_xlabel("NPS"); axes[1].set_ylabel("GMV (Cr)")

    _save_plot(fig, save_dir, "nps_revenue.png")

    corr = df["NPS"].corr(df["total_gmv"])
    direction = "Higher NPS -> Higher Revenue" if corr > 0 else "Higher NPS -> Lower Revenue"
    print(f"\n  NPS-Revenue Correlation: {corr:.3f}")
    print(f"    Direction: {direction}")


# =============================================================================
# 12. MASTER EDA RUNNER
# =============================================================================

def run_full_eda(data_dir=None, save_dir=None):
    """
    Run the complete EDA pipeline.

    Returns:
        (data, classifications, issues, corr_matrix)
    """
    paths = get_paths()
    data_dir = data_dir or paths["data_dir"]
    save_dir = save_dir or paths["plots_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("[START] EDA PIPELINE")
    print("=" * 60)

    print("\n[STEP 1] Loading Data...")
    data = load_all_data(data_dir)
    if not data:
        logger.error("No data loaded")
        return None, None, None, None

    print("\n[STEP 2] Auto-Classifying Columns...")
    classifications = {}
    for name, df in data.items():
        try:
            classifications[name] = auto_classify_columns(df, name)
        except Exception as e:
            logger.error("  Classification failed for %s: %s", name, e)

    print("\n[STEP 3] Validating Data...")
    try:
        issues = validate_mmix_data(data)
    except Exception as e:
        logger.error("  Validation failed: %s", e)
        issues = [str(e)]

    print("\n[STEP 4] Data Quality...")
    for name, df in data.items():
        try:
            data_quality_report(df, name)
        except Exception as e:
            logger.error("  Quality report failed for %s: %s", name, e)

    corr_matrix = None
    if "monthly" in data:
        print("\n[STEP 5] National Trends...")
        try:
            national_trends(data["monthly"], save_dir)
        except Exception as e:
            logger.error("  National trends failed: %s", e)

        print("\n[STEP 6] Category Breakdown...")
        try:
            category_breakdown(data["monthly"], save_dir)
        except Exception as e:
            logger.error("  Category breakdown failed: %s", e)

        print("\n[STEP 7] Media Investment...")
        try:
            media_investment_analysis(data["monthly"], save_dir)
        except Exception as e:
            logger.error("  Media investment failed: %s", e)

        print("\n[STEP 8] Correlation Analysis...")
        try:
            corr_matrix = correlation_analysis(data["monthly"], save_dir)
        except Exception as e:
            logger.error("  Correlation analysis failed: %s", e)

    if "transactions" in data and "special_sales" in data:
        print("\n[STEP 9] Sale Impact...")
        try:
            special_sale_impact(data["transactions"], data["special_sales"], save_dir)
        except Exception as e:
            logger.error("  Sale impact failed: %s", e)

    if "monthly" in data:
        print("\n[STEP 10] Channel Overlap...")
        try:
            channel_overlap_analysis(data["monthly"], save_dir)
        except Exception as e:
            logger.error("  Channel overlap failed: %s", e)

        print("\n[STEP 11] NPS vs Revenue...")
        try:
            nps_revenue_analysis(data["monthly"], save_dir)
        except Exception as e:
            logger.error("  NPS analysis failed: %s", e)

    print("\n" + "=" * 60)
    print("[DONE] EDA PIPELINE COMPLETE")
    print(f"  Plots saved to: {save_dir}/")
    print("=" * 60)

    return data, classifications, issues, corr_matrix


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _prepare_monthly(df):
    """Ensure Date is datetime and sorted."""
    df = df.copy()
    if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])
    if "Date" in df.columns:
        df = df.sort_values("Date")
    return df


def _save_plot(fig, save_dir, filename):
    """Save figure and close."""
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  [OK] Saved %s", filename)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    data, classifications, issues, corr_matrix = run_full_eda()
