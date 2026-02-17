"""
=============================================================================
data_aggregation.py -- Weekly / Monthly Data Aggregation
=============================================================================
Builds the modeling dataset at the requested granularity.

Weekly:  Aggregates firstfile.csv to weekly GMV/units/discount,
         distributes monthly channel spend evenly across weeks,
         maps sale events to exact weeks, repeats monthly NPS.

Monthly: Passes SecondFile.csv through unchanged.

Usage:
    from data_aggregation import build_modeling_dataset
    result = build_modeling_dataset(clean_data, granularity='weekly')
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import MEDIA_CHANNELS, get_paths, get_channel_cols, find_col, logger


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_modeling_dataset(clean_data, granularity="weekly"):
    """
    Build the modeling-ready dataset at the requested granularity.

    Args:
        clean_data:  dict of cleaned DataFrames (from outlier pipeline)
        granularity: 'weekly' or 'monthly'

    Returns:
        {
            "data":        updated clean_data dict (monthly key replaced),
            "log":         list of action strings,
            "summary":     text for narrative generation,
            "decisions":   list of decision dicts,
            "granularity": str,
            "n_periods":   int,
        }
        or None on failure.
    """
    log = []
    decisions = []

    # -----------------------------------------------------------------
    # MONTHLY -- pass through SecondFile.csv unchanged
    # -----------------------------------------------------------------
    if granularity == "monthly":
        n = len(clean_data.get("monthly", []))
        log.append(f"Monthly granularity: {n} months from SecondFile.csv")
        return {
            "data": clean_data, "log": log,
            "summary": f"Monthly granularity: {n} data points from SecondFile.csv.",
            "decisions": decisions, "granularity": "monthly", "n_periods": n,
        }

    # -----------------------------------------------------------------
    # WEEKLY -- aggregate transactions + distribute channel spend
    # -----------------------------------------------------------------
    if granularity == "weekly":
        logger.info("Building weekly dataset...")

        # Validate inputs
        if "transactions" not in clean_data:
            logger.error("transactions (firstfile.csv) not found")
            return None
        if "monthly" not in clean_data:
            logger.error("monthly (SecondFile.csv) not found -- needed for channel spend")
            return None

        tx = clean_data["transactions"].copy()
        monthly = clean_data["monthly"].copy()
        special_sales = clean_data.get("special_sales")
        nps_data = clean_data.get("nps")

        # Step 1: transactions -> weekly
        try:
            weekly = _aggregate_transactions_weekly(tx)
            log.append(f"Aggregated transactions to {len(weekly)} weeks")
            logger.info("  [OK] Transactions -> %d weeks", len(weekly))
        except Exception as e:
            logger.error("  Weekly aggregation failed: %s", e)
            return None

        # Step 2: distribute monthly channel spend
        try:
            weekly = _distribute_monthly_spend(weekly, monthly)
            log.append("Distributed monthly channel spend evenly across weeks")
            logger.info("  [OK] Channel spend distributed")
        except Exception as e:
            logger.error("  Spend distribution failed: %s", e)
            return None

        # Step 3: map sale events
        try:
            weekly = _map_sale_events_weekly(weekly, special_sales)
            log.append("Mapped sale events to weekly level")
            logger.info("  [OK] Sale events mapped")
        except Exception as e:
            logger.error("  Sale mapping failed: %s", e)
            weekly["sale_flag"] = 0
            weekly["sale_days"] = 0
            weekly["sale_intensity"] = 0

        # Step 4: add NPS
        try:
            weekly = _add_weekly_nps(weekly, monthly, nps_data)
            log.append("Added NPS (monthly repeated per week)")
            logger.info("  [OK] NPS added")
        except Exception as e:
            logger.error("  NPS addition failed: %s", e)

        # Step 5: derived columns
        try:
            weekly = _add_derived_columns(weekly)
            log.append("Added derived columns")
            logger.info("  [OK] Derived columns added")
        except Exception as e:
            logger.error("  Derived columns failed: %s", e)

        # Step 6: clean weekly data (outlier detection)
        try:
            from outlier_detection import clean_weekly
            weekly, weekly_log = clean_weekly(weekly)
            log.extend([entry["reason"] for entry in weekly_log])
        except Exception as e:
            logger.error("  Weekly cleaning failed: %s", e)

        n_periods = len(weekly)
        logger.info("  [SUMMARY] Weekly: %d weeks", n_periods)
        if "Date" in weekly.columns and n_periods > 0:
            logger.info("            Range: %s to %s",
                         weekly["Date"].min().strftime("%Y-%m-%d"),
                         weekly["Date"].max().strftime("%Y-%m-%d"))
            logger.info("            Avg weekly GMV: %.2f Cr",
                         weekly["total_gmv"].mean() / 1e7)

        # Replace monthly with weekly in clean_data
        updated = clean_data.copy()
        updated["monthly"] = weekly

        summary = (
            f"Weekly granularity: {n_periods} data points "
            f"(aggregated from {len(tx):,} transactions). "
            f"Channel spend distributed evenly from monthly totals. "
            f"Enables individual channel modeling and proper cross-validation."
        )

        decisions.append({
            "item": "weekly_aggregation",
            "action": (
                "Aggregated daily transactions to weekly. Monthly channel spend "
                "distributed evenly across weeks (assumes uniform intra-month spend)."
            ),
            "auto": True,
        })

        return {
            "data": updated, "log": log, "summary": summary,
            "decisions": decisions, "granularity": "weekly", "n_periods": n_periods,
        }

    # -----------------------------------------------------------------
    # UNKNOWN
    # -----------------------------------------------------------------
    logger.error("Unknown granularity: %s (use 'weekly' or 'monthly')", granularity)
    return None


# =============================================================================
# STEP 1: AGGREGATE TRANSACTIONS TO WEEKLY
# =============================================================================

def _aggregate_transactions_weekly(tx):
    """
    Aggregate daily transactions to weekly level.

    Creates: total_gmv, total_Units, total_Discount, total_Mrp,
             n_transactions, Revenue_<category>, Units_<category>.

    Returns:
        DataFrame with one row per week.
    """
    if not pd.api.types.is_datetime64_any_dtype(tx["Date"]):
        tx["Date"] = pd.to_datetime(tx["Date"])

    gmv_col = find_col(tx, ["gmv_new", "GMV", "gmv"])
    units_col = find_col(tx, ["units", "Units_sold", "Units"])
    discount_col = find_col(tx, ["discount", "Discount"])
    mrp_col = find_col(tx, ["product_mrp", "MRP", "mrp"])

    # Ensure numeric
    for col in [gmv_col, units_col, discount_col, mrp_col]:
        if col and col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce").fillna(0)

    # Week start (Monday)
    tx["week_start"] = tx["Date"].dt.to_period("W").apply(lambda x: x.start_time)

    # Core aggregation
    agg = {}
    if gmv_col:
        agg["total_gmv"] = (gmv_col, "sum")
    if units_col:
        agg["total_Units"] = (units_col, "sum")
    if discount_col:
        agg["total_Discount"] = (discount_col, "sum")
    if mrp_col:
        agg["total_Mrp"] = (mrp_col, "sum")
    agg["n_transactions"] = (gmv_col or tx.columns[0], "count")

    weekly = tx.groupby("week_start").agg(**agg).reset_index()
    weekly.rename(columns={"week_start": "Date"}, inplace=True)
    weekly = weekly.sort_values("Date").reset_index(drop=True)
    weekly["year_month"] = weekly["Date"].dt.to_period("M")

    # Category-level revenue
    cat_col = find_col(tx, ["product_category", "Product_Category"])
    if cat_col and gmv_col:
        try:
            cat_rev = (
                tx.pivot_table(index="week_start", columns=cat_col,
                               values=gmv_col, aggfunc="sum", fill_value=0)
                .reset_index()
                .rename(columns={"week_start": "Date"})
            )
            cat_rev.columns = [
                f"Revenue_{c}" if c != "Date" else c for c in cat_rev.columns
            ]
            cat_rev["Date"] = pd.to_datetime(cat_rev["Date"])
            weekly = weekly.merge(cat_rev, on="Date", how="left")
        except Exception as e:
            logger.warning("  Category revenue pivot failed: %s", e)

    # Category-level units
    if cat_col and units_col:
        try:
            cat_units = (
                tx.pivot_table(index="week_start", columns=cat_col,
                               values=units_col, aggfunc="sum", fill_value=0)
                .reset_index()
                .rename(columns={"week_start": "Date"})
            )
            cat_units.columns = [
                f"Units_{c}" if c != "Date" else c for c in cat_units.columns
            ]
            cat_units["Date"] = pd.to_datetime(cat_units["Date"])
            weekly = weekly.merge(cat_units, on="Date", how="left")
        except Exception as e:
            logger.warning("  Category units pivot failed: %s", e)

    return weekly


# =============================================================================
# STEP 2: DISTRIBUTE MONTHLY CHANNEL SPEND
# =============================================================================

def _distribute_monthly_spend(weekly_df, monthly_df):
    """
    Distribute monthly channel spend evenly across weeks in that month.
    Option A: even distribution (most defensible when weekly spend is unknown).

    Returns:
        weekly_df with channel columns added.
    """
    df = weekly_df.copy()
    m = monthly_df.copy()

    if "Date" in m.columns and not pd.api.types.is_datetime64_any_dtype(m["Date"]):
        m["Date"] = pd.to_datetime(m["Date"])
    m["year_month"] = m["Date"].dt.to_period("M")

    # Count weeks per month
    wpm = df.groupby("year_month").size().reset_index(name="n_weeks")
    m = m.merge(wpm, on="year_month", how="left")

    # Identify spend columns
    ch_present = [c for c in MEDIA_CHANNELS if c in m.columns]
    spend_cols = ch_present.copy()
    if "Total.Investment" in m.columns:
        spend_cols.append("Total.Investment")

    # Divide by weeks
    ws = m[["year_month", "n_weeks"] + [c for c in spend_cols if c in m.columns]].copy()
    for col in spend_cols:
        if col in ws.columns:
            ws[col] = pd.to_numeric(ws[col], errors="coerce").fillna(0) / ws["n_weeks"]
    ws.drop(columns=["n_weeks"], inplace=True)

    # Merge into weekly
    df = df.merge(ws, on="year_month", how="left")
    for col in spend_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# =============================================================================
# STEP 3: MAP SALE EVENTS TO WEEKS
# =============================================================================

def _map_sale_events_weekly(weekly_df, special_sales_df):
    """
    Map sale events to weekly level: sale_flag, sale_days, sale_intensity.

    Returns:
        weekly_df with sale columns added.
    """
    df = weekly_df.copy()

    if special_sales_df is None:
        df["sale_flag"] = 0
        df["sale_days"] = 0
        df["sale_intensity"] = 0
        return df

    ss = special_sales_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(ss["Date"]):
        ss["Date"] = pd.to_datetime(ss["Date"])

    name_col = find_col(ss, ["Sales Name", "Sales_Name", "Sales_name"])
    ss["week_start"] = ss["Date"].dt.to_period("W").apply(lambda x: x.start_time)

    # Sale days per week
    sd = ss.groupby("week_start")["Date"].nunique().reset_index()
    sd.columns = ["Date", "sale_days"]

    # Distinct events per week
    if name_col:
        si = ss.groupby("week_start")[name_col].nunique().reset_index()
        si.columns = ["Date", "sale_intensity"]
    else:
        si = sd.rename(columns={"sale_days": "sale_intensity"})

    df = df.merge(sd, on="Date", how="left")
    df = df.merge(si, on="Date", how="left")
    df["sale_days"] = df["sale_days"].fillna(0).astype(int)
    df["sale_intensity"] = df["sale_intensity"].fillna(0).astype(int)
    df["sale_flag"] = (df["sale_days"] > 0).astype(int)

    return df


# =============================================================================
# STEP 4: ADD NPS
# =============================================================================

def _add_weekly_nps(weekly_df, monthly_df, nps_data=None):
    """
    Repeat monthly NPS value for each week in that month.

    Returns:
        weekly_df with NPS column.
    """
    df = weekly_df.copy()
    m = monthly_df.copy()

    if "Date" in m.columns and not pd.api.types.is_datetime64_any_dtype(m["Date"]):
        m["Date"] = pd.to_datetime(m["Date"])

    if "NPS" in m.columns:
        m["year_month"] = m["Date"].dt.to_period("M")
        nps_map = m[["year_month", "NPS"]].copy()
        df = df.merge(nps_map, on="year_month", how="left")
    elif nps_data is not None and "NPS" in nps_data.columns:
        nps = nps_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(nps["Date"]):
            nps["Date"] = pd.to_datetime(nps["Date"])
        nps["year_month"] = nps["Date"].dt.to_period("M")
        df = df.merge(nps[["year_month", "NPS"]], on="year_month", how="left")

    return df


# =============================================================================
# STEP 5: DERIVED COLUMNS
# =============================================================================

def _add_derived_columns(weekly_df):
    """
    Add columns that downstream modules expect:
    Total.Investment (if missing), Year, Month, month label.

    Returns:
        weekly_df with derived columns.
    """
    df = weekly_df.copy()

    ch = get_channel_cols(df)
    if "Total.Investment" not in df.columns and ch:
        df["Total.Investment"] = df[ch].sum(axis=1)

    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["month"] = df["Date"].dt.strftime("%b-%y")

    return df


# =============================================================================
# RUN STANDALONE
# =============================================================================

if __name__ == "__main__":
    from eda_pipeline import load_all_data
    from outlier_detection import run_outlier_pipeline

    data = load_all_data()
    clean_data, _, _ = run_outlier_pipeline(data)

    # Test weekly
    result = build_modeling_dataset(clean_data, granularity="weekly")
    if result:
        w = result["data"]["monthly"]
        print(f"\nWeekly dataset: {w.shape}")
        print(f"Columns: {w.columns.tolist()}")
        print(w.head().to_string())

    # Test monthly
    result_m = build_modeling_dataset(clean_data, granularity="monthly")
    if result_m:
        print(f"\nMonthly: {result_m['n_periods']} periods")
