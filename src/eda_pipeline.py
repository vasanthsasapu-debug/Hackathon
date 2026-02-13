"""
=============================================================================
EDA PIPELINE -- E-Commerce Marketing Mix (MMIX)
=============================================================================
Modules:
  1. Data Loading & Auto-Classification
  2. Data Validation
  3. Data Quality Assessment
  4. National-Level Trends
  5. Category-Level Breakdown
  6. Media Investment Analysis
  7. Correlation Analysis (Media vs Revenue)
  8. Special Sale Impact Analysis
  9. Channel Overlap Analysis (Traditional vs Digital)
  10. NPS vs Revenue Analysis
  11. Summary & Key Findings
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Plotting defaults
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# =============================================================================
# CONSTANTS
# =============================================================================
MEDIA_CHANNELS = ['TV', 'Digital', 'Sponsorship', 'Content.Marketing',
                  'Online.marketing', 'Affiliates', 'SEM', 'Radio', 'Other']
CRORE = 1e7
THOUSAND = 1e3
LAKH = 1e5


# =============================================================================
# 1. DATA LOADING & AUTO-CLASSIFICATION
# =============================================================================

def load_all_data(data_dir=None):
    """Load all datasets from the data directory. Returns dict of DataFrames."""

    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
    data = {}
    file_map = {
        'transactions': ('firstfile.csv', ['Date'], ','),
        'sales':        ('Sales.csv', ['Date'], '\t'),
        'monthly':      ('SecondFile.csv', ['Date'], ','),
        'special_sales':('SpecialSale.csv', ['Date'], ','),
        'investment':   ('MediaInvestment.csv', None, ','),
        'nps':          ('MonthlyNPSscore.csv', ['Date'], ','),
        'products':     ('ProductList.csv', None, ','),
    }

    for key, (fname, date_cols, delimiter) in file_map.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] File not found: {fpath} -- skipping '{key}'")
            continue
        try:
            if key == 'sales':
                data[key] = pd.read_csv(fpath, sep=delimiter, parse_dates=date_cols,
                                        date_format='%d-%m-%Y %H:%M')
                # GMV is object dtype -- coerce to numeric
                if 'GMV' in data[key].columns:
                    data[key]['GMV'] = pd.to_numeric(
                        data[key]['GMV'].astype(str).str.strip(), errors='coerce'
                    )
            elif key == 'investment':
                data[key] = pd.read_csv(fpath, sep=delimiter)
                # Fix leading space in ' Affiliates'
                data[key].columns = data[key].columns.str.strip()
            elif key == 'products':
                data[key] = pd.read_csv(fpath, sep=delimiter)
                # Frequency is object (has commas)
                if 'Frequency' in data[key].columns:
                    data[key]['Frequency'] = pd.to_numeric(
                        data[key]['Frequency'].astype(str).str.replace(',', '').str.strip(),
                        errors='coerce'
                    )
            elif date_cols:
                data[key] = pd.read_csv(fpath, sep=delimiter, parse_dates=date_cols)
            else:
                data[key] = pd.read_csv(fpath, sep=delimiter)

            # Drop index columns
            if 'Unnamed: 0' in data[key].columns:
                data[key] = data[key].drop('Unnamed: 0', axis=1)

            print(f"  [OK] {key:20s} -> {data[key].shape[0]:>8,} rows x {data[key].shape[1]:>3} cols  ({fname})")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {e}")

    return data


def auto_classify_columns(df, dataset_name=""):
    """
    Automatically classify columns by their likely role in MMIX analysis.
    Returns a dict: {role: [col_names]}
    """
    classification = {
        'id': [], 'date': [], 'metric_revenue': [], 'metric_units': [],
        'metric_price': [], 'metric_discount': [], 'metric_investment': [],
        'metric_score': [], 'dimension_category': [], 'dimension_promotion': [],
        'other': []
    }

    for col in df.columns:
        cl = col.lower().strip()
        dtype = df[col].dtype

        if 'date' in cl or dtype == 'datetime64[ns]':
            classification['date'].append(col)
        elif 'id' in cl:
            classification['id'].append(col)
        elif any(k in cl for k in ['gmv', 'revenue']):
            classification['metric_revenue'].append(col)
        elif 'sales' in cl and 'name' in cl:
            classification['dimension_promotion'].append(col)
        elif 'unit' in cl:
            classification['metric_units'].append(col)
        elif 'mrp' in cl or 'price' in cl:
            classification['metric_price'].append(col)
        elif 'discount' in cl:
            classification['metric_discount'].append(col)
        elif 'nps' in cl:
            classification['metric_score'].append(col)
        elif any(k in cl for k in ['tv', 'digital', 'sponsor', 'content',
                                     'online', 'affiliate', 'sem', 'radio',
                                     'other', 'investment']):
            classification['metric_investment'].append(col)
        elif any(k in cl for k in ['category', 'subcategory', 'vertical',
                                     'analytic', 'product', 'sub_category']):
            classification['dimension_category'].append(col)
        elif cl in ['year', 'month']:
            classification['date'].append(col)
        else:
            classification['other'].append(col)

    classification = {k: v for k, v in classification.items() if v}

    print(f"\n  Auto-Classification for '{dataset_name}':")
    for role, cols in classification.items():
        print(f"    {role:25s} -> {cols}")

    return classification


# =============================================================================
# 2. DATA VALIDATION
# =============================================================================

def validate_mmix_data(data):
    """Validate all datasets for common issues. Returns list of issues."""
    issues = []
    print("\n[VALIDATE] Checking data quality...")

    if 'monthly' in data:
        m = data['monthly']

        # Required columns
        required = {
            'revenue': ['total_gmv'],
            'units': ['total_Units'],
            'investment': ['Total.Investment'],
            'date': ['Date'],
            'score': ['NPS']
        }
        for group, cols in required.items():
            for c in cols:
                if c not in m.columns:
                    msg = f"Monthly: Missing critical column '{c}' ({group})"
                    issues.append(msg)
                    print(f"  [ERROR] {msg}")

        # Duplicate dates
        if 'Date' in m.columns:
            dups = m['Date'].duplicated().sum()
            if dups > 0:
                msg = f"Monthly: {dups} duplicate dates found"
                issues.append(msg)
                print(f"  [WARN] {msg}")
            else:
                print("  [OK] No duplicate dates in monthly data")

        # Negative values
        numeric_cols = m.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            neg_count = (m[c] < 0).sum()
            if neg_count > 0:
                msg = f"Monthly: {neg_count} negative values in '{c}'"
                issues.append(msg)
                print(f"  [WARN] {msg}")

        # Sample size
        n_months = len(m)
        if n_months < 24:
            msg = (f"Monthly: Only {n_months} months of data. "
                   f"MMIX typically needs 60+. Model complexity will be limited.")
            issues.append(msg)
            print(f"  [WARN] {msg}")
        print(f"  [INFO] Sample size: {n_months} months")

        # Missing values in channels
        inv_cols = _get_channel_cols(m)
        for c in inv_cols:
            nulls = pd.to_numeric(m[c], errors='coerce').isnull().sum()
            if nulls > 0:
                msg = f"Monthly: {nulls} missing values in channel '{c}'"
                issues.append(msg)
                print(f"  [WARN] {msg}")

        # Anomaly check
        if 'Date' in m.columns and 'total_gmv' in m.columns:
            median_gmv = m['total_gmv'].median()
            low_months = m[m['total_gmv'] < median_gmv * 0.05]
            if len(low_months) > 0:
                dates = low_months['Date'].dt.strftime('%b-%Y').tolist()
                msg = f"Monthly: Anomalously low GMV in: {dates}"
                issues.append(msg)
                print(f"  [WARN] {msg}")

    if 'transactions' in data:
        tx = data['transactions']
        gmv_col = _find_col(tx, ['gmv_new', 'GMV', 'gmv'])
        if gmv_col:
            neg = (pd.to_numeric(tx[gmv_col], errors='coerce') < 0).sum()
            if neg > 0:
                msg = f"Transactions: {neg} negative GMV values"
                issues.append(msg)
                print(f"  [WARN] {msg}")

    if 'special_sales' in data:
        ss = data['special_sales']
        for c in ['Date', 'Sales Name']:
            if c not in ss.columns:
                alt = [col for col in ss.columns if c.lower().replace(' ', '') in col.lower().replace(' ', '')]
                if alt:
                    print(f"  [INFO] SpecialSale: '{c}' not found, using '{alt[0]}'")
                else:
                    msg = f"SpecialSale: Missing column '{c}'"
                    issues.append(msg)
                    print(f"  [ERROR] {msg}")

    if not issues:
        print("  [OK] All validations passed!")
    else:
        print(f"\n  Total issues found: {len(issues)}")

    return issues


def _get_channel_cols(df):
    """Helper: Get media channel columns that exist in the dataframe."""
    return [c for c in MEDIA_CHANNELS if c in df.columns]


def _find_col(df, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_get_col(df, col_name, fallback=None):
    """Safely get a column, returning fallback if not found."""
    if col_name in df.columns:
        return df[col_name]
    if fallback and fallback in df.columns:
        return df[fallback]
    return None


# =============================================================================
# 3. DATA QUALITY ASSESSMENT
# =============================================================================

def data_quality_report(df, name=""):
    """Generate a data quality summary for a dataframe."""
    print(f"\n{'='*60}")
    print(f"  DATA QUALITY REPORT: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    quality = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
    })
    print(quality.to_string())

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"\n  Numeric Summary:")
        print(df[num_cols].describe().round(2).to_string())

    return quality


# =============================================================================
# 4. NATIONAL-LEVEL TRENDS
# =============================================================================

def national_trends(monthly_df, save_dir=None):
    """Plot national-level GMV, Units, Discount trends over time."""
    if save_dir is None:
        save_dir = _default_save_dir()

    df = monthly_df.copy()
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('National-Level Trends', fontsize=16, fontweight='bold')

    if 'total_gmv' in df.columns:
        axes[0, 0].plot(df['Date'], df['total_gmv'] / CRORE, marker='o', color='#2196F3', linewidth=2)
        axes[0, 0].set_title('Total GMV (Cr)')
        axes[0, 0].set_ylabel('GMV (Cr)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    if 'total_Units' in df.columns:
        axes[0, 1].plot(df['Date'], df['total_Units'] / THOUSAND, marker='s', color='#4CAF50', linewidth=2)
        axes[0, 1].set_title('Total Units Sold (K)')
        axes[0, 1].set_ylabel('Units (Thousands)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    if 'total_Discount' in df.columns:
        axes[1, 0].plot(df['Date'], df['total_Discount'] / CRORE, marker='^', color='#FF5722', linewidth=2)
        axes[1, 0].set_title('Total Discount (Cr)')
        axes[1, 0].set_ylabel('Discount (Cr)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    if 'total_Discount' in df.columns and 'total_Mrp' in df.columns:
        df['discount_pct'] = (df['total_Discount'] / df['total_Mrp'] * 100)
        axes[1, 1].bar(df['Date'], df['discount_pct'], color='#FF9800', width=20)
        axes[1, 1].set_title('Discount as % of MRP')
        axes[1, 1].set_ylabel('Discount %')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'national_trends.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("[OK] National trends plotted.\n")


# =============================================================================
# 5. CATEGORY-LEVEL BREAKDOWN
# =============================================================================

def category_breakdown(monthly_df, save_dir=None):
    """Analyze revenue and units by product category over time."""
    if save_dir is None:
        save_dir = _default_save_dir()

    df = monthly_df.copy()
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    rev_cols = [c for c in df.columns if c.startswith('Revenue_')]
    unit_cols = [c for c in df.columns if c.startswith('Units_') and c != 'total_Units']

    if not rev_cols:
        print("[WARN] No Revenue_ columns found. Skipping category breakdown.")
        return None

    categories = [c.replace('Revenue_', '') for c in rev_cols]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Category-Level Analysis', fontsize=16, fontweight='bold')

    for col, cat in zip(rev_cols, categories):
        axes[0].plot(df['Date'], df[col] / CRORE, marker='o', label=cat, linewidth=2)
    axes[0].set_title('Revenue by Category (Cr)')
    axes[0].set_ylabel('Revenue (Cr)')
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis='x', rotation=45)

    latest = df.iloc[-1]
    rev_vals = [latest[c] for c in rev_cols if pd.notna(latest[c])]
    valid_cats = [cat for c, cat in zip(rev_cols, categories) if pd.notna(latest[c])]
    axes[1].pie(rev_vals, labels=valid_cats, autopct='%1.1f%%', startangle=140,
                colors=sns.color_palette("Set2", len(valid_cats)))
    axes[1].set_title(f'Revenue Share ({latest["Date"].strftime("%b-%Y")})')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'category_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.show()

    summary = pd.DataFrame({
        'Category': categories,
        'Avg Monthly Revenue (Cr)': [df[c].mean() / CRORE for c in rev_cols],
        'Avg Monthly Units': [df[c].mean() for c in unit_cols] if len(unit_cols) == len(rev_cols) else [None]*len(rev_cols),
        'Revenue Std Dev (Cr)': [df[c].std() / CRORE for c in rev_cols],
    }).round(2)
    print("\n  Category Summary:")
    print(summary.to_string(index=False))

    return summary


# =============================================================================
# 6. MEDIA INVESTMENT ANALYSIS
# =============================================================================

def media_investment_analysis(monthly_df, save_dir=None):
    """Analyze media investment trends and share of spend."""
    if save_dir is None:
        save_dir = _default_save_dir()

    df = monthly_df.copy()
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    channels = _get_channel_cols(df)
    if not channels:
        print("[WARN] No media channel columns found. Skipping.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Media Investment Analysis', fontsize=16, fontweight='bold')

    for ch in channels:
        vals = pd.to_numeric(df[ch], errors='coerce') / CRORE
        axes[0].plot(df['Date'], vals, marker='o', label=ch, linewidth=2)
    axes[0].set_title('Channel Spend Over Time (Cr)')
    axes[0].set_ylabel('Spend (Cr)')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='x', rotation=45)

    avg_spend = {}
    for ch in channels:
        val = pd.to_numeric(df[ch], errors='coerce').mean()
        if not np.isnan(val):
            avg_spend[ch] = val

    axes[1].barh(list(avg_spend.keys()),
                 [v / CRORE for v in avg_spend.values()],
                 color=sns.color_palette("viridis", len(avg_spend)))
    axes[1].set_title('Average Monthly Spend by Channel (Cr)')
    axes[1].set_xlabel('Avg Spend (Cr)')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'media_investment.png'), dpi=150, bbox_inches='tight')
    plt.show()

    total = sum(avg_spend.values())
    share_df = pd.DataFrame({
        'Channel': list(avg_spend.keys()),
        'Avg Spend (Cr)': [f"{v/CRORE:.2f}" for v in avg_spend.values()],
        'Share %': [f"{v/total*100:.1f}" for v in avg_spend.values()]
    })
    print("\n  Share of Spend:")
    print(share_df.to_string(index=False))

    return share_df


# =============================================================================
# 7. CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(monthly_df, save_dir=None):
    """Compute and visualize correlations between media channels and revenue."""
    if save_dir is None:
        save_dir = _default_save_dir()

    df = monthly_df.copy()
    channels = _get_channel_cols(df)
    target = 'total_gmv'
    extra_vars = ['total_Discount', 'NPS', 'Total.Investment']

    if target not in df.columns:
        print(f"[WARN] Target column '{target}' not found. Skipping correlation.")
        return None

    all_vars = [target] + channels + [v for v in extra_vars if v in df.columns]
    corr_df = df[all_vars].apply(pd.to_numeric, errors='coerce')
    corr_matrix = corr_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title('Full Correlation Matrix')

    if channels:
        ch_corr = corr_matrix.loc[channels, target].sort_values(ascending=True)
        colors = ['#4CAF50' if v > 0 else '#F44336' for v in ch_corr.values]
        axes[1].barh(ch_corr.index, ch_corr.values, color=colors)
        axes[1].set_title(f'Channel Correlation with {target}')
        axes[1].axvline(x=0, color='black', linewidth=0.8)
        axes[1].set_xlim(-1, 1)

        print(f"\n  Channel Correlations with {target}:")
        for ch, val in ch_corr.items():
            strength = "strong" if abs(val) > 0.6 else "moderate" if abs(val) > 0.3 else "weak"
            direction = "positive" if val > 0 else "negative"
            print(f"    {ch:25s} -> {val:+.3f} ({strength}, {direction})")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'correlation_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return corr_matrix


# =============================================================================
# 8. SPECIAL SALE IMPACT ANALYSIS
# =============================================================================

def special_sale_impact(transactions_df, special_sales_df, save_dir=None):
    """Analyze the impact of special sales on daily GMV."""
    if save_dir is None:
        save_dir = _default_save_dir()

    tx = transactions_df.copy()
    ss = special_sales_df.copy()

    if 'Date' in tx.columns and not pd.api.types.is_datetime64_any_dtype(tx['Date']):
        tx['Date'] = pd.to_datetime(tx['Date'])
    if 'Date' in ss.columns and not pd.api.types.is_datetime64_any_dtype(ss['Date']):
        ss['Date'] = pd.to_datetime(ss['Date'])

    sale_name_col = _find_col(ss, ['Sales Name', 'Sales_Name', 'Sales_name', 'sales_name'])
    gmv_col = _find_col(tx, ['gmv_new', 'GMV', 'gmv'])

    if not sale_name_col:
        print("[WARN] Cannot find sale name column in SpecialSale. Skipping.")
        return
    if not gmv_col:
        print("[WARN] Cannot find GMV column in transactions. Skipping.")
        return

    sale_dates = set(ss['Date'].dt.date)
    tx['is_sale'] = tx['Date'].dt.date.apply(lambda d: d in sale_dates)

    daily = tx.groupby(['Date', 'is_sale']).agg(
        daily_gmv=(gmv_col, 'sum'),
        daily_units=('units', 'sum'),
    ).reset_index()

    sale_gmv = daily[daily['is_sale']]['daily_gmv'].values
    nonsale_gmv = daily[~daily['is_sale']]['daily_gmv'].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Special Sale Impact', fontsize=16, fontweight='bold')

    box_data = [nonsale_gmv / LAKH]
    box_labels = ['Non-Sale Days']
    if len(sale_gmv) > 0:
        box_data.append(sale_gmv / LAKH)
        box_labels.append('Sale Days')
    axes[0].boxplot(box_data, labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor='#E3F2FD'))
    axes[0].set_title('Daily GMV Distribution (Lakhs)')
    axes[0].set_ylabel('GMV (Lakhs)')

    ss_summary = ss.groupby(sale_name_col).size().reset_index(name='Days')
    axes[1].barh(ss_summary[sale_name_col], ss_summary['Days'],
                 color=sns.color_palette("Set2", len(ss_summary)))
    axes[1].set_title('Sale Event Duration (Days)')
    axes[1].set_xlabel('Number of Days')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'special_sale_impact.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  Sale vs Non-Sale Summary:")
    print(f"    Non-Sale Days -> Avg GMV: {nonsale_gmv.mean()/LAKH:.1f}L, "
          f"Median: {np.median(nonsale_gmv)/LAKH:.1f}L")
    if len(sale_gmv) > 0:
        print(f"    Sale Days     -> Avg GMV: {sale_gmv.mean()/LAKH:.1f}L, "
              f"Median: {np.median(sale_gmv)/LAKH:.1f}L")
        lift = (sale_gmv.mean() / nonsale_gmv.mean() - 1) * 100
        print(f"    Sale Lift: {lift:+.1f}%")


# =============================================================================
# 9. CHANNEL OVERLAP ANALYSIS
# =============================================================================

def channel_overlap_analysis(monthly_df, save_dir=None):
    """Compare traditional vs digital channel spend and relationship with revenue."""
    if save_dir is None:
        save_dir = _default_save_dir()

    df = monthly_df.copy()
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    traditional = ['TV', 'Radio', 'Sponsorship']
    digital = ['Digital', 'Online.marketing', 'Affiliates', 'SEM', 'Content.Marketing']

    trad_cols = [c for c in traditional if c in df.columns]
    digi_cols = [c for c in digital if c in df.columns]

    if not trad_cols and not digi_cols:
        print("[WARN] No channel columns found. Skipping overlap analysis.")
        return

    df['traditional_spend'] = df[trad_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['digital_spend'] = df[digi_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    df['total_spend'] = df['traditional_spend'] + df['digital_spend']
    df['digital_share'] = np.where(
        df['total_spend'] > 0,
        df['digital_spend'] / df['total_spend'] * 100, 0
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Traditional vs Digital Channel Analysis', fontsize=16, fontweight='bold')

    axes[0].bar(df['Date'], df['traditional_spend'] / CRORE, label='Traditional', color='#3F51B5')
    axes[0].bar(df['Date'], df['digital_spend'] / CRORE,
                bottom=df['traditional_spend'] / CRORE, label='Digital', color='#00BCD4')
    axes[0].set_title('Spend: Traditional vs Digital (Cr)')
    axes[0].set_ylabel('Spend (Cr)')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot(df['Date'], df['digital_share'], marker='o', color='#00BCD4', linewidth=2)
    axes[1].set_title('Digital Share of Total Spend (%)')
    axes[1].set_ylabel('Digital Share %')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    if 'total_gmv' in df.columns:
        axes[2].scatter(df['traditional_spend'] / CRORE, df['total_gmv'] / CRORE,
                        label='Traditional', s=100, color='#3F51B5', edgecolors='white')
        axes[2].scatter(df['digital_spend'] / CRORE, df['total_gmv'] / CRORE,
                        label='Digital', s=100, color='#00BCD4', edgecolors='white')
        axes[2].set_title('Spend vs Revenue')
        axes[2].set_xlabel('Spend (Cr)')
        axes[2].set_ylabel('GMV (Cr)')
        axes[2].legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'channel_overlap.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n  Traditional vs Digital Summary:")
    print(f"    Avg Traditional Spend: {df['traditional_spend'].mean()/CRORE:.1f} Cr/month")
    print(f"    Avg Digital Spend:     {df['digital_spend'].mean()/CRORE:.1f} Cr/month")
    print(f"    Avg Digital Share:     {df['digital_share'].mean():.1f}%")


# =============================================================================
# 10. NPS vs REVENUE ANALYSIS
# =============================================================================

def nps_revenue_analysis(monthly_df, save_dir=None):
    """Analyze relationship between NPS score and revenue."""
    if save_dir is None:
        save_dir = _default_save_dir()

    df = monthly_df.copy()
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if 'NPS' not in df.columns or 'total_gmv' not in df.columns:
        print("[WARN] NPS or total_gmv column missing. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NPS vs Revenue Analysis', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax1.plot(df['Date'], df['total_gmv'] / CRORE, 'b-o', label='GMV', linewidth=2)
    ax2.plot(df['Date'], df['NPS'], 'r-s', label='NPS', linewidth=2)
    ax1.set_ylabel('GMV (Cr)', color='blue')
    ax2.set_ylabel('NPS Score', color='red')
    ax1.set_title('GMV & NPS Trend')
    ax1.tick_params(axis='x', rotation=45)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='upper left')

    nps_clean = df[['NPS', 'total_gmv']].dropna()
    axes[1].scatter(nps_clean['NPS'], nps_clean['total_gmv'] / CRORE, s=120, c='#9C27B0', edgecolors='white')
    if len(nps_clean) > 2:
        z = np.polyfit(nps_clean['NPS'], nps_clean['total_gmv'] / CRORE, 1)
        p = np.poly1d(z)
        x_line = np.linspace(nps_clean['NPS'].min(), nps_clean['NPS'].max(), 50)
        axes[1].plot(x_line, p(x_line), 'r--', alpha=0.7)
    axes[1].set_title('NPS vs GMV')
    axes[1].set_xlabel('NPS Score')
    axes[1].set_ylabel('GMV (Cr)')

    corr = df['NPS'].corr(df['total_gmv'])
    print(f"\n  NPS-Revenue Correlation: {corr:.3f}")
    direction = "Higher NPS -> Higher Revenue" if corr > 0 else "Higher NPS -> Lower Revenue"
    print(f"    Direction: {direction}")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'nps_revenue.png'), dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# HELPERS
# =============================================================================

def _default_save_dir():
    """Auto-detect default save directory from project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'outputs', 'plots')


# =============================================================================
# 11. MASTER EDA RUNNER
# =============================================================================

def run_full_eda(data_dir=None, save_dir=None):
    """Run the complete EDA pipeline end-to-end."""
    if save_dir is None:
        save_dir = _default_save_dir()

    print("=" * 60)
    print("[START] EDA PIPELINE -- E-Commerce MMIX")
    print("=" * 60)

    # Step 1: Load
    print("\n[STEP 1] Loading Data...")
    data = load_all_data(data_dir)
    if not data:
        print("[ERROR] No data loaded.")
        return None, None, None, None

    # Step 2: Classify
    print("\n[STEP 2] Auto-Classifying Columns...")
    classifications = {}
    for name, df in data.items():
        classifications[name] = auto_classify_columns(df, name)

    # Step 3: Validate
    print("\n[STEP 3] Validating Data...")
    issues = validate_mmix_data(data)

    # Step 4: Quality
    print("\n[STEP 4] Data Quality Assessment...")
    for name, df in data.items():
        data_quality_report(df, name)

    corr_matrix = None
    if 'monthly' in data:
        print("\n[STEP 5] National-Level Trends...")
        national_trends(data['monthly'], save_dir)

        print("\n[STEP 6] Category-Level Breakdown...")
        category_breakdown(data['monthly'], save_dir)

        print("\n[STEP 7] Media Investment Analysis...")
        media_investment_analysis(data['monthly'], save_dir)

        print("\n[STEP 8] Correlation Analysis...")
        corr_matrix = correlation_analysis(data['monthly'], save_dir)

    if 'transactions' in data and 'special_sales' in data:
        print("\n[STEP 9] Special Sale Impact...")
        special_sale_impact(data['transactions'], data['special_sales'], save_dir)

    if 'monthly' in data:
        print("\n[STEP 10] Channel Overlap Analysis...")
        channel_overlap_analysis(data['monthly'], save_dir)

        print("\n[STEP 11] NPS vs Revenue...")
        nps_revenue_analysis(data['monthly'], save_dir)

    print("\n" + "=" * 60)
    print("[DONE] EDA PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Plots saved to: {save_dir}/")

    return data, classifications, issues, corr_matrix


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    data, classifications, issues, corr_matrix = run_full_eda()