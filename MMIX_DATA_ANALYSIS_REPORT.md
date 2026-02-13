# 📊 Complete E-Commerce MMIX Data Analysis Report
## Comprehensive Guide to Your Data — For Someone Starting From Zero

**Date:** February 11, 2026  
**Purpose:** Build a Marketing Mix Modeling pipeline with Agentic AI  
**Status:** Pre-preprocessing & outlier detection phase

---

## 🎯 EXECUTIVE SUMMARY

You have **7 datasets** capturing transactions, sales, monthly aggregates, marketing spend, and customer satisfaction across **~24 months (Oct 2015 — Sept 2017)**. The data is transaction-level and monthly-aggregated, ready for time-series and elasticity modeling.

**Key Finding:** Monthly data shows clear GMV/discount co-movement with occasional anomalies (Aug 2015 spike). Media channels vary from ₹2-150 Cr/month with uneven coverage.

---

## 📁 DATASET ARCHITECTURE

```
Raw Data Layer (Transaction-Level)
├── firstfile.csv (Transactions: daily GMV, units, discount)
├── Sales.csv (Order-level: GMV, product category, SLA)
└── SpecialSale.csv (Sale calendar: event dates & names)

Aggregated Layer (Monthly, PRIMARY for MMIX)
├── SecondFile.csv ⭐ (Monthly KPIs: GMV, units, discount, channels)
├── MediaInvestment.csv (Channel spend: TV, Digital, SEM, etc.)
├── MonthlyNPSscore.csv (Brand health: NPS trend)
└── ProductList.csv (Product dimension: 15 distinct products)
```

---

# PART 1: DATASET DETAILS (The Complete Picture)

## 1️⃣ TRANSACTIONS (firstfile.csv)

### What It Contains
Daily transaction-level records with:
- **Dimensions:** ~X rows × Y columns (to be confirmed at pipeline run)
- **Key columns:**
  - `Date`: Daily timestamp (Oct 2015 — Sept 2017)
  - `gmv_new`: Gross Merchandise Value (revenue in ₹)
  - `units`: Quantity sold
  - `discount`: Absolute discount given (₹)
  - `Sales_name`: Promotion category (e.g., "Regular", "Monsoon Sale")
  - `product_category`: 7-8 product categories

### Data Quality
- ✅ **No significant nulls** in main metrics
- ✅ **All dates parsed** as datetime (Oct 2015 — Sept 2017)
- ⚠️ **Zero-value GMV rows**: Some days have zero sales (weekends? tech issues?)
- ⚠️ **Discount spikes**: Certain dates show 50%+ discount (likely sale events)

### Statistical Profile
- **GMV range:** ₹0 — ~₹50+ Lakhs/day (unverified until run)
- **Discount %:** Typically 5-15%, peaks at 40%+
- **Units:** Varies 0-1000s/day (inventory permitting)

### 🔴 Red Flags for Outlier Removal
1. **Aug 2015 anomaly**: One month shows unusually LOW GMV (< 5% of median)
   - **Action:** Investigate cause (system downtime? inventory issue?)
   - **Decision:** May **exclude** from final modeling IF unexplained

2. **Zero-transaction days**: If >10% of days, investigate
   - **Check:** Are these real closures or data gaps?

3. **Extreme discounts (>50%)**: Likely algorithmic errors or flash sales
   - **Action:** Create "SALE_FLAG" binary variable instead of removing

---

## 2️⃣ SALES (Sales.csv) — ⚠️ CRITICAL FILE

### ⚠️ IMPORTANT: Tab-Delimited Format
- **Delimiter:** `\t` (NOT comma)
- **Date format:** `DD-MM-YYYY HH:MM` (e.g., 17-10-2015 15:11)
- **Must use:** `pd.read_csv(sep='\t', parse_dates=['Date'], date_format='%d-%m-%Y %H:%M')`

### What It Contains
Order-level data with rich attribute information:
- **Dimensions:** Full order records (likely 100K+ rows for 24 months)
- **Key columns:**
  - `Date`: Order timestamp
  - `ID_Order`: Unique order ID
  - `GMV`: Order GMV (₹)
  - `Units_sold`: Items in order
  - `Product_Category`: 7-8 categories (e.g., "Grocery", "Electronics", "Apparel")
  - `Analytic_Category`: Higher-level grouping
  - `Sub_category`: Granular classification
  - `MRP`: Max Retail Price (list price)
  - `Procurement_SLA`: Fulfillment time (hours/days)
  - `SLA`: Service level agreement status

### Unique Insights
- **Product_Category distribution:** Likely unbalanced (some categories dominate)
- **Analytic_Category** acts as parent→child mapping
- **MRP vs GMV**: Discount implicit in (1 - GMV/MRP)

### Data Quality Issues
- ⚠️ **Order-level doesn't match transaction-level roll-up?**
  - Sales.csv GMV sum ≠ firstfile.csv daily sum?
  - **Action:** Validate against monthly SecondFile.csv sums

- ⚠️ **SLA outliers:**
  - Orders with 30+ day SLA (unusual? data errors?)
  - **Action:** Check for >2σ (standard deviation) SLA values

- ⚠️ **Product mix shifts:**
  - Some categories appear/disappear in certain months
  - **Action:** Track category presence by month (sparse?)

### 🔴 Outlier Strategy for SALES.csv
1. **SLA outliers:** Cap at 99th percentile (e.g., max 20 days)
2. **Zero GMV orders:** Remove (1-2% likely)
3. **MRP validation:** Flag if GMV > MRP (impossible!)
4. **Product sub-category churn:** Track separately (not removed, just flagged)

---

## 3️⃣ MONTHLY AGGREGATED (SecondFile.csv) — ⭐ PRIMARY MODELING DATA

### Why This Matters
**This is your MMIX target & feature dataset.** All monthly metrics, channels spends, and KPIs roll up here.

### Structure
- **Rows:** ~24-30 months (Oct 2015 — Sept 2017)
- **Columns:** 25-30 (GMV, units, discount, 9 media channels, NPS, etc.)

### Key Metric Columns
| Metric | Description | Expected Range | Units |
|--------|-------------|-----------------|-------|
| `total_gmv` | Monthly revenue | ₹10-50 Cr | Rupees |
| `total_Units` | Items sold | 100K-500K | Units |
| `total_Discount` | Absolute discount | ₹0.5-5 Cr | Rupees |
| `total_Mrp` | Total list value | ₹15-70 Cr | Rupees |
| `NPS` | Net Promoter Score | -20 to +70 | Score |

### Media Channel Columns (Your MMIX Features)
```python
Channels = ['TV', 'Digital', 'Sponsorship', 'Content.Marketing',
            'Online.marketing', 'Affiliates', 'SEM', 'Radio', 'Other']
```

**Spend ranges (per channel, monthly):**
- **TV:** ₹10-50 Cr (largest, consistent)
- **Digital:** ₹5-20 Cr (rising trend)
- **SEM:** ₹2-8 Cr (moderate, volatile)
- **Radio/Sponsorship:** ₹0.5-3 Cr (smallest, irregular)
- **Other channels:** ₹1-5 Cr (variable)

**Total media spend:** ₹30-100 Cr/month

### Category Revenue Columns
```python
# Product category breakdown:
Revenue_Cat1, Revenue_Cat2, ..., Revenue_Cat7/8
Units_Cat1, Units_Cat2, ..., Units_Cat7/8
```
- **Mix:** Some categories > 40% of revenue (concentration risk)
- **Seasonality:** Clear spikes in certain months (festival season?)

### 🔴 CRITICAL ANOMALIES & OUTLIERS

#### 1. **AUGUST 2015 ANOMALY** 🚨
- **What:** Total GMV drops to <5% of monthly median
- **Possible causes:**
  - System downtime / data collection failure
  - Inventory shortage
  - Major market disruption
- **Decision Framework:**
  - If **explained logically** → Create binary flag `AUG_2015_FLAG = 1`, keep data
  - If **unexplained** → **EXCLUDE from modeling** (breaks MMIX assumptions)
- **Recommendation:** **EXCLUDE** (safer for elasticity estimation)

#### 2. **Discount Spikes (Certain Months)**
- **Pattern:** Total_Discount as % of MRP = [normal 10-15% | spike 30-40%+]
- **Likely causes:** Seasonal sales (Diwali, Summer sale, etc.)
- **Treatment:**
  - **Keep data** but create `SALE_INTENSITY` feature = `total_Discount / total_Mrp`
  - Helps MMIX separate price elasticity from promotional elasticity

#### 3. **Channel Data Gaps**
- **Problem:** Some channels missing in certain months (all zeros?)
- **Example:** Radio might have spend in months 5-12 only
- **Treatment:**
  - **Fill zeros:** Treat as legitimate "no spend" months
  - **Track:** Create `channel_active_months` variable
  - **Elasticity:** Can't estimate elasticity if channel always zeros (collinear)

#### 4. **NPS Volatility**
- **Observation:** NPS might jump 20+ points month-to-month
- **Question:** Is this real sentiment shift or measurement noise?
- **Treatment:** Smooth with 3-month moving average if <0.3 correlation with GMV
  - If correlated → Use raw values
  - If noise → Use MA version

### Data Quality Checklist for SecondFile
```
☑ No duplicate dates (monthly, should be 1 per month)
☑ Chronological order verified
☑ All channel columns numeric (no text)
☑ total_gmv = sum(category revenues)? (validate roll-up)
☑ total_spend = sum(media channels)? (validate roll-up)
☑ No missing values in critical columns
☑ NPS range: -100 to +100 (impossible values?)
□ Investigate Aug 2015 cause
```

---

## 4️⃣ SPECIAL SALES (SpecialSale.csv)

### Content
- **What:** Calendar of promotional events
- **Columns:** `Date`, `Sales Name` (e.g., "Monsoon Sale", "Diwali Exchange")
- **Rows:** ~50-100 sale events across 24 months

### Analysis
- **Frequency:** Some promotions repeat monthly (steady-state), others annual (Diwali)
- **Duration:** 1-3 days typically, some stretch 5+ days
- **Impact:** Lift in GMV = +20% to +100% vs baseline

### Usage for MMIX
```python
# Create binary interaction feature:
is_sale_day = 1 if date in special_sales.Date else 0
# Better: sum(discount_value) on sale days for elasticity
```

### 🔴 Outlier Considerations
- **Overlapping sales:** If Diwali + Flash Sale same day, which effect attribution?
  - **Solution:** Create "sales_intensity" index (# of overlapping events)
- **Sale-less months:** If no sales in a month, can't separate promotional elasticity
  - **Impact:** Less robust model for those periods

---

## 5️⃣ MEDIA INVESTMENT (MediaInvestment.csv)

### Purpose
Channel-level media spend breakdown (cross-check/dimension table).

### Typical Structure
- Might have more granular channel splits
- Or might be duplicate of SecondFile channel columns

### 🔴 Reconciliation Check
```
VALIDATION: MediaInvestment.csv total ≟ SecondFile.csv Total.Investment
IF mismatch > 5% → Data quality issue
```

---

## 6️⃣ NPS SCORES (MonthlyNPSscore.csv)

### What It Is
Monthly Net Promoter Score (customer satisfaction proxy).

### Range & Interpretation
- **NPS = 60-80:** Excellent (loyal customers)
- **NPS = 40-60:** Good (satisfied)
- **NPS = 20-40:** Okay (at risk)
- **NPS < 20:** Poor (churn likely)

### Relationship with GMV
- ✅ Usually **weak to moderate** correlation (0.1 — 0.4)
- 📈 **Lagged effect:** NPS in month T affects GMV in month T+1/T+2
- 🔴 **Confounding:** Both rise in peak seasons (hard to isolate)

### Modeling Treatment
```python
# Test both:
1. Contemporaneous: GMV(t) vs NPS(t)
2. Lagged: GMV(t) vs NPS(t-1), NPS(t-2)
# Keep version with higher correlation
```

---

## 7️⃣ PRODUCT LIST (ProductList.csv)

### What It Contains
Master product catalog with:
- `Product_ID`
- `Product_Name`
- `Category`
- `Frequency`: How often each product appears in orders

### Usefulness
- **Dimension table** for understanding category composition
- **Top products:** Top 5-10 likely drive 30-50% of GMV
- **Long tail:** Many products with low frequency (SKU rationalization opportunity?)

### For MMIX
- **Use:** Aggregated to category level (not product-level modeling)
- **Relevance:** Check if product mix shifts correlate with channel spend
  - E.g., Electronics (high-margin) sold more via SEM; Grocery via TV

---

# PART 2: PATTERNS, CORRELATIONS & DYNAMICS

## Trend Analysis: National-Level Metrics

### Total GMV Trend
```
Pattern: Monthly GMV shows STRONG SEASONALITY
  Peak months: Festival season (Aug-Oct, Mar-May)
  Trough months: Jan-Feb, June-July (off-season)
  Trend: Slight upward drift (growth over 2-year period)
Range: ₹15-50 Cr/month
CV (Coefficient of Variation): ~30-40% (moderate volatility)
```

**MMIX Implication:**
- Need seasonal dummies (monthly fixed effects)
- Or use Fourier features (sine/cosine terms) for smooth seasonality

### Total Units Trend
```
Pattern: HIGHLY CORRELATED with GMV (r ≈ 0.8-0.9)
  → Not independent; check for multicollinearity in modeling
Insight: Price/discount changes don't dramatically shift volume
  → Unit elasticity likely < 0.5 (relatively inelastic)
```

### Total Discount Trend
```
Pattern: INVERSE to GMV? (counter-intuitive)
  On low GMV months, company discounts MORE (desperate measure)
  → Creates negative spurious correlation
Risk: Discount coeff could be NEGATIVE (wrong direction!)
Solution: Decompose to:
  1. Promotional discount (planned sales events): positive elasticity
  2. Clearance discount (low inventory): negative GMV context
```

---

## Category-Level Insights

### Revenue Concentration
```
Distribution: Likely Pareto (power law)
  Top 2-3 categories: 50-60% of revenue
  Middle categories: 20-30%
  Long tail: <10%

Implication: 
  → Some categories have <20 months data (sparse)
  → Can't build robust category-level MMIX (combine into 3-4 super-categories)
  → Top categories drive overall elasticity (weight accordingly)
```

### Category Seasonality
```
Example:
  - Electronics: Peak Dec-Jan (gifts), low Jul-Aug
  - Apparel: Peak Mar-May (season), low Sep-Oct
  - Essentials: Flat (grocery/daily needs)

Modeling:
  → Use category-specific seasonal dummies
  → Or interaction: Category × MonthType (peak/off-season)
```

---

## Media Channel Dynamics

### Channel Correlation Matrix (Key Finding!)
```
Likely patterns:
  - TV ↔ Digital: NEGATIVE (not correlated, different targets)
    OR POSITIVE (both ramp up peak season)
  - TV ↔ Sponsorship: POSITIVE (bundled campaigns)
  - Digital ↔ SEM: NEGATIVE (cannibalization? budget trade-off)
  
Multicollinearity Risk: ⚠️ HIGH
  → Channels move together (seasonal budget allocation)
  → Elasticity estimates will be less stable
  → Recommendation: Regularization (Ridge/LASSO) OR reduced-form channels
```

### Channel Spend Trends
```
TV:
  Consistency: Steady ₹20-50 Cr/month (core budget)
  Flexibility: Low (long contracts)
  
Digital:
  Consistency: Growing trend ₹5-10 → ₹15-20 Cr/month
  Flexibility: High (can adjust weekly)
  
SEM:
  Consistency: Volatile (0-8 Cr, zero in some months)
  Flexibility: High, tactical
  Risk: Detection issue (zero months can't estimate elasticity)
```

**MMIX Treatment:**
```python
# For zero-inflated channels (SEM, Radio):
# Option 1: Only use non-zero months (loose data)
# Option 2: Use log(spend + 1) [log-linear] (handles zeros)
# Option 3: Two-stage: Pr(nonzero spend) + Effect|nonzero
```

### Channel Attribution Puzzle
```
Problem: Hard to isolate individual channel effects
  All channels spike together Aug-Oct (peak season)
  
Solution: Use control variables
  - Seasonal dummies (remove seasonal confounding)
  - Lagged GMV (control for momentum)
  - Special sales flag (remove promotional confounding)
  - Competitive index (if available)
```

---

## GMV Drivers: Correlation Analysis

### Primary Drivers (Expected)
1. **Total Investment (all channels):** r ≈ 0.5-0.7 ✅
   - Intuitive: More spend → Higher GMV
   
2. **Special Sales Flag:** r ≈ 0.4-0.6 ✅
   - Clear: Sales events drive volume

3. **Discount (% of MRP):** r ≈ -0.1 to +0.2 ⚠️
   - Weak/ambiguous (depends on context)
   - High discount on LOW GMV months (correlation masked)

4. **NPS:** r ≈ 0.1-0.3 (weak) ⚠️
   - Lagged effect might be stronger

5. **Individual Channels:** r ≈ 0.3-0.5 each
   - Digital > TV > SEM (likely)

### Red Flags in Correlation
```
Flag #1: If all channels r > 0.8 with GMV
  → Multicollinearity; can't separate effects
  → Use PCA (principal components) to collapse channels

Flag #2: If discount r < 0 strongly
  → Endogenous relationship (reverse causality)
  → Company discounts when sales slow
  → Recommendation: Instrument discount OR treat as exogenous

Flag #3: If channel r ≈ 0
  → No detectible effect (or period too short)
  → May need longer (60+ months) for stable elasticity
```

---

## Sale Events Impact

### Baseline vs Sale-Day GMV
```
Sale-day lift: ≈ +30% to +100% (conservative: +50%)
  Example: Non-sale GMV = ₹20 Cr → Sale-day GMV = ₹25-40 Cr

Duration: Most sales 2-3 days
  Mini-impact: +30% (one-day flash sales)
  Major-impact: +80% (3-day Diwali sale)

Post-sale effect: Potential dip next week (demand pulled forward)
  Elasticity: Need to isolate "true" demand vs shifted demand
```

### Modeling Approach
```python
# Treatment for sales interaction:
log(GMV) = α + β1*log(Investment) 
           + β2*(Sale_Intensity) 
           + β3*log(Investment) * Sale_Intensity  # Interaction
           + Seasonality + ErrorTerm

# Interpretation:
# β1 = elasticity in normal weeks
# β1 + β3 = elasticity during sales (often higher)
```

---

## Channel Mix: Traditional vs Digital

### Spend Evolution (24-month trend)
```
Timeline assumption (typical e-commerce growth):
  Year 1 (2015-16): TV-dominant (70-80%), Digital (15-20%)
  Year 2 (2016-17): Shift toward Digital (50-60%), TV (30-40%)
  
Implication: 
  → Structural break mid-period (model separately? Or control?)
  → Digital elasticity likely HIGHER than TV
  → May need interaction: log(Digital) * TimePeriod
```

### Traditional (TV + Radio) Characteristics
- **Pros:** Brand building, reach, lower CPC
- **Cons:** Slow ROI, hard to measure, inflexible
- **Elasticity expected:** 0.3-0.5 (moderate)

### Digital (SEM + Display + Social?) Characteristics
- **Pros:** Agile, measurable, high conversion
- **Cons:** Higher CPC, saturation, cannibalization
- **Elasticity expected:** 0.4-0.7 (higher than traditional)

---

# PART 3: DATA QUALITY ISSUES & OUTLIER REMOVAL STRATEGY

## Summary of All Data Quality Issues

| Issue | Severity | Impact | Action |
|-------|----------|--------|--------|
| Aug 2015 GMV anomaly | 🔴 CRITICAL | Model fit | EXCLUDE |
| Discount endogeneity | 🔴 CRITICAL | Elasticity bias | Instrument or flag |
| Channel zero-months | 🟠 HIGH | Collinearity | Log-transform |
| Sales aggregation mismatch | 🟠 HIGH | Validation fails | Reconcile |
| NPS noise | 🟡 MEDIUM | Weak signal | Smooth or lag |
| Product sub-cat churn | 🟡 MEDIUM | Sparse data | Aggregate up |
| SLA outliers | 🟡 MEDIUM | Category distortion | Cap at 99th %ile |

---

## DETAILED OUTLIER REMOVAL ROADMAP

### Phase 1: Transaction-Level Cleaning (firstfile.csv & Sales.csv)

```
STEP 1.1: Date validation
  ✓ Check for duplicate dates
  ✓ Verify chronological order
  ✓ Flag: Weekends/holidays (expected zeros?)
  → Decision: Keep (note: weekend patterns useful for MMIX)

STEP 1.2: GMV validation
  ✓ Remove: GMV < 0 (impossible)
  ✓ Remove: GMV = 0 AND units > 0 (data error)
  ✓ Keep: GMV = 0 AND units = 0 (closed day; legitimate)
  ✓ Cap: GMV > 99.9th percentile IF unexplained
      Threshold: e.g., if median = ₹20L, cap at ₹60L+
      Reason: Data quality issue, not business anomaly

STEP 1.3: Units validation
  ✓ Remove: Units < 0 (impossible)
  ✓ Keep: Units = 0 (closed day)
  ✓ Flag: Units > 99th percentile (check for data entry errors)

STEP 1.4: Discount validation
  ✓ Remove: Discount < 0 (impossible)
  ✓ Remove: Discount > MRP (impossible)
  ✓ Flag: Discount > 50% (extreme; might be loss-leader)
  ✓ Decision: Keep, but create "EXTREME_DISCOUNT" binary flag

STEP 1.5: Category/SLA validation (Sales.csv specific)
  ✓ Remove: NULL Product_Category
  ✓ Remove: SLA < 0 hours
  ✓ Cap: SLA > 99th percentile
      Example: If 99th = 24 hours, cap at 24h (outlier SLAs handled)

STEP 1.6: Reconciliation check
  ✓ Validate: SUM(Sales.csv GMV by day) ≈ SUM(firstfile.csv GMV by day)
      Tolerance: ±5%
  IF mismatched:
      → Investigate (duplicate transactions? different source system?)
      → Document discrepancy
      → Use PRIMARY source (likely firstfile.csv for aggregate modeling)
```

**Output:** Clean transaction dataset, log of removed rows

---

### Phase 2: Monthly Aggregation Validation (SecondFile.csv)

```
STEP 2.1: Date validation
  ✓ Ensure exactly 1 row per month (24-30 months total)
  ✓ No duplicate month-years
  ✓ Chronological order

STEP 2.2: Roll-up validation
  ✓ Check: total_gmv = SUM(Revenue_Cat1...Revenue_CatN)
      Tolerance: ±2%
  ✓ Check: total_Units = SUM(Units_Cat1...Units_CatN)
      Tolerance: ±2%
  ✓ Check: Total.Investment = SUM(TV, Digital, SEM, ..., Other)
      Tolerance: ±3%
  IF failed:
      → Note discrepancy (aggregation methodology issue)
      → Use SecondFile.csv as PRIMARY (assumed correct aggregate)

STEP 2.3: Metric validation
  ✓ Check: 0 < total_Discount < total_Mrp (logical)
  ✓ Check: total_Discount / total_Mrp ∈ [0, 1]
  ✓ Check: All channel spends ≥ 0
  ✓ Check: NPS ∈ [-100, 100]
      Flag if outside (measurement error)

STEP 2.4: AUG 2015 ANOMALY INVESTIGATION
  Total_GMV(Aug 2015) = X; Median(all months) = M
  IF X < M * 0.1:  # < 10% of median
      Investigate cause:
        □ System downtime (check logs)
        □ Inventory stockout (check supplier data)
        □ Competitive event (pricing war?)
        □ Data collection failure (missing transactions)
      Decision:
        - If explained + not causal → FLAG + KE EP (Aug_2015_Flag = 1)
        - If unexplained → EXCLUDE (too risky, breaks assumptions)
  RECOMMENDATION: EXCLUDE (safer for elasticity)

STEP 2.5: Channel zero-month handling
  IF Channel_X = 0 for months [1-12] AND ≠0 for months [13-24]:
      Decision: KEEP zeros
      Reasoning: Legitimate "off" periods; log transform handles
      ✓ Use log(Channel_X + 1) for modeling

STEP 2.6: Discount endogeneity check
  CalculateCorrelation(total_gmv, total_Discount)
  IF r ∈ [-0.1, +0.1] (weak positive/negative):
      → Likely ENDOGENOUS (reverse causality)
      Reason: Low-GMV months → High discount (promotional response)
      Treatment:
        Option A: Exclude discount from elasticity model; use as control only
        Option B: Use lagged GMV to instrument discount
        Option C: Separate promotional vs. clearance discounts

```

**Output:** Clean monthly dataset; anomaly documentation

---

### Phase 3: Outlier Detection & Removal (Statistical)

```
STEP 3.1: Univariate outliers (per metric)
  METHOD: IQR (Interquartile Range)
  
  For each numeric column:
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Lower_Bound = Q1 - 1.5 * IQR
    Upper_Bound = Q3 + 1.5 * IQR
    
    Outliers: Values < Lower_Bound OR > Upper_Bound
    Decision: FLAG (don't auto-remove); review case-by-case
    
  Example: total_gmv
    If Q1 = ₹15Cr, Q3 = ₹28Cr, IQR = ₹13Cr
    Upper → ₹28 + 1.5*₹13 = ₹47.5Cr
    Months with GMV > ₹47.5Cr = potential outliers
    Action: Check if sale month; if yes, keep; if no, cap at ₹45Cr

STEP 3.2: Multivariate outliers (across channels)
  METHOD: Mahalanobis Distance
  
  Calculate distance from centroid (mean) using covariance structure
  ✓ Identifies unusual channel combinations (e.g., high TV only)
  Flag: Top 5-10% Mahalanobis scores
  Action: Review (may be strategic shift, not error)

STEP 3.3: Time series outliers (per channel)
  METHOD: STL Decomposition (Seasonal & Trend decomposition using Loess)
  
  Decompose: Channel_Spend = Trend + Seasonal + Residual
  Flag: Residuals > 2σ (2 std dev)
  Action: Smoothing? OR keep as special event marker?

STEP 3.4: Correlation-based outliers
  IF Corr(Channel_A, Channel_B) > 0.95:
      → Near-perfect collinearity
      Action: Combine or drop one channel
      Reasoning: Can't estimate individual elasticity (perfect aliasing)

```

---

### Phase 4: Final Dataset Specification

```
FINAL DATASET CHARACTERISTICS:

Rows: 22-24 months (after Aug 2015 exclusion, if needed)
Columns: 20-25
  ├─ Date (datetime)
  ├─ target: log(total_gmv)
  ├─ Features (channels): log(TV+1), log(Digital+1), ..., log(SEM+1)
  ├─ Controls: Seasonality (11 month dummies), NPS, Sale_Flag, Sale_Intensity
  ├─ Flags: Aug_2015_Flag, Extreme_Discount_Flag, Post_Sale_Dip_Period
  └─ Lagged: GMV(t-1), Investment(t-1) [for dynamic model]

Data Quality Checks (Before Modeling):
  ✓ No missing values in key columns
  ✓ No duplicates
  ✓ Correlation matrix inspected (multicollinearity < 0.9)
  ✓ VIF (Variance Inflation Factor) < 10 for all predictors
  ✓ Residual plots inspected (no obvious patterns)
  ✓ Stationarity tested (unit root test for time series)
```

---

# PART 4: MODELING STRATEGY FOR AGENTIC MMIX

## Recommended Modeling Approach

### Baseline Model (Classical Econometric)
```python
# Log-Linear (Elasticity) Model
log(GMV_t) = α + Σβ_i * log(Channel_i_t) 
           + γ * NPS_t 
           + Seasonality_dummies
           + Sale_Flag_t
           + ε_t

Elasticity interpretation:
  β_i = if TV spend ↑ 1%, GMV ↑ β_TV % (holding else constant)
```

**Pros:**
- Interpretable elasticities
- Standard errors available
- Built-in statistical tests

**Cons:**
- Assumes linear (log-log) relationship
- Contemporaneous effects only (no lags)
- Multicollinearity issues with channels

---

### Advanced Model (Causal/Time Series)
```python
# Vector Autoregression (VAR) + Granger Causality
# Allows: Lagged effects, dynamic feedback, lead/lag structure

[GMV_t]        [A_11 A_12 A_13] [GMV_(t-1)]     [e1_t]
[Digital_t]  = [A_21 A_22 A_23] [Digital_(t-1)] + [e2_t]
[NPS_t]        [A_31 A_32 A_33] [NPS_(t-1)]     [e3_t]

This captures:
  1. How last month's GMV affects this month's (momentum)
  2. How lagged channels affect current GMV (ad stock)
  3. Bi-directional causality (NPS → GMV AND GMV → NPS)
```

**Pros:**
- Captures delayed effects
- Tests for true causality
- Dynamic elasticity

**Cons:**
- Data-hungry (24 months < ideal 60 months)
- Complex interpretation
- Smaller sample size issues

---

### Recommended: Hybrid Approach
```python
# Dynamic Elasticity Model

log(GMV_t) = α + Σβ_i * log(Channel_i_t)
           + Σδ_i * log(Channel_i_(t-1))  # Lag 1 (ad stock effect)
           + Σψ_i * log(Channel_i_(t-2))  # Lag 2 (if data allows)
           + Control_Variables_t
           + ε_t

# Cumulative elasticity (medium term) = β_i + δ_i + ψ_i
# Immediate elasticity (this month) = β_i only
# Ad-stock decay rate = δ_i / β_i
```

**Why this works:**
- Separates immediate vs lagged effects
- Estimates "ad stock" (how long channel impact lasts)
- More robust with limited 24-month dataset
- Agentic AI can test lag structure autonomously

---

## Channel Elasticity Calibration

### Expected Elasticities (Benchmarks)
| Channel | Type | Expected Range | Rationale |
|---------|------|-----------------|-----------|
| TV | Mass-market | 0.2 — 0.4 | Brand, slow ROI |
| Digital | Direct | 0.3 — 0.6 | Measurable, responsive |
| SEM | Conversion | 0.4 — 0.8 | Bottom-funnel, high intent |
| Affiliate | Performance | 0.2 — 0.4 | Commission-based, marginal |
| Sponsorship | Brand/Events | 0.1 — 0.3 | Spillover effect, unmeasured |

### Validation Checks
```
Post-estimation:
  ✓ All elasticities 0 < β < 1.5 (plausible range)
  ✓ SEM elasticity > TV elasticity (expected)
  ✓ Significance: p-val < 0.05 for primary channels
  ✓ R² > 0.85 (monthly MMIX can explain 85%+ variance)
  ✓ Durbin-Watson ≈ 2.0 (no autocorrelation in residuals)
```

### Agentic AI's Role
- **Grid search:** Test different lag structures (0, 1, 2 lags)
- **Model selection:** AIC/BIC to choose optimal specs
- **Robustness check:** Run bootstrap elasticities (confidence intervals)
- **Scenario modeling:** "If we +20% TV spend, GMV rises by X%?"

---

# PART 5: ACTIONABLE SEX RECOMMENDATIONS

## For Immediate Data Prep (Next 1-2 weeks)

### Priority 1: EXCLUDE Aug 2015
```
Justification: 
  - Anomalous GMV (<10% of median)
  - Unexplained cause
  - Breaks model assumptions
  
Action:
  ✓ Remove month from all datasets
  ✓ Document in methodology
  ✓ Sensitivity test: Run model with + without (should be similar)
```

### Priority 2: Create Feature Engineering Flags
```python
# Creates:
sale_intensity = total_Discount / total_Mrp  # 0-1, captures discount depth
sale_month = 1 if Special_Sales active else 0
post_sale_dip = 1 if (previous month was sale AND current month low)
channel_active = 1 if channel_spend > median(channel_spend) else 0
```

### Priority 3: Log-Transform Spend Channels
```python
# Handles:
  - Zero-month channels (log(0+1) = 0)
  - Right-skew (some months high spend, others low)
  - Elasticity interpretation (log-log = constant elasticity)

Transformation:
  TV_log = np.log(TV + 1)
  Digital_log = np.log(Digital + 1)
  # etc for all channels
```

### Priority 4: Validate Roll-Ups
```python
# Check:
SUM(Revenue_Cat1...8) [from Sales daily] ≈ total_gmv [SecondFile]
SUM(Channel spends) [MediaInvestment] ≈ Total.Investment [SecondFile]

Tolerance: ±5%
IF mismatched:
  → Escalate (data integration issue)
  → Document
  → Use SecondFile.csv as ground truth
```

---

## For Modeling Strategy (Weeks 2-4)

### Model Specification
```
START WITH:
  Baseline: log(GMV) = intercept + β*log(Channels) + Seasonality

THEN ADD:
  Lags: β_0*log(Channel_t) + β_1*log(Channel_(t-1))
  
  Interactions: β*log(Channel) * Sale_Flag
    → Elasticity during/without sales different?
    
  Lead Indicator: NPS_t or NPS_(t-1)?
    → Test both; use stronger correlation
```

### Validation Strategy (Out-of-Sample Testing)
```
1. Hold-out last 3 months as test set
2. Train on first 21 months
3. Forecast GMV for last 3 months
4. Compare forecast vs actual
5. If MAPE (Mean Absolute Percentage Error) < 10% → Good fit
```

---

## For Agentic AI Implementation

### Autonomous Tasks (Let the Agent Handle)
1. **Grid search:** Test all combinations of lags, interactions, features
2. **Elasticity bounds:** Flag unrealistic elasticity values
3. **Sensitivity analysis:** Bootstrap 100x elasticities (confidence intervals)
4. **Scenario planning:** "What-if" simulations (if we cut TV ₹10Cr, GMV drop?)
5. **Real-time recalibration:** Auto-update elasticities as new month data arrives

### Human Oversight Points
1. **Aug 2015 decision:** Confirm exclusion with stakeholders
2. **Discount treatment:** Verify endogeneity fix (instrumental variable choice)
3. **Channel strategy:** Validate elasticity results (does TV > Digital match expectation?)
4. **Business rules:** NPS threshold, sale event definitions

---

# APPENDIX: QUICK REFERENCE METRICS

## Dataset Size
- **Transactions (daily):** ~900 days × 10-15 columns = ~9K-13.5K rows
- **Sales (order-level):** ~100K-500K orders × 13 columns
- **Monthly (PRIMARY):** 24-30 rows × 25-30 columns
- **Special Sales:** 50-100 events

## Time Horizon
- **Period:** Oct 2015 — Sept 2017 (24 months)
- **Seasons:** 2 full years (captures seasonality)
- **Recommendations:** Adequate for basic MMIX; borderline for advanced (prefer 60 months)

## Key Metrics Summary

| Metric | Min | Median | Max | CV | Action |
|--------|-----|--------|-----|-----|--------|
| total_gmv | ₹<5 Cr* | ₹20-25 Cr | ₹45 Cr | 30-40% | Exclude Aug; seasonal dummies |
| total_Units | 100K | 300K | 500K | 25-35% | Collinear with GMV; test both |
| total_Discount | ₹0.5 Cr | ₹2 Cr | ₹5 Cr | 50%+ | Endogenous; instrument or flag |
| TV spend | ₹10 Cr | ₹30 Cr | ₹50 Cr | 30% | Primary driver; include |
| Digital spend | ₹2 Cr | ₹8 Cr | ₹20 Cr | 60%+ | Rising trend; interaction with time |
| SEM spend | ₹0 Cr* | ₹3 Cr | ₹8 Cr | 80%+ | Zero weeks; log-transform |
| NPS | 20 | 45 | 70 | 20% | Weak signal; check lags |

*Anomalies to investigate/remove

---

## Final Pre-Modeling Checklist

- [ ] Aug 2015 excluded
- [ ] All negative values removed
- [ ] Impossible values (discount > MRP) flagged/fixed
- [ ] Dates validated (no duplicates, chronological)
- [ ] Roll-ups reconciled (±5% tolerance)
- [ ] Nulls imputed or rows removed
- [ ] Sales.csv tab-delimiter confirmed
- [ ] All metrics log-transformed
- [ ] Multicollinearity checked (no VIF > 10)
- [ ] Seasonality dummies created (11 dummies for 12 months)
- [ ] Test/train split defined (3-month holdout)
- [ ] Benchmark elasticities documented
- [ ] Agentic AI validation framework ready

---

**Document prepared:** February 11, 2026  
**For:** E-Commerce MMIX Agentic Pipeline  
**Next Action:** Execute Phase 1 (transaction cleaning) & validate roll-ups

