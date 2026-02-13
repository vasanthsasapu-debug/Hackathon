"""
=============================================================================
MODELING ENGINE -- E-Commerce MMIX
=============================================================================
Agent-ready: Every function returns:
    {"data": ..., "log": [...], "summary": "...", "decisions": [...]}

Modules:
  1. Model Registry (specs, types, transforms)
  2. Model Builder (train individual models)
  3. Ordinality Enforcement
  4. Cross-Validation (Leave-One-Out for small n)
  5. Model Scorer & Ranker
  6. Insight Convergence Analysis
  7. Scenario Simulator
  8. Diagnostics & Visualization
  9. Master Pipeline
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product as iter_product
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style("whitegrid")


# =============================================================================
# 1. MODEL REGISTRY
# =============================================================================

def get_feature_specs():
    """
    Define all feature specifications (business hypotheses to test).
    Each spec answers a different question about what drives GMV.
    """
    return {
        "spec_A_grouped_channels": {
            "features": ["log_spend_traditional", "log_spend_digital_performance", "sale_flag"],
            "description": "Grouped channel types + sale events",
            "question": "Do traditional and digital channel groups explain GMV differently?"
        },
        "spec_B_total_spend": {
            "features": ["log_Total_Investment", "sale_flag", "nps_standardized"],
            "description": "Aggregate spend + NPS control + sale events",
            "question": "Does total spend with brand health explain GMV?"
        },
        "spec_C_top_channels": {
            "features": ["log_Online.marketing", "log_Sponsorship", "sale_flag"],
            "description": "Top 2 correlated individual channels + sale events",
            "question": "Do the strongest individual channels suffice?"
        },
        "spec_D_with_momentum": {
            "features": ["log_spend_digital_performance", "sale_flag", "total_gmv_lag1"],
            "description": "Digital performance + sale + lagged GMV (momentum)",
            "question": "Does last month's GMV predict this month's?"
        },
        "spec_E_discount_effect": {
            "features": ["log_Total_Investment", "discount_intensity", "sale_flag"],
            "description": "Total spend + discount depth + sale events",
            "question": "Does discounting drive GMV beyond media spend?"
        },
        "spec_F_mixed_channels": {
            "features": ["log_Affiliates", "log_TV", "sale_flag"],
            "description": "Performance (Affiliates) + brand (TV) + sale events",
            "question": "Does a mix of performance and brand channels work?"
        },
        "spec_G_spend_only": {
            "features": ["log_spend_digital_performance", "log_spend_traditional", "discount_intensity"],
            "description": "All spend groups + discount, no sale flag",
            "question": "Can spend alone explain GMV without sale events?"
        },
        "spec_H_sale_duration": {
            "features": ["log_Total_Investment", "sale_days", "nps_standardized"],
            "description": "Total spend + sale duration + NPS",
            "question": "Does sale duration matter more than binary flag?"
        }
    }


def get_model_types():
    """Define model types (algorithms) to test."""
    return {
        "OLS": {
            "description": "Ordinary Least Squares -- baseline, fully interpretable",
            "builder": build_ols
        },
        "Ridge": {
            "description": "Ridge regression -- handles multicollinearity, more stable",
            "builder": build_ridge
        },
        "Bayesian": {
            "description": "Bayesian Ridge -- uncertainty estimates, regularized",
            "builder": build_bayesian
        }
    }


def get_transform_variants():
    """Define target variable transformations to test."""
    return {
        "log_log": {
            "target": "log_total_gmv",
            "use_log_features": True,
            "description": "Log-Log: elasticity (1% spend change -> beta% GMV change)"
        },
        "log_linear": {
            "target": "log_total_gmv",
            "use_log_features": False,
            "description": "Log-Linear: raw spend features, log target (tests if diminishing returns assumption holds)"
        }
    }


# =============================================================================
# 2. MODEL BUILDERS
# =============================================================================

def build_ols(X, y, feature_names):
    """
    Build OLS model using statsmodels (gives full statistical output).
    Runs both raw and standardized versions:
    - Raw: for interpretable coefficients in original units
    - Standardized: for comparing relative feature importance
    """
    # --- Raw OLS (interpretable coefficients) ---
    X_const = sm.add_constant(X, has_constant='add')
    try:
        model = sm.OLS(y, X_const).fit()
        coefficients = dict(zip(['const'] + feature_names, model.params))
        pvalues = dict(zip(['const'] + feature_names, model.pvalues))

        # --- Standardized OLS (comparable magnitudes) ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_const = sm.add_constant(X_scaled, has_constant='add')
        model_std = sm.OLS(y, X_scaled_const).fit()
        standardized_coefs = dict(zip(['const'] + feature_names, model_std.params))

        return {
            "model": model,
            "scaler": scaler,
            "coefficients": coefficients,
            "standardized_coefficients": standardized_coefs,
            "pvalues": pvalues,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "aic": model.aic,
            "bic": model.bic,
            "predictions": model.fittedvalues.values,
            "residuals": model.resid.values,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_ridge(X, y, feature_names):
    """Build Ridge regression using sklearn."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)

        coefficients = dict(zip(feature_names, model.coef_))
        coefficients['const'] = model.intercept_

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot
        n, p = X.shape
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1) if n > p + 1 else r_sq

        return {
            "model": model,
            "scaler": scaler,
            "coefficients": coefficients,
            "pvalues": {},  # Ridge doesn't give p-values
            "r_squared": r_sq,
            "adj_r_squared": adj_r_sq,
            "predictions": predictions,
            "residuals": (y - predictions),
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_bayesian(X, y, feature_names):
    """Build Bayesian Ridge regression using sklearn."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = BayesianRidge()
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)

        coefficients = dict(zip(feature_names, model.coef_))
        coefficients['const'] = model.intercept_

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot
        n, p = X.shape
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1) if n > p + 1 else r_sq

        return {
            "model": model,
            "scaler": scaler,
            "coefficients": coefficients,
            "pvalues": {},
            "r_squared": r_sq,
            "adj_r_squared": adj_r_sq,
            "predictions": predictions,
            "residuals": (y - predictions),
            "alpha": model.alpha_,  # Precision of the noise
            "lambda": model.lambda_,  # Precision of the weights
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# 3. ORDINALITY ENFORCEMENT
# =============================================================================

def check_ordinality(coefficients, feature_names):
    """
    Check if coefficients satisfy business logic:
    - Spend channels should have positive coefficients (more spend -> more GMV)
    - sale_flag / sale_days should be positive
    - NPS can be any direction (control variable)
    - discount_intensity can be any direction (ambiguous)
    """
    spend_features = [f for f in feature_names if any(
        k in f.lower() for k in ['log_', 'spend_', 'investment']
    )]
    sale_features = [f for f in feature_names if 'sale' in f.lower()]

    violations = []
    checks = []

    for f in spend_features:
        coef = coefficients.get(f, 0)
        passed = coef >= 0
        checks.append({"feature": f, "coefficient": coef, "expected": ">= 0", "passed": passed})
        if not passed:
            violations.append(f"{f} = {coef:.4f} (expected positive)")

    for f in sale_features:
        coef = coefficients.get(f, 0)
        passed = coef >= 0
        checks.append({"feature": f, "coefficient": coef, "expected": ">= 0", "passed": passed})
        if not passed:
            violations.append(f"{f} = {coef:.4f} (expected positive)")

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "checks": checks,
        "n_violations": len(violations)
    }


# =============================================================================
# 4. CROSS-VALIDATION (Leave-One-Out)
# =============================================================================

def loo_cross_validation(X, y, model_type_name, builder_func):
    """
    Leave-One-Out cross-validation.
    Best choice for very small n (11 data points).
    Returns CV R-squared and stability metrics.
    """
    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    fold_r2 = []

    feature_names = list(X.columns) if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y

    for train_idx, test_idx in loo.split(X_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        result = builder_func(X_train, y_train, feature_names)
        if not result.get('success', False):
            continue

        if model_type_name == "OLS":
            X_test_const = sm.add_constant(X_test, has_constant='add')
            pred = result['model'].predict(X_test_const)
        else:
            scaler = result.get('scaler')
            if scaler:
                X_test_scaled = scaler.transform(X_test)
                pred = result['model'].predict(X_test_scaled)
            else:
                pred = result['model'].predict(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(pred[0])

    if len(y_true_all) < 3:
        return {"cv_r2": None, "cv_rmse": None, "cv_mape": None, "stability": 0}

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    cv_r2 = r2_score(y_true_all, y_pred_all)
    cv_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    cv_mape = mean_absolute_percentage_error(y_true_all, y_pred_all) * 100

    # Stability: how close is CV R2 to training R2? (1.0 = perfect stability)
    # Will be computed externally by comparing cv_r2 to train_r2

    return {
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "cv_mape": cv_mape,
        "y_true": y_true_all,
        "y_pred": y_pred_all
    }


# =============================================================================
# 5. MODEL SCORER & RANKER
# =============================================================================

def score_model(train_result, cv_result, ordinality_result):
    """
    Compute composite score for a single model.
    Weights: 40% fit + 30% stability + 30% ordinality
    """
    if not train_result.get('success', False):
        return {"composite": 0, "fit_score": 0, "stability_score": 0, "ordinality_score": 0}

    # Fit score: adjusted R-squared (0 to 1)
    adj_r2 = train_result.get('adj_r_squared', 0)
    fit_score = max(0, min(1, adj_r2))

    # Stability score: 1 - abs(train_r2 - cv_r2) / train_r2
    train_r2 = train_result.get('r_squared', 0)
    cv_r2 = cv_result.get('cv_r2', 0)
    if cv_r2 is not None and train_r2 > 0:
        stability_score = max(0, 1 - abs(train_r2 - cv_r2) / train_r2)
    else:
        stability_score = 0

    # Ordinality score: 1 if pass, penalized by violations
    if ordinality_result['passed']:
        ordinality_score = 1.0
    else:
        n_features = len(ordinality_result['checks'])
        n_violations = ordinality_result['n_violations']
        ordinality_score = max(0, 1 - n_violations / max(n_features, 1))

    composite = 0.4 * fit_score + 0.3 * stability_score + 0.3 * ordinality_score

    return {
        "composite": round(composite, 4),
        "fit_score": round(fit_score, 4),
        "stability_score": round(stability_score, 4),
        "ordinality_score": round(ordinality_score, 4)
    }


def rank_models(all_results):
    """Rank all models by composite score. Return sorted list."""
    ranked = sorted(all_results, key=lambda x: x['scores']['composite'], reverse=True)
    for i, r in enumerate(ranked):
        r['rank'] = i + 1
    return ranked


# =============================================================================
# 6. INSIGHT CONVERGENCE ANALYSIS
# =============================================================================

def analyze_convergence(all_results):
    """
    Across all successful models, which insights are consistent?
    This is the real value -- not which model is #1, but what findings hold.
    """
    log = []
    insights = {}

    successful = [r for r in all_results if r['train_result'].get('success', False)]
    if not successful:
        return {"insights": {}, "log": ["No successful models to analyze"],
                "summary": "No models succeeded."}

    # Collect all coefficients across models
    feature_coefs = {}
    for r in successful:
        coefs = r['train_result'].get('coefficients', {})
        for feat, val in coefs.items():
            if feat == 'const':
                continue
            if feat not in feature_coefs:
                feature_coefs[feat] = []
            feature_coefs[feat].append(val)

    # Analyze each feature
    for feat, values in feature_coefs.items():
        n_models = len(values)
        if n_models < 2:
            continue

        mean_coef = np.mean(values)
        std_coef = np.std(values)
        n_positive = sum(1 for v in values if v > 0)
        n_negative = sum(1 for v in values if v < 0)
        pct_positive = n_positive / n_models * 100

        if pct_positive >= 80:
            direction = "POSITIVE (confirmed)"
        elif pct_positive <= 20:
            direction = "NEGATIVE (confirmed)"
        else:
            direction = "MIXED (inconclusive)"

        insights[feat] = {
            "mean_coefficient": round(mean_coef, 4),
            "std_coefficient": round(std_coef, 4),
            "n_models": n_models,
            "pct_positive": round(pct_positive, 1),
            "direction": direction
        }

        log.append(f"{feat}: mean={mean_coef:.4f}, {direction} ({n_positive}/{n_models} positive)")

    # Sale flag convergence
    sale_keys = [k for k in insights if 'sale' in k.lower()]
    if sale_keys:
        all_positive = all(insights[k]['pct_positive'] >= 80 for k in sale_keys)
        log.append(f"Sale events: {'CONFIRMED as driver' if all_positive else 'MIXED evidence'}")

    summary = (
        f"Convergence analysis across {len(successful)} models. "
        f"{len(insights)} features analyzed. "
        f"{sum(1 for v in insights.values() if 'confirmed' in v['direction'].lower())} "
        f"features have consistent direction across models."
    )

    return {"insights": insights, "log": log, "summary": summary}


# =============================================================================
# 7. SCENARIO SIMULATOR
# =============================================================================

def build_scenario_simulator(best_model_result, feature_matrix, feature_names, target_col):
    """
    Build a what-if scenario simulator using the best model.
    Returns a function that takes spend changes and predicts GMV.
    """
    if not best_model_result['train_result'].get('success', False):
        return None

    model = best_model_result['train_result']['model']
    model_type = best_model_result['model_type']
    scaler = best_model_result['train_result'].get('scaler', None)

    # Get baseline values (mean of each feature)
    baseline = feature_matrix[feature_names].mean().to_dict()
    baseline_target = feature_matrix[target_col].mean()

    def simulate(changes):
        """
        Simulate GMV change given spend changes.

        Args:
            changes: dict of {feature_name: multiplier}
                     e.g. {"log_Online.marketing": 1.2} means +20% spend
                     For log features, we convert: log(original * multiplier + 1)

        Returns:
            dict with baseline GMV, new GMV, change %
        """
        scenario = baseline.copy()

        for feat, multiplier in changes.items():
            if feat in scenario:
                if feat.startswith('log_'):
                    # Convert back from log, apply multiplier, re-log
                    original = np.expm1(scenario[feat])
                    new_val = original * multiplier
                    scenario[feat] = np.log1p(new_val)
                else:
                    scenario[feat] = scenario[feat] * multiplier

        X_new = np.array([[scenario[f] for f in feature_names]])

        if model_type == "OLS":
            X_new_const = sm.add_constant(X_new, has_constant='add')
            predicted = model.predict(X_new_const)[0]
        else:
            if scaler:
                X_new_scaled = scaler.transform(X_new)
                predicted = model.predict(X_new_scaled)[0]
            else:
                predicted = model.predict(X_new)[0]

        # Convert from log if needed
        if target_col.startswith('log_'):
            predicted_gmv = np.expm1(predicted)
            baseline_gmv = np.expm1(baseline_target)
        else:
            predicted_gmv = predicted
            baseline_gmv = baseline_target

        change_pct = (predicted_gmv / baseline_gmv - 1) * 100

        return {
            "baseline_gmv": baseline_gmv,
            "predicted_gmv": predicted_gmv,
            "change_pct": change_pct,
            "scenario": changes
        }

    return simulate


def run_standard_scenarios(simulator):
    """Run a set of standard business scenarios."""
    if simulator is None:
        return []

    scenarios = [
        {"name": "Baseline (no change)", "changes": {}},
        {"name": "+20% Online.marketing", "changes": {"log_Online.marketing": 1.2}},
        {"name": "+20% Sponsorship", "changes": {"log_Sponsorship": 1.2}},
        {"name": "+20% TV", "changes": {"log_TV": 1.2}},
        {"name": "+20% Total Investment", "changes": {"log_Total_Investment": 1.2}},
        {"name": "-10% all spend", "changes": {
            "log_spend_traditional": 0.9, "log_spend_digital_performance": 0.9,
            "log_spend_digital_brand": 0.9
        }},
        {"name": "+50% digital_performance", "changes": {"log_spend_digital_performance": 1.5}},
        {"name": "Add sale event", "changes": {"sale_flag": 999}},  # Force to 1
    ]

    results = []
    for s in scenarios:
        # Filter changes to only features the simulator knows
        try:
            result = simulator(s['changes'])
            result['scenario_name'] = s['name']
            results.append(result)
        except Exception:
            continue

    return results


# =============================================================================
# 8. DIAGNOSTICS & VISUALIZATION
# =============================================================================

def plot_model_rankings(ranked_results, top_n=10, save_dir=None):
    """Plot top N model rankings."""
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    top = ranked_results[:top_n]
    names = [f"{r['spec_name']}\n{r['model_type']}\n{r['transform']}" for r in top]
    composites = [r['scores']['composite'] for r in top]
    fits = [r['scores']['fit_score'] for r in top]
    stabilities = [r['scores']['stability_score'] for r in top]
    ordinalities = [r['scores']['ordinality_score'] for r in top]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Model Rankings', fontsize=16, fontweight='bold')

    # Composite scores
    colors = ['#4CAF50' if c > 0.7 else '#FF9800' if c > 0.5 else '#F44336' for c in composites]
    axes[0].barh(range(len(top)), composites, color=colors)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel('Composite Score')
    axes[0].set_title(f'Top {top_n} Models (Composite Score)')
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()

    # Component breakdown for top 5
    top5 = top[:5]
    x = range(len(top5))
    w = 0.25
    names5 = [f"{r['spec_name'][:15]}\n{r['model_type']}" for r in top5]
    axes[1].bar([i - w for i in x], [r['scores']['fit_score'] for r in top5], w, label='Fit', color='#2196F3')
    axes[1].bar(x, [r['scores']['stability_score'] for r in top5], w, label='Stability', color='#4CAF50')
    axes[1].bar([i + w for i in x], [r['scores']['ordinality_score'] for r in top5], w, label='Ordinality', color='#FF9800')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names5, fontsize=8)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Top 5 Models -- Score Breakdown')
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'model_rankings.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_best_model_diagnostics(best_result, feature_matrix, save_dir=None):
    """Plot diagnostics for the best model."""
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    train = best_result['train_result']
    if not train.get('success', False):
        print("  [WARN] Best model failed -- cannot plot diagnostics")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Best Model: {best_result['spec_name']} | {best_result['model_type']} | "
        f"{best_result['transform']}", fontsize=13, fontweight='bold'
    )

    predictions = train['predictions']
    residuals = train['residuals']
    target = best_result['transform_config']['target']
    y_actual = feature_matrix[target].values[:len(predictions)]

    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_actual, predictions, s=80, color='#9C27B0', edgecolors='white')
    min_val = min(y_actual.min(), predictions.min())
    max_val = max(y_actual.max(), predictions.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title(f'Actual vs Predicted (R2={train["r_squared"]:.3f})')

    # 2. Residuals vs Predicted
    axes[0, 1].scatter(predictions, residuals, s=80, color='#FF5722', edgecolors='white')
    axes[0, 1].axhline(y=0, color='black', ls='--', alpha=0.5)
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')

    # 3. Coefficients
    coefs = {k: v for k, v in train['coefficients'].items() if k != 'const'}
    feat_names = list(coefs.keys())
    feat_vals = list(coefs.values())
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in feat_vals]
    axes[1, 0].barh(feat_names, feat_vals, color=colors)
    axes[1, 0].axvline(x=0, color='black', ls='-', alpha=0.3)
    axes[1, 0].set_title('Feature Coefficients (Elasticities)')

    # 4. Actual vs Predicted over time
    if 'Date' in feature_matrix.columns:
        dates = feature_matrix['Date'].values[:len(predictions)]
        axes[1, 1].plot(dates, y_actual, 'b-o', label='Actual', linewidth=2)
        axes[1, 1].plot(dates, predictions, 'r--s', label='Predicted', linewidth=2)
        axes[1, 1].set_title('Actual vs Predicted Over Time')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].plot(y_actual, 'b-o', label='Actual', linewidth=2)
        axes[1, 1].plot(predictions, 'r--s', label='Predicted', linewidth=2)
        axes[1, 1].set_title('Actual vs Predicted (Index)')
        axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'best_model_diagnostics.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_scenario_results(scenario_results, save_dir=None):
    """Plot scenario simulation results."""
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    if not scenario_results:
        return

    names = [s['scenario_name'] for s in scenario_results]
    changes = [s['change_pct'] for s in scenario_results]
    colors = ['#4CAF50' if c >= 0 else '#F44336' for c in changes]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(names, changes, color=colors)
    ax.axvline(x=0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel('GMV Change %')
    ax.set_title('Scenario Simulation Results (vs Baseline)')

    for i, (name, change) in enumerate(zip(names, changes)):
        ax.text(change + (0.5 if change >= 0 else -0.5), i,
                f'{change:+.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'scenarios.png'), dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 9. MASTER MODELING PIPELINE
# =============================================================================

def run_modeling_pipeline(fe_result, save_dir=None):
    """
    Run the complete modeling pipeline.

    Args:
        fe_result: output from run_feature_engineering()
        save_dir: where to save plots

    Returns:
        {
            "ranked_models": list of all models ranked,
            "top_10": top 10 models,
            "best_model": single best model detail,
            "convergence": insight convergence analysis,
            "scenarios": scenario simulation results,
            "full_log": list of all actions,
            "summaries": dict of step summaries,
            "decisions": list of all decisions
        }
    """
    if save_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, 'outputs', 'plots')

    print("=" * 70)
    print("[START] MODELING PIPELINE")
    print("=" * 70)

    full_log = []
    summaries = {}
    all_decisions = []

    feature_matrix = fe_result['data']
    feature_sets = fe_result['feature_sets']

    specs = get_feature_specs()
    model_types = get_model_types()
    transforms = get_transform_variants()

    # --- Build all model combinations ---
    print(f"\n[STEP 1] Building Model Combinations...")
    print(f"  Specs: {len(specs)} | Types: {len(model_types)} | Transforms: {len(transforms)}")
    print(f"  Total candidates: {len(specs) * len(model_types) * len(transforms)}")

    all_results = []
    model_count = 0
    failed_count = 0

    for spec_name, spec in specs.items():
        for type_name, type_config in model_types.items():
            for trans_name, trans_config in transforms.items():
                model_count += 1
                target = trans_config['target']
                use_log = trans_config['use_log_features']

                # Resolve features based on transform variant
                # For log_linear: swap log_ features to raw equivalents
                features = []
                for f in spec['features']:
                    if not use_log and f.startswith('log_'):
                        # Try to find the raw version
                        raw_name = f.replace('log_', '')
                        if raw_name in feature_matrix.columns:
                            features.append(raw_name)
                        elif f in feature_matrix.columns:
                            features.append(f)  # fallback to log if raw not found
                        else:
                            features.append(f)
                    else:
                        features.append(f)

                # Check if all columns exist
                required = features + [target]
                missing = [c for c in required if c not in feature_matrix.columns]
                if missing:
                    failed_count += 1
                    continue

                # Prepare data -- drop NaN rows for this specific combination
                model_df = feature_matrix[required].dropna()
                if len(model_df) < len(features) + 2:
                    failed_count += 1
                    full_log.append(f"[SKIP] {spec_name}|{type_name}|{trans_name}: "
                                    f"only {len(model_df)} rows after NaN drop")
                    continue

                X = model_df[features]
                y = model_df[target]

                # Train
                builder = type_config['builder']
                train_result = builder(X.values, y.values, features)

                if not train_result.get('success', False):
                    failed_count += 1
                    continue

                # Ordinality check
                ord_result = check_ordinality(train_result['coefficients'], features)

                # Cross-validation
                cv_result = loo_cross_validation(X, y, type_name, builder)

                # Score
                scores = score_model(train_result, cv_result, ord_result)

                result = {
                    "spec_name": spec_name,
                    "spec_config": {**spec, "resolved_features": features},
                    "model_type": type_name,
                    "transform": trans_name,
                    "transform_config": trans_config,
                    "train_result": train_result,
                    "ordinality": ord_result,
                    "cv_result": cv_result,
                    "scores": scores,
                    "n_observations": len(model_df)
                }
                all_results.append(result)

    full_log.append(f"Built {len(all_results)} successful models out of {model_count} candidates "
                    f"({failed_count} failed/skipped)")
    print(f"  [OK] {len(all_results)} models built successfully, {failed_count} skipped")

    # --- Rank models ---
    print(f"\n[STEP 2] Ranking Models...")
    ranked = rank_models(all_results)

    top_10 = ranked[:10]
    print(f"\n  {'Rank':<5} {'Spec':<30} {'Type':<10} {'Transform':<12} "
          f"{'Composite':<10} {'Fit':<8} {'Stable':<8} {'Ordinal':<8}")
    print("  " + "-" * 95)
    for r in top_10:
        s = r['scores']
        ord_tag = "PASS" if r['ordinality']['passed'] else "FAIL"
        print(f"  {r['rank']:<5} {r['spec_name'][:28]:<30} {r['model_type']:<10} "
              f"{r['transform']:<12} {s['composite']:<10.4f} {s['fit_score']:<8.4f} "
              f"{s['stability_score']:<8.4f} {ord_tag:<8}")

    summaries['ranking'] = (
        f"Ranked {len(ranked)} models. Top model: {ranked[0]['spec_name']} "
        f"with {ranked[0]['model_type']} ({ranked[0]['transform']}) -- "
        f"composite score {ranked[0]['scores']['composite']:.3f}."
    )

    # --- Convergence analysis ---
    print(f"\n[STEP 3] Insight Convergence Analysis...")
    convergence = analyze_convergence(ranked)
    full_log.extend(convergence['log'])
    summaries['convergence'] = convergence['summary']
    print(f"  {convergence['summary']}")

    if convergence['insights']:
        print(f"\n  {'Feature':<40} {'Mean Coef':<12} {'Direction':<25} {'Models':<8}")
        print("  " + "-" * 85)
        for feat, info in sorted(convergence['insights'].items(),
                                  key=lambda x: abs(x[1]['mean_coefficient']), reverse=True):
            print(f"  {feat[:38]:<40} {info['mean_coefficient']:<12.4f} "
                  f"{info['direction']:<25} {info['n_models']:<8}")

    # --- Best model detail ---
    best = ranked[0]
    print(f"\n[STEP 4] Best Model Detail...")
    print(f"  Spec:        {best['spec_name']}")
    print(f"  Type:        {best['model_type']}")
    print(f"  Transform:   {best['transform']}")
    print(f"  R-squared:   {best['train_result']['r_squared']:.4f}")
    print(f"  Adj R-sq:    {best['train_result']['adj_r_squared']:.4f}")
    if best['cv_result'].get('cv_r2') is not None:
        print(f"  CV R-sq:     {best['cv_result']['cv_r2']:.4f}")
        print(f"  CV MAPE:     {best['cv_result']['cv_mape']:.2f}%")
    print(f"  Ordinality:  {'PASS' if best['ordinality']['passed'] else 'FAIL'}")

    print(f"\n  Elasticities (Coefficients):")
    for feat, coef in best['train_result']['coefficients'].items():
        if feat == 'const':
            continue
        pval = best['train_result'].get('pvalues', {}).get(feat, None)
        pval_str = f"p={pval:.3f}" if pval is not None else "p=N/A"
        std_coef = best['train_result'].get('standardized_coefficients', {}).get(feat, None)
        std_str = f"std_coef={std_coef:+.4f}" if std_coef is not None else ""
        print(f"    {feat:35s} = {coef:+.4f}  ({pval_str})  {std_str}")

    # --- Scenario simulation ---
    print(f"\n[STEP 5] Scenario Simulation...")
    target_col = best['transform_config']['target']
    feature_names = best['spec_config']['resolved_features']
    simulator = build_scenario_simulator(best, feature_matrix, feature_names, target_col)
    scenario_results = run_standard_scenarios(simulator)

    if scenario_results:
        print(f"\n  {'Scenario':<35} {'Baseline GMV':<15} {'New GMV':<15} {'Change':<10}")
        print("  " + "-" * 75)
        for s in scenario_results:
            print(f"  {s['scenario_name']:<35} {s['baseline_gmv']/1e7:<15.2f} "
                  f"{s['predicted_gmv']/1e7:<15.2f} {s['change_pct']:+.1f}%")

    summaries['scenarios'] = (
        f"Ran {len(scenario_results)} scenarios using best model. "
        f"Key finding: {scenario_results[1]['scenario_name'] if len(scenario_results) > 1 else 'N/A'} "
        f"yields {scenario_results[1]['change_pct']:+.1f}% GMV change."
        if scenario_results else "No scenarios could be run."
    )

    # --- Visualizations ---
    print(f"\n[STEP 6] Generating Plots...")
    plot_model_rankings(ranked, top_n=10, save_dir=save_dir)
    plot_best_model_diagnostics(best, feature_matrix, save_dir=save_dir)
    plot_scenario_results(scenario_results, save_dir=save_dir)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("[DONE] MODELING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Models built:     {len(ranked)}")
    print(f"  Best model:       {best['spec_name']} | {best['model_type']} | {best['transform']}")
    print(f"  Best composite:   {best['scores']['composite']:.4f}")
    print(f"  Scenarios tested: {len(scenario_results)}")

    return {
        "ranked_models": ranked,
        "top_10": top_10,
        "best_model": best,
        "convergence": convergence,
        "scenarios": scenario_results,
        "simulator": simulator,
        "full_log": full_log,
        "summaries": summaries,
        "decisions": all_decisions
    }


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    from eda_pipeline import load_all_data
    from outlier_detection import run_outlier_pipeline
    from feature_engineering import run_feature_engineering

    data = load_all_data()
    clean_data, outlier_log, assumptions = run_outlier_pipeline(data)
    fe_result = run_feature_engineering(clean_data)
    model_result = run_modeling_pipeline(fe_result)
