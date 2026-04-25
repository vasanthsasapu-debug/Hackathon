"""
=============================================================================
modeling_engine.py -- Model Building, Ranking, Scenarios
=============================================================================
Modules:
  1. Model Registry (specs, types, transforms)
  2. Model Builders (OLS, Ridge, Lasso, ElasticNet, Bayesian, XGBoost, RF)
  3. Ordinality Enforcement
  4. Cross-Validation (LOO)
  5. Scorer & Ranker
  6. Insight Convergence
  7. Scenario Simulator (dynamic, multi-model)
  8. Diagnostics & Visualization
  9. Master Pipeline
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

from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm

from config import (
    MEDIA_CHANNELS, LOG_TO_RAW_MAP, CHANNEL_GROUPS, MODEL_SETTINGS,
    get_paths, get_channel_cols, find_col, logger
)

plt.rcParams["figure.figsize"] = (14, 6)
sns.set_style("whitegrid")


# =============================================================================
# 1. MODEL REGISTRY
# =============================================================================

def get_feature_specs():
    """Fallback specs if feature_engineering doesn't provide them."""
    return {
        "spec_B_total_spend": {
            "features": ["log_Total_Investment", "sale_flag", "nps_standardized"],
            "description": "Total investment + sale + NPS",
        },
        "spec_E_discount_effect": {
            "features": ["log_Total_Investment", "discount_intensity", "sale_flag"],
            "description": "Total spend + discount + sale",
        },
        "spec_H_sale_duration": {
            "features": ["log_Total_Investment", "sale_days", "nps_standardized"],
            "description": "Total spend + sale duration + NPS",
        },
    }


def get_model_types(model_filter="all"):
    """
    All supported model types, filtered by argument.
    
    Args:
        model_filter: 'all', 'linear', or comma-separated names
    
    Note: XGBoost and RandomForest were removed because tree-based models
    don't produce interpretable elasticities (coefficients), which are
    essential for MMIX business recommendations. The 6 linear models
    provide full coefficient interpretability and handle multicollinearity
    via regularization.
    """
    all_types = {
        "OLS":         {"description": "OLS -- baseline, p-values",       "builder": build_ols},
        "Ridge":       {"description": "Ridge -- L2, stable",            "builder": build_ridge},
        "Lasso":       {"description": "Lasso -- L1, feature selection", "builder": build_lasso},
        "ElasticNet":  {"description": "ElasticNet -- L1+L2 combined",   "builder": build_elasticnet},
        "Bayesian":    {"description": "Bayesian Ridge -- uncertainty",  "builder": build_bayesian},
        "Huber":       {"description": "Huber -- robust to outliers",    "builder": build_huber},
    }

    if model_filter == "all" or model_filter == "linear":
        return all_types
    else:
        # Comma-separated list
        selected = [m.strip() for m in model_filter.split(",")]
        filtered = {k: v for k, v in all_types.items() if k in selected}
        if not filtered:
            logger.warning("No valid models in '%s'. Using all.", model_filter)
            return all_types
        return filtered


def get_transform_variants():
    """Target/feature transform combinations."""
    return {
        "log_log":    {"target": "log_total_gmv", "use_log_features": True,
                       "description": "Log-Log: elasticity"},
        "log_linear": {"target": "log_total_gmv", "use_log_features": False,
                       "description": "Log-Linear: raw spend, log target"},
    }


# =============================================================================
# 2. MODEL BUILDERS
# =============================================================================

def build_ols(X, y, feature_names):
    """OLS with NaN/Inf protection and standardized coefficients."""
    try:
        X_c = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_c).fit()

        # Check for degenerate results
        if np.any(np.isnan(model.params)) or np.any(np.isinf(model.params)):
            return {"success": False, "error": "OLS produced NaN/Inf coefficients"}

        coefs = dict(zip(["const"] + feature_names, model.params))
        pvals = dict(zip(["const"] + feature_names, model.pvalues))

        # Standardized coefficients for importance comparison
        std_coefs = {}
        try:
            sc = StandardScaler()
            X_s = sm.add_constant(sc.fit_transform(X), has_constant="add")
            std_coefs = dict(zip(["const"] + feature_names, sm.OLS(y, X_s).fit().params))
        except Exception:
            pass

        return {
            "model": model, "scaler": None, "coefficients": coefs,
            "standardized_coefficients": std_coefs, "pvalues": pvals,
            "r_squared": model.rsquared, "adj_r_squared": model.rsquared_adj,
            "aic": model.aic, "bic": model.bic,
            "predictions": np.asarray(model.fittedvalues),
            "residuals": np.asarray(model.resid), "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_ridge(X, y, feature_names):
    """Ridge with L2 regularization."""
    try:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = Ridge(alpha=1.0)
        m.fit(Xs, y)
        p = m.predict(Xs)
        coefs = dict(zip(feature_names, m.coef_))
        coefs["const"] = m.intercept_
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_lasso(X, y, feature_names):
    """Lasso with CV-tuned alpha."""
    try:
        from sklearn.linear_model import LassoCV
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = LassoCV(cv=min(5, len(y) - 1), random_state=42, max_iter=10000)
        m.fit(Xs, y)
        p = m.predict(Xs)
        coefs = dict(zip(feature_names, m.coef_))
        coefs["const"] = m.intercept_
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p,
                "alpha": m.alpha_, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_elasticnet(X, y, feature_names):
    """ElasticNet with CV-tuned alpha and l1_ratio."""
    try:
        from sklearn.linear_model import ElasticNetCV
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                         cv=min(5, len(y) - 1), random_state=42, max_iter=10000)
        m.fit(Xs, y)
        p = m.predict(Xs)
        coefs = dict(zip(feature_names, m.coef_))
        coefs["const"] = m.intercept_
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p,
                "alpha": m.alpha_, "l1_ratio": m.l1_ratio_, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_bayesian(X, y, feature_names):
    """Bayesian Ridge with uncertainty."""
    try:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = BayesianRidge()
        m.fit(Xs, y)
        p = m.predict(Xs)
        coefs = dict(zip(feature_names, m.coef_))
        coefs["const"] = m.intercept_
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_huber(X, y, feature_names):
    """Huber regression -- robust to outliers, downweights high-residual points."""
    try:
        from sklearn.linear_model import HuberRegressor
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = HuberRegressor(epsilon=1.35, max_iter=200)
        m.fit(Xs, y)
        p = m.predict(Xs)
        coefs = dict(zip(feature_names, m.coef_))
        coefs["const"] = m.intercept_
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_xgboost(X, y, feature_names):
    """XGBoost with conservative hyperparams."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {"success": False, "error": "xgboost not installed"}
    try:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                         min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0)
        m.fit(Xs, y)
        p = m.predict(Xs)
        imp = dict(zip(feature_names, m.feature_importances_))
        coefs = _signed_importance(X, y, feature_names, imp)
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs,
                "feature_importances": imp, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_random_forest(X, y, feature_names):
    """Random Forest ensemble."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_split=5,
                                  min_samples_leaf=3, max_features="sqrt",
                                  random_state=42, n_jobs=-1)
        m.fit(Xs, y)
        p = m.predict(Xs)
        imp = dict(zip(feature_names, m.feature_importances_))
        coefs = _signed_importance(X, y, feature_names, imp)
        r2, adj = _compute_r2(y, p, X.shape)
        return {"model": m, "scaler": sc, "coefficients": coefs,
                "feature_importances": imp, "pvalues": {},
                "r_squared": r2, "adj_r_squared": adj,
                "predictions": p, "residuals": y - p, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# 3. ORDINALITY
# =============================================================================

def check_ordinality(coefficients, feature_names, feature_correlations=None):
    """
    Check that coefficients have the expected sign based on business logic.

    Only UNAMBIGUOUS features are constrained:
      - sale_flag, sale_days: MUST be >= 0 (more sales = more GMV, always)
      - log_Total_Investment / Total.Investment: MUST be >= 0 (aggregate spend)

    Channel groups and individual channels are NOT constrained because:
      - Multicollinearity can flip signs even for positively-correlated channels
      - Convergence analysis shows some groups (e.g. digital_brand) are consistently
        negative when controlling for other spend — this is a real signal, not an error
      - Regularized models (Ridge/Lasso) produce reliable signs even with collinearity

    Other features (NPS, discount, lag) have ambiguous expected direction
    and are not constrained.

    Args:
        coefficients:         dict of {feature: coef}
        feature_names:        list of feature names
        feature_correlations: dict of {feature: corr_with_gmv} (for logging only)
    """
    violations, checks = [], []
    corr = feature_correlations or {}

    # Only these features have unambiguous expected positive direction
    MUST_POSITIVE = ["sale_flag", "sale_days", "sale_intensity",
                     "log_total_investment", "total.investment"]

    for f in feature_names:
        fl = f.lower()

        must_be_positive = any(k in fl for k in MUST_POSITIVE)

        if must_be_positive:
            c = coefficients.get(f, 0)
            ok = c >= 0
            checks.append({"feature": f, "coefficient": c,
                          "expected": "positive (unambiguous)", "passed": ok})
            if not ok:
                violations.append(f"{f} = {c:.4f} (expected positive)")
        else:
            # Log but don't enforce — channel groups, NPS, discount, lag
            c = coefficients.get(f, 0)
            feat_corr = corr.get(f)
            corr_note = f"corr={feat_corr:+.2f}" if feat_corr is not None else "no constraint"
            checks.append({"feature": f, "coefficient": c,
                          "expected": f"any ({corr_note})", "passed": True})

    return {"passed": len(violations) == 0, "violations": violations,
            "checks": checks, "n_violations": len(violations)}


# =============================================================================
# 4. CROSS-VALIDATION
# =============================================================================

def loo_cross_validation(X, y, type_name, builder):
    """Leave-One-Out CV. Returns cv_r2, cv_rmse, cv_mape."""
    loo = LeaveOneOut()
    yt, yp = [], []
    fnames = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
    Xa = X.values if hasattr(X, "values") else X
    ya = y.values if hasattr(y, "values") else y

    for tr, te in loo.split(Xa):
        try:
            res = builder(Xa[tr], ya[tr], fnames)
            if not res.get("success"):
                continue
            if type_name == "OLS":
                pred = res["model"].predict(sm.add_constant(Xa[te], has_constant="add"))
            else:
                sc = res.get("scaler")
                pred = res["model"].predict(sc.transform(Xa[te])) if sc else res["model"].predict(Xa[te])
            yt.append(ya[te][0])
            yp.append(pred[0])
        except Exception:
            continue

    if len(yt) < 3:
        return {"cv_r2": None, "cv_rmse": None, "cv_mape": None}
    yt, yp = np.array(yt), np.array(yp)
    return {
        "cv_r2": r2_score(yt, yp),
        "cv_rmse": np.sqrt(mean_squared_error(yt, yp)),
        "cv_mape": mean_absolute_percentage_error(yt, yp) * 100,
    }


# =============================================================================
# 5. SCORER & RANKER
# =============================================================================

def score_model(train, cv, ordin, vif_max=None):
    """Composite: 35% fit + 25% stability + 25% ordinality + 15% VIF."""
    if not train.get("success"):
        return {"composite": 0, "fit_score": 0, "stability_score": 0,
                "ordinality_score": 0, "vif_score": 0, "vif_max": None}
    w = MODEL_SETTINGS["score_weights"]
    fit = max(0, min(1, train.get("adj_r_squared", 0)))
    tr2 = train.get("r_squared", 0)
    cv2 = cv.get("cv_r2", 0)
    stab = max(0, 1 - abs(tr2 - cv2) / tr2) if cv2 and tr2 > 0 else 0
    ordscore = 1.0 if ordin["passed"] else max(0, 1 - ordin["n_violations"] / max(len(ordin["checks"]), 1))
    if vif_max is not None:
        vs = 1.0 if vif_max <= 5 else 0.7 if vif_max <= 10 else 0.3 if vif_max <= 50 else 0.1
    else:
        vs = 0.5
    comp = w["fit"] * fit + w["stability"] * stab + w["ordinality"] * ordscore + w["vif"] * vs
    return {"composite": round(comp, 4), "fit_score": round(fit, 4),
            "stability_score": round(stab, 4), "ordinality_score": round(ordscore, 4),
            "vif_score": round(vs, 4), "vif_max": round(vif_max, 2) if vif_max else None}


def rank_models(results):
    """Sort by composite score descending."""
    ranked = sorted(results, key=lambda x: x["scores"]["composite"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


# =============================================================================
# 6. INSIGHT CONVERGENCE
# =============================================================================

def analyze_convergence(results):
    """Which findings hold across all models?"""
    ok = [r for r in results if r["train_result"].get("success")]
    if not ok:
        return {"insights": {}, "log": [], "summary": "No models."}
    fc = {}
    for r in ok:
        for f, v in r["train_result"].get("coefficients", {}).items():
            if f == "const":
                continue
            fc.setdefault(f, []).append(v)
    insights, log = {}, []
    for f, vals in fc.items():
        if len(vals) < 2:
            continue
        npos = sum(1 for v in vals if v > 0)
        pct = npos / len(vals) * 100
        d = "POSITIVE (confirmed)" if pct >= 80 else "NEGATIVE (confirmed)" if pct <= 20 else "MIXED (inconclusive)"
        insights[f] = {"mean_coefficient": round(np.mean(vals), 4),
                        "n_models": len(vals), "pct_positive": round(pct, 1), "direction": d}
        log.append(f"{f}: mean={np.mean(vals):.4f}, {d}")
    conf = sum(1 for v in insights.values() if "confirmed" in v["direction"].lower())
    return {"insights": insights, "log": log,
            "summary": f"Convergence across {len(ok)} models. {len(insights)} features, {conf} confirmed."}


# =============================================================================
# 7. SCENARIO SIMULATOR
# =============================================================================

def build_scenario_simulator(model_result, feature_matrix, feature_names, target_col,
                             full_monthly_data=None):
    """
    Build simulator that accepts raw channel names and translates
    to whatever features the model uses.

    Uses a DUAL-BASELINE approach:
      - Channel scenarios: baseline = overall feature means (standard)
      - Sale scenarios: baseline = non-sale period means, so the delta
        (0 → sale_days=3) is real rather than (mean≈3 → 3) which shows 0% change.
        This correctly measures "what happens if we activate a sale in a
        month that wouldn't otherwise have one."
    """
    if not model_result["train_result"].get("success"):
        return None

    model = model_result["train_result"]["model"]
    mtype = model_result["model_type"]
    scaler = model_result["train_result"].get("scaler")
    baseline = feature_matrix[feature_names].mean().to_dict()
    baseline_target = feature_matrix[target_col].mean()

    # --- Compute non-sale baseline for sale scenarios ---
    # When simulating "Activate sale event", the counterfactual is a month
    # WITHOUT a sale. Using the overall mean as baseline includes sale months,
    # making sale_days baseline ≈ 3, so setting sale_days=3 shows 0% change.
    # The non-sale baseline sets sale features to their non-sale-period values.
    SALE_FEATURES = {"sale_flag", "sale_days", "sale_intensity"}
    sale_features_in_model = [f for f in feature_names if f in SALE_FEATURES]

    non_sale_baseline = baseline.copy()
    if sale_features_in_model:
        # Try to find non-sale rows
        sale_flag_col = "sale_flag" if "sale_flag" in feature_matrix.columns else None
        if sale_flag_col:
            non_sale_mask = feature_matrix[sale_flag_col] == 0
            n_non_sale = non_sale_mask.sum()
            if n_non_sale >= 2:
                # Use mean of non-sale periods for ALL features (not just sale features)
                # This gives a realistic "typical non-sale month" as baseline
                non_sale_baseline = feature_matrix.loc[non_sale_mask, feature_names].mean().to_dict()
                logger.info("  Non-sale baseline: %d non-sale periods found (of %d total)",
                            n_non_sale, len(feature_matrix))
            else:
                # Fallback: just zero out sale features in the overall baseline
                for sf in sale_features_in_model:
                    non_sale_baseline[sf] = 0.0
                logger.info("  Non-sale baseline: <2 non-sale periods, zeroing sale features")
        else:
            # No sale_flag column — zero out any sale features
            for sf in sale_features_in_model:
                non_sale_baseline[sf] = 0.0
            logger.info("  Non-sale baseline: no sale_flag column, zeroing sale features")

        # Log the difference for transparency
        for sf in sale_features_in_model:
            logger.info("    %s: overall_mean=%.2f, non_sale_baseline=%.2f",
                        sf, baseline.get(sf, 0), non_sale_baseline.get(sf, 0))

    # Raw channel means
    raw_means = {}
    for ch in MEDIA_CHANNELS:
        for src in [feature_matrix, full_monthly_data]:
            if src is not None and ch in src.columns:
                v = pd.to_numeric(src[ch], errors="coerce").mean()
                if not np.isnan(v):
                    raw_means[ch] = v
                    break
    total_mean = sum(raw_means.values()) if raw_means else 0

    # Baseline prediction (using overall mean — channel scenarios use this)
    Xb = np.array([[baseline[f] for f in feature_names]])
    try:
        if mtype == "OLS":
            bp = model.predict(sm.add_constant(Xb, has_constant="add"))[0]
        else:
            bp = model.predict(scaler.transform(Xb))[0] if scaler else model.predict(Xb)[0]
    except Exception:
        bp = baseline_target
    base_gmv = np.expm1(bp) if target_col.startswith("log_") else bp

    # Non-sale baseline prediction (sale scenarios measure change FROM this)
    Xns = np.array([[non_sale_baseline[f] for f in feature_names]])
    try:
        if mtype == "OLS":
            ns_bp = model.predict(sm.add_constant(Xns, has_constant="add"))[0]
        else:
            ns_bp = model.predict(scaler.transform(Xns))[0] if scaler else model.predict(Xns)[0]
    except Exception:
        ns_bp = bp
    non_sale_base_gmv = np.expm1(ns_bp) if target_col.startswith("log_") else ns_bp

    logger.info("  Scenario simulator: features=%s, channels=%d", feature_names, len(raw_means))
    logger.info("  Overall baseline=%.2f Cr, Non-sale baseline=%.2f Cr",
                base_gmv / 1e7, non_sale_base_gmv / 1e7)
    if raw_means:
        logger.info("  Raw channel means: %s", {k: f"{v:.0f}" for k, v in raw_means.items()})
    else:
        logger.warning("  WARNING: No raw channel means — scenarios will show 0%% change")

    def simulate(changes):
        """
        Args:
            changes: raw channel multipliers and/or binary overrides.
                     {"Online.marketing": 1.2, "TV": 0.9, "sale_flag": 1}
        Returns:
            {"baseline_gmv", "predicted_gmv", "change_pct", "scenario"}

        When changes include sale features, uses the non-sale baseline as
        starting point so the delta is meaningful (e.g. sale_days: 0→3
        instead of 3→3).
        """
        # Determine if this is a sale scenario
        has_sale_changes = any(f in SALE_FEATURES for f in changes)
        has_channel_changes = any(f not in SALE_FEATURES for f in changes)

        # Pick the right baseline:
        # - Pure sale scenario → non-sale baseline (measure sale activation impact)
        # - Pure channel scenario → overall baseline (measure spend change impact)
        # - Mixed (channel + sale) → non-sale baseline (measure combined impact)
        if has_sale_changes:
            sc = non_sale_baseline.copy()
            ref_gmv = non_sale_base_gmv
        else:
            sc = baseline.copy()
            ref_gmv = base_gmv

        ch_mults = {}

        for feat, val in changes.items():
            if feat in SALE_FEATURES:
                if feat in sc:
                    sc[feat] = val
            elif feat in raw_means:
                ch_mults[feat] = val
            elif feat in sc:
                _apply_mult(sc, feat, val)

        # Track which channels actually affected a model feature
        modeled_channels = set()
        note = None

        if ch_mults and total_mean > 0:
            new_total = sum(raw_means.get(c, 0) * ch_mults.get(c, 1.0)
                            for c in raw_means)
            tmult = new_total / total_mean

            # Use the appropriate baseline for channel feature adjustments
            feat_baseline = non_sale_baseline if has_sale_changes else baseline

            for feat in feature_names:
                if feat in SALE_FEATURES:
                    continue  # Already handled above
                if feat == "log_Total_Investment" or feat == "Total.Investment":
                    # All channels route through total investment
                    if feat == "log_Total_Investment":
                        sc[feat] = np.log1p(np.expm1(feat_baseline[feat]) * tmult)
                    else:
                        sc[feat] = feat_baseline[feat] * tmult
                    modeled_channels.update(ch_mults.keys())
                elif feat.startswith("log_spend_"):
                    grp = feat.replace("log_spend_", "")
                    if grp in CHANNEL_GROUPS:
                        chs = CHANNEL_GROUPS[grp]
                        og = sum(raw_means.get(c, 0) for c in chs)
                        ng = sum(raw_means.get(c, 0) * ch_mults.get(c, 1.0) for c in chs)
                        if og > 0:
                            sc[feat] = np.log1p(np.expm1(feat_baseline[feat]) * ng / og)
                            # Channels in this group that were changed are modeled
                            modeled_channels.update(c for c in chs if c in ch_mults)
                elif feat.startswith("spend_") and feat.replace("spend_", "") in CHANNEL_GROUPS:
                    grp = feat.replace("spend_", "")
                    chs = CHANNEL_GROUPS[grp]
                    og = sum(raw_means.get(c, 0) for c in chs)
                    ng = sum(raw_means.get(c, 0) * ch_mults.get(c, 1.0) for c in chs)
                    if og > 0:
                        sc[feat] = feat_baseline[feat] * (ng / og)
                        modeled_channels.update(c for c in chs if c in ch_mults)
                elif feat in LOG_TO_RAW_MAP:
                    rn = LOG_TO_RAW_MAP[feat]
                    if rn in ch_mults:
                        sc[feat] = np.log1p(np.expm1(feat_baseline[feat]) * ch_mults[rn])
                        modeled_channels.add(rn)
                elif feat in ch_mults:
                    sc[feat] = feat_baseline[feat] * ch_mults[feat]
                    modeled_channels.add(feat)

            # Identify channels that were changed but couldn't be routed
            unmodeled = set(ch_mults.keys()) - modeled_channels
            if unmodeled:
                note = (f"Channel(s) {', '.join(sorted(unmodeled))} not in model features "
                        f"— impact cannot be estimated. Model uses: "
                        f"{', '.join(f for f in feature_names if f not in SALE_FEATURES)}")

        Xn = np.array([[sc[f] for f in feature_names]])
        try:
            if mtype == "OLS":
                pred = model.predict(sm.add_constant(Xn, has_constant="add"))[0]
            else:
                pred = model.predict(scaler.transform(Xn))[0] if scaler else model.predict(Xn)[0]
        except Exception:
            pred = ns_bp if has_sale_changes else bp

        pgmv = np.expm1(pred) if target_col.startswith("log_") else pred
        chg = (pgmv / ref_gmv - 1) * 100 if ref_gmv != 0 else 0
        result = {"baseline_gmv": ref_gmv, "predicted_gmv": pgmv,
                  "change_pct": chg, "scenario": changes}
        if note:
            result["note"] = note
        return result

    return simulate


def run_standard_scenarios(simulator):
    """Standard set of business scenarios."""
    if simulator is None:
        return []
    scenarios = [
        {"name": "Baseline (no change)", "changes": {}},
        {"name": "+20% Online.marketing", "changes": {"Online.marketing": 1.2}},
        {"name": "+20% Sponsorship", "changes": {"Sponsorship": 1.2}},
        {"name": "+20% TV", "changes": {"TV": 1.2}},
        {"name": "+20% all digital (Onl+Aff+SEM)",
         "changes": {"Online.marketing": 1.2, "Affiliates": 1.2, "SEM": 1.2}},
        {"name": "+20% all channels", "changes": {ch: 1.2 for ch in MEDIA_CHANNELS}},
        {"name": "-10% all channels", "changes": {ch: 0.9 for ch in MEDIA_CHANNELS}},
        {"name": "+50% digital performance",
         "changes": {"Online.marketing": 1.5, "Affiliates": 1.5, "SEM": 1.5}},
        {"name": "Activate sale event (vs non-sale month)",
         "changes": {"sale_flag": 1, "sale_days": 3, "sale_intensity": 1}},
        {"name": "Sale event (4 days)", "changes": {"sale_flag": 1, "sale_days": 4, "sale_intensity": 1}},
        {"name": "Sale event (6 days)", "changes": {"sale_flag": 1, "sale_days": 6, "sale_intensity": 1}},
        {"name": "Shift 10% TV -> Online.mkt", "changes": {"TV": 0.9, "Online.marketing": 1.1}},
        {"name": "+10% Sponsorship + sale (vs non-sale)",
         "changes": {"Sponsorship": 1.1, "sale_flag": 1, "sale_days": 3, "sale_intensity": 1}},
    ]
    results = []
    for s in scenarios:
        try:
            r = simulator(s["changes"])
            r["scenario_name"] = s["name"]
            results.append(r)
        except Exception:
            continue
    return results


def run_custom_scenario(simulator, channel_changes, sale_flag=None):
    """
    Dynamic scenario with percentage inputs.
    channel_changes: {"Online.marketing": 20, "TV": -10}
    """
    if simulator is None:
        return None
    changes = {ch: 1 + pct / 100 for ch, pct in channel_changes.items()}
    if sale_flag is not None:
        changes["sale_flag"] = sale_flag
    r = simulator(changes)
    print(f"\n  Custom Scenario:")
    for ch, pct in channel_changes.items():
        print(f"    {ch}: {pct:+.0f}%")
    if sale_flag is not None:
        print(f"    sale_flag: {sale_flag}")
    print(f"  Baseline: {r['baseline_gmv'] / 1e7:.2f} Cr")
    print(f"  Predicted: {r['predicted_gmv'] / 1e7:.2f} Cr")
    print(f"  Change: {r['change_pct']:+.1f}%")
    if r.get("note"):
        print(f"  NOTE: {r['note']}")
    return r

def run_interactive_scenarios(simulator):
    """Comprehensive grouped business scenarios."""
    if simulator is None:
        return []
 
    print(f"\n  --- Comprehensive Scenarios ---")
    groups = {
        "Individual Channel Impact (+20% each)": [
            {"name": f"+20% {ch}", "changes": {ch: 1.2}}
            for ch in ["Online.marketing", "Sponsorship", "TV", "SEM", "Affiliates", "Digital"]
        ],
        "Budget Reallocation": [
            {"name": "Shift 10% TV -> Online.mkt",
             "changes": {"TV": 0.9, "Online.marketing": 1.1}},
            {"name": "Shift 10% Sponsorship -> SEM",
             "changes": {"Sponsorship": 0.9, "SEM": 1.1}},
            {"name": "Shift 20% traditional -> digital",
             "changes": {"TV": 0.8, "Sponsorship": 0.8,
                        "Online.marketing": 1.15, "SEM": 1.15, "Affiliates": 1.1}},
        ],
        "Scale Scenarios": [
            {"name": "All channels +10%", "changes": {ch: 1.1 for ch in MEDIA_CHANNELS}},
            {"name": "All channels +20%", "changes": {ch: 1.2 for ch in MEDIA_CHANNELS}},
            {"name": "All channels -20%", "changes": {ch: 0.8 for ch in MEDIA_CHANNELS}},
        ],
        "Promotional Scenarios (vs non-sale baseline)": [
            {"name": "Sale event only (vs non-sale month)",
             "changes": {"sale_flag": 1, "sale_days": 3, "sale_intensity": 1}},
            {"name": "Sale + 10% all (vs non-sale month)",
             "changes": {**{ch: 1.1 for ch in MEDIA_CHANNELS},
                        "sale_flag": 1, "sale_days": 3, "sale_intensity": 1}},
            {"name": "No sale + 20% all",
             "changes": {**{ch: 1.2 for ch in MEDIA_CHANNELS},
                        "sale_flag": 0, "sale_days": 0, "sale_intensity": 0}},
        ],
        "Sale Duration Curve (vs non-sale baseline)": [
            {"name": f"Sale event ({d} days)",
             "changes": {"sale_flag": 1, "sale_days": d, "sale_intensity": 1}}
            for d in [3, 4, 5, 6]
        ],
    }
 
    all_results = []
    for gname, scenarios in groups.items():
        print(f"\n  --- {gname} ---")
        print(f"  {'Scenario':<45} {'Base (Cr)':<12} {'Pred (Cr)':<12} {'Change':<8}")
        print(f"  " + "-" * 80)
        for s in scenarios:
            try:
                r = simulator(s["changes"])
                r["scenario_name"] = s["name"]
                r["group"] = gname
                all_results.append(r)
                note_marker = " *" if r.get("note") else ""
                print(f"  {s['name']:<45} {r['baseline_gmv']/1e7:<12.2f} "
                      f"{r['predicted_gmv']/1e7:<12.2f} {r['change_pct']:+.1f}%{note_marker}")
            except Exception:
                continue
 
    # Print any notes about unmodeled channels
    noted = [r for r in all_results if r.get("note")]
    if noted:
        print(f"\n  * NOTE: {len(noted)} scenario(s) involve channels not in the model:")
        for r in noted:
            print(f"    - {r['scenario_name']}: {r['note']}")

    return all_results

# =============================================================================
# 8. VISUALIZATION
# =============================================================================

def plot_model_rankings(ranked, top_n=10, save_dir=None):
    """Top N model bar chart + score breakdown."""
    save_dir = save_dir or get_paths()["plots_dir"]
    top = ranked[:top_n]
    names = [f"{r['spec_name'][:18]}\n{r['model_type']}\n{r['transform']}" for r in top]
    comp = [r["scores"]["composite"] for r in top]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Model Rankings", fontsize=16, fontweight="bold")

    clrs = ["#4CAF50" if c > 0.7 else "#FF9800" if c > 0.5 else "#F44336" for c in comp]
    axes[0].barh(range(len(top)), comp, color=clrs)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlim(0, 1); axes[0].set_title(f"Top {top_n}")
    axes[0].invert_yaxis()

    t5 = top[:5]; x = range(len(t5)); w = 0.2
    n5 = [f"{r['spec_name'][:12]}\n{r['model_type']}" for r in t5]
    axes[1].bar([i - 1.5 * w for i in x], [r["scores"]["fit_score"] for r in t5], w, label="Fit", color="#2196F3")
    axes[1].bar([i - 0.5 * w for i in x], [r["scores"]["stability_score"] for r in t5], w, label="Stability", color="#4CAF50")
    axes[1].bar([i + 0.5 * w for i in x], [r["scores"]["ordinality_score"] for r in t5], w, label="Ordinality", color="#FF9800")
    axes[1].bar([i + 1.5 * w for i in x], [r["scores"]["vif_score"] for r in t5], w, label="VIF", color="#9C27B0")
    axes[1].set_xticks(x); axes[1].set_xticklabels(n5, fontsize=7)
    axes[1].set_title("Top 5 Breakdown"); axes[1].legend(fontsize=8); axes[1].set_ylim(0, 1.1)

    _save_fig(fig, save_dir, "model_rankings.png")


def plot_best_diagnostics(best, fm, save_dir=None):
    """Actual-vs-predicted, residuals, coefficients, time series."""
    save_dir = save_dir or get_paths()["plots_dir"]
    tr = best["train_result"]
    if not tr.get("success"):
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Best: {best['spec_name']} | {best['model_type']} | {best['transform']}", fontsize=13, fontweight="bold")

    pred = tr["predictions"]
    tgt = best["transform_config"]["target"]
    ya = fm[tgt].values[:len(pred)]

    axes[0, 0].scatter(ya, pred, s=80, color="#9C27B0", edgecolors="white")
    mn, mx = min(ya.min(), pred.min()), max(ya.max(), pred.max())
    axes[0, 0].plot([mn, mx], [mn, mx], "r--", alpha=0.7)
    axes[0, 0].set_title(f"Actual vs Pred (R2={tr['r_squared']:.3f})")

    axes[0, 1].scatter(pred, tr["residuals"], s=80, color="#FF5722", edgecolors="white")
    axes[0, 1].axhline(0, color="black", ls="--", alpha=0.5)
    axes[0, 1].set_title("Residuals")

    coefs = {k: v for k, v in tr["coefficients"].items() if k != "const"}
    axes[1, 0].barh(list(coefs.keys()), list(coefs.values()),
                    color=["#4CAF50" if v >= 0 else "#F44336" for v in coefs.values()])
    axes[1, 0].axvline(0, color="black", ls="-", alpha=0.3)
    axes[1, 0].set_title("Coefficients")

    if "Date" in fm.columns:
        d = fm["Date"].values[:len(pred)]
        axes[1, 1].plot(d, ya, "b-o", label="Actual", lw=2)
        axes[1, 1].plot(d, pred, "r--s", label="Predicted", lw=2)
        axes[1, 1].tick_params(axis="x", rotation=45)
    else:
        axes[1, 1].plot(ya, "b-o", label="Actual", lw=2)
        axes[1, 1].plot(pred, "r--s", label="Predicted", lw=2)
    axes[1, 1].set_title("Over Time"); axes[1, 1].legend()

    _save_fig(fig, save_dir, "best_model_diagnostics.png")


def plot_scenarios(results, save_dir=None):
    """Horizontal bar of scenario GMV changes. Unmodeled channels shown in gray."""
    save_dir = save_dir or get_paths()["plots_dir"]
    if not results:
        return
    names = [s["scenario_name"] for s in results]
    chg = [s["change_pct"] for s in results]
    has_note = [bool(s.get("note")) for s in results]

    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.45)))
    colors = []
    for c, noted in zip(chg, has_note):
        if noted:
            colors.append("#BDBDBD")  # Gray for unmodeled
        elif c >= 0:
            colors.append("#4CAF50")
        else:
            colors.append("#F44336")
    ax.barh(names, chg, color=colors)
    ax.axvline(0, color="black", ls="-", alpha=0.3)
    ax.set_xlabel("GMV Change %"); ax.set_title("Scenarios vs Baseline")
    for i, (c, noted) in enumerate(zip(chg, has_note)):
        label = f"{c:+.1f}%" + (" †" if noted else "")
        ax.text(c + (0.3 if c >= 0 else -0.3), i, label, va="center", fontsize=9)

    # Add legend if any notes exist
    if any(has_note):
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4CAF50", label="Positive impact"),
            Patch(facecolor="#F44336", label="Negative impact"),
            Patch(facecolor="#BDBDBD", label="† Channel(s) not in model"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    _save_fig(fig, save_dir, "scenarios.png")


def _save_fig(fig, save_dir, fname):
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  [OK] Saved %s", fname)


# =============================================================================
# 9. MASTER PIPELINE
# =============================================================================

def run_modeling_pipeline(fe_result, clean_data=None, save_dir=None, top_n_scenarios=1,model_filter="all", skip_scenarios=False):
    """
    Run complete modeling pipeline.

    Args:
        fe_result:       output from run_feature_engineering()
        clean_data:      original clean data dict (for raw channel means)
        save_dir:        plot directory
        top_n_scenarios: how many top models run scenarios (default 1)

    Returns:
        dict with ranked_models, top_10, best_model, convergence,
        scenarios, simulator, run_custom, etc.
    """
    save_dir = save_dir or get_paths()["plots_dir"]

    print("=" * 70)
    print("[START] MODELING PIPELINE")
    print("=" * 70)

    full_log, summaries, decisions = [], {}, []
    fm = fe_result["data"]
    specs = fe_result.get("feature_sets", get_feature_specs())
    types = get_model_types(model_filter)
    transforms = get_transform_variants()

    # --- Pre-compute feature correlations with target for ordinality checks ---
    # This allows the ordinality check to know which channels have negative
    # correlation with GMV (e.g., digital brand) and allow negative coefficients.
    target_col = "log_total_gmv" if "log_total_gmv" in fm.columns else "total_gmv"
    feature_correlations = {}
    if target_col in fm.columns:
        for col in fm.select_dtypes(include=[np.number]).columns:
            if col != target_col:
                try:
                    feature_correlations[col] = fm[col].corr(fm[target_col])
                except Exception:
                    pass
    logger.info("  Computed %d feature-target correlations for ordinality", len(feature_correlations))

    # --- Build all models ---
    total = len(specs) * len(types) * len(transforms)
    print(f"\n[STEP 1] Building Models...")
    print(f"  Specs: {len(specs)} | Types: {len(types)} | Transforms: {len(transforms)} | Total: {total}")

    results = []
    built, failed = 0, 0

    for sn, sp in specs.items():
        for tn, tc in types.items():
            for trn, trc in transforms.items():
                built += 1
                target = trc["target"]
                use_log = trc["use_log_features"]

                # Resolve features
                feats = []
                for f in sp["features"]:
                    if not use_log and f in LOG_TO_RAW_MAP:
                        raw = LOG_TO_RAW_MAP[f]
                        feats.append(raw if raw in fm.columns else f)
                    else:
                        feats.append(f)

                # Validate columns
                missing = [c for c in feats + [target] if c not in fm.columns]
                if missing:
                    failed += 1
                    print(f"    [SKIP] {sn}|{tn}|{trn}: missing cols {missing}")
                    continue

                mdf = fm[feats + [target]].dropna()
                if len(mdf) < len(feats) + 2:
                    failed += 1
                    print(f"    [SKIP] {sn}|{tn}|{trn}: only {len(mdf)} rows for {len(feats)} features")
                    continue

                X, y = mdf[feats], mdf[target]

                try:
                    tr = tc["builder"](X.values, y.values, feats)
                except Exception as e:
                    failed += 1
                    full_log.append(f"[FAIL] {sn}|{tn}|{trn}: {e}")
                    print(f"    [FAIL] {sn}|{tn}|{trn}: EXCEPTION {e}")
                    continue

                if not tr.get("success"):
                    failed += 1
                    full_log.append(f"[FAIL] {sn}|{tn}|{trn}: {tr.get('error', '?')}")
                    print(f"    [FAIL] {sn}|{tn}|{trn}: {tr.get('error', '?')}")
                    continue

                ordin = check_ordinality(tr["coefficients"], feats, feature_correlations)

                try:
                    cv = loo_cross_validation(X, y, tn, tc["builder"])
                except Exception:
                    cv = {"cv_r2": None, "cv_rmse": None, "cv_mape": None}

                vmax = _calc_vif(X, feats)
                scores = score_model(tr, cv, ordin, vif_max=vmax)

                results.append({
                    "spec_name": sn, "spec_config": {**sp, "resolved_features": feats},
                    "model_type": tn, "transform": trn, "transform_config": trc,
                    "train_result": tr, "ordinality": ordin, "cv_result": cv,
                    "scores": scores, "n_observations": len(mdf),
                })

    print(f"  [OK] {len(results)} built, {failed} skipped/failed")

    if not results:
        logger.error("No models succeeded")
        return None

    # --- Rank ---
    print(f"\n[STEP 2] Ranking...")
    ranked = rank_models(results)
    top10 = ranked[:10]

    print(f"\n  {'Rk':<4} {'Spec':<28} {'Type':<12} {'Trn':<12} {'Comp':<8} {'Fit':<7} {'Stab':<7} {'VIF':<8} {'Ord':<5}")
    print("  " + "-" * 100)
    for r in top10:
        s = r["scores"]
        vif = f"{s['vif_max']}" if s.get("vif_max") else "N/A"
        print(f"  {r['rank']:<4} {r['spec_name'][:26]:<28} {r['model_type']:<12} "
              f"{r['transform']:<12} {s['composite']:<8.4f} {s['fit_score']:<7.4f} "
              f"{s['stability_score']:<7.4f} {vif:<8} {'PASS' if r['ordinality']['passed'] else 'FAIL':<5}")

    summaries["ranking"] = f"Ranked {len(ranked)} models. Best: {ranked[0]['spec_name']} ({ranked[0]['model_type']})"

    # --- Convergence ---
    print(f"\n[STEP 3] Convergence...")
    conv = analyze_convergence(ranked)
    print(f"  {conv['summary']}")
    if conv["insights"]:
        print(f"\n  {'Feature':<40} {'Mean':<10} {'Direction':<25} {'N':<5}")
        print("  " + "-" * 80)
        for f, i in sorted(conv["insights"].items(), key=lambda x: abs(x[1]["mean_coefficient"]), reverse=True):
            print(f"  {f[:38]:<40} {i['mean_coefficient']:<10.4f} {i['direction']:<25} {i['n_models']:<5}")

    # --- Convergence-based ordinality re-scoring (Pass 2) ---
    # Use convergence directions as the "truth" for what sign each feature
    # should have. A model that agrees with convergence gets full ordinality
    # score; one that contradicts convergence gets penalized.
    print(f"\n[STEP 3b] Re-scoring ordinality using convergence as truth...")
    convergence_signs = {}
    for feat, info in conv.get("insights", {}).items():
        direction = info.get("direction", "")
        if "POSITIVE (confirmed)" in direction:
            convergence_signs[feat] = "positive"
        elif "NEGATIVE (confirmed)" in direction:
            convergence_signs[feat] = "negative"
        # MIXED features get no constraint

    if convergence_signs:
        rescore_count = 0
        for r in ranked:
            coefs = r["train_result"].get("coefficients", {})
            violations = []
            checks = []

            for feat, expected_sign in convergence_signs.items():
                c = coefs.get(feat)
                if c is None:
                    continue
                if expected_sign == "positive" and c < 0:
                    violations.append(f"{feat} = {c:.4f} (convergence says positive)")
                    checks.append({"feature": feat, "coefficient": c,
                                  "expected": "positive (convergence)", "passed": False})
                elif expected_sign == "negative" and c > 0:
                    violations.append(f"{feat} = {c:.4f} (convergence says negative)")
                    checks.append({"feature": feat, "coefficient": c,
                                  "expected": "negative (convergence)", "passed": False})
                else:
                    checks.append({"feature": feat, "coefficient": c,
                                  "expected": f"{expected_sign} (convergence)", "passed": True})

            # Update ordinality with convergence-based check
            r["ordinality"] = {
                "passed": len(violations) == 0,
                "violations": violations,
                "checks": checks,
                "n_violations": len(violations),
                "method": "convergence-based",
            }

            # Re-score composite with updated ordinality
            vmax = r["scores"].get("vif_max")
            r["scores"] = score_model(
                r["train_result"], r["cv_result"], r["ordinality"], vif_max=vmax
            )
            rescore_count += 1

        # Re-rank
        ranked = rank_models(ranked)
        top10 = ranked[:10]
        print(f"  Re-scored {rescore_count} models using {len(convergence_signs)} convergence-confirmed features")
        print(f"  Convergence truth: {convergence_signs}")

        # Print updated rankings
        print(f"\n  {'Rk':<4} {'Spec':<28} {'Type':<12} {'Trn':<12} {'Comp':<8} {'Fit':<7} {'Stab':<7} {'VIF':<8} {'Ord':<5}")
        print("  " + "-" * 100)
        for r in top10:
            s = r["scores"]
            vif = f"{s['vif_max']}" if s.get("vif_max") else "N/A"
            print(f"  {r['rank']:<4} {r['spec_name'][:26]:<28} {r['model_type']:<12} "
                  f"{r['transform']:<12} {s['composite']:<8.4f} {s['fit_score']:<7.4f} "
                  f"{s['stability_score']:<7.4f} {vif:<8} {'PASS' if r['ordinality']['passed'] else 'FAIL':<5}")
    else:
        print(f"  No confirmed convergence features — keeping original ordinality")

    # --- Best model ---
    best = ranked[0]
    print(f"\n[STEP 4] Best Model...")
    print(f"  {best['spec_name']} | {best['model_type']} | {best['transform']}")
    print(f"  R2={best['train_result']['r_squared']:.4f} | Adj={best['train_result']['adj_r_squared']:.4f}")
    if best["cv_result"].get("cv_r2"):
        print(f"  CV R2={best['cv_result']['cv_r2']:.4f} | MAPE={best['cv_result']['cv_mape']:.2f}%")
    print(f"  Ordinality: {'PASS' if best['ordinality']['passed'] else 'FAIL'}")
    print(f"\n  Coefficients:")
    for f, c in best["train_result"]["coefficients"].items():
        if f == "const":
            continue
        pv = best["train_result"].get("pvalues", {}).get(f)
        ps = f"p={pv:.3f}" if pv is not None else "p=N/A"
        std = best["train_result"].get("standardized_coefficients", {}).get(f)
        ss = f"std={std:+.4f}" if std else ""
        print(f"    {f:35s} = {c:+.4f}  ({ps})  {ss}")

    # --- Scenarios ---
    primary_sim, primary_scen, all_sims, all_scen = None, [], {}, {}
    if not skip_scenarios:
        print(f"\n[STEP 5] Scenarios (top {top_n_scenarios} models)...")
        monthly = clean_data.get("monthly") if clean_data else None

        for i in range(min(top_n_scenarios, len(ranked))):
            m = ranked[i]
            key = f"#{m['rank']} {m['spec_name'][:20]}|{m['model_type']}"
            tc = m["transform_config"]["target"]
            fn = m["spec_config"]["resolved_features"]

            sim = build_scenario_simulator(m, fm, fn, tc, monthly)
            if sim is None:
                continue
            all_sims[key] = sim
            sr = run_standard_scenarios(sim)
            run_interactive_scenarios(sim)
            all_scen[key] = sr

            print(f"\n  --- {key} ---")
            print(f"  {'Scenario':<45} {'Base (Cr)':<12} {'Pred (Cr)':<12} {'Change':<8}")
            print("  " + "-" * 80)
            for s in sr:
                note_marker = " *" if s.get("note") else ""
                print(f"  {s['scenario_name']:<45} {s['baseline_gmv']/1e7:<12.2f} "
                      f"{s['predicted_gmv']/1e7:<12.2f} {s['change_pct']:+.1f}%{note_marker}")
            # Print notes
            noted = [s for s in sr if s.get("note")]
            if noted:
                print(f"\n  * {len(noted)} scenario(s) involve channels not in model:")
                for s in noted:
                    print(f"    {s['scenario_name']}: {s['note']}")

        # Multi-model comparison
        if top_n_scenarios > 1 and len(all_scen) > 1:
            _print_multi_model_comparison(all_scen)

        primary_sim = list(all_sims.values())[0] if all_sims else None
        primary_scen = list(all_scen.values())[0] if all_scen else []
    else:
        print(f"\n[STEP 5] Scenarios SKIPPED (will run after agent approval)")

    # --- Plots ---
    print(f"\n[STEP 6] Plots...")
    try:
        plot_model_rankings(ranked, top_n=10, save_dir=save_dir)
        plot_best_diagnostics(best, fm, save_dir=save_dir)
        if primary_scen:
            plot_scenarios(primary_scen, save_dir=save_dir)
    except Exception as e:
        logger.error("Plotting failed: %s", e)

    print("\n" + "=" * 70)
    print("[DONE] MODELING COMPLETE")
    print("=" * 70)
    print(f"  Models: {len(ranked)} | Best: {best['spec_name']} | {best['model_type']} | "
          f"Composite: {best['scores']['composite']:.4f}")

    return {
        "ranked_models": ranked, "top_10": top10, "best_model": best,
        "convergence": conv, "scenarios": primary_scen,
        "all_scenario_results": all_scen,
        "simulator": primary_sim, "all_simulators": all_sims,
        "full_log": full_log, "summaries": summaries, "decisions": decisions,
        "run_custom": lambda ch, sale=None: run_custom_scenario(primary_sim, ch, sale),
    }


def _print_multi_model_comparison(all_scen):
    """Side-by-side scenario comparison across models."""
    print(f"\n  --- Multi-Model Comparison ---")
    keys = list(all_scen.keys())
    first = all_scen[keys[0]]
    snames = [s["scenario_name"] for s in first]

    header = f"  {'Scenario':<35}"
    for k in keys:
        header += f" {k.split('|')[1][:10]:>10}"
    print(header)
    print("  " + "-" * (35 + 11 * len(keys)))

    for sn in snames:
        row = f"  {sn:<35}"
        for k in keys:
            match = [s for s in all_scen[k] if s["scenario_name"] == sn]
            row += f" {match[0]['change_pct']:>+9.1f}%" if match else f" {'N/A':>10}"
        print(row)


# =============================================================================
# HELPERS
# =============================================================================

def _compute_r2(y, pred, shape):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, p = shape
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    return r2, adj


def _signed_importance(X, y, fnames, imp):
    coefs = {}
    for i, f in enumerate(fnames):
        corr = np.corrcoef(X[:, i], y)[0, 1] if X.shape[0] > 2 else 0
        coefs[f] = (1 if corr >= 0 else -1) * imp[f]
    coefs["const"] = 0
    return coefs


def _apply_mult(scenario, feat, mult):
    if feat.startswith("log_"):
        scenario[feat] = np.log1p(max(0, np.expm1(scenario[feat]) * mult))
    else:
        scenario[feat] = scenario[feat] * mult


def _calc_vif(X, fnames):
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        Xa = X.values if hasattr(X, "values") else X
        if Xa.shape[0] <= Xa.shape[1] + 1:
            return None
        return max(variance_inflation_factor(Xa, i) for i in range(Xa.shape[1]))
    except Exception:
        return None


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    from eda_pipeline import load_all_data
    from outlier_detection import run_outlier_pipeline
    from feature_engineering import run_feature_engineering

    data = load_all_data()
    clean, _, _ = run_outlier_pipeline(data)
    fe = run_feature_engineering(clean)
    result = run_modeling_pipeline(fe, clean_data=clean)

    # Example custom scenario
    # result["run_custom"]({"Online.marketing": 30, "TV": -15}, sale=1)