"""
=============================================================================
MODELING ENGINE
=============================================================================
Incremental feature selection, model training, ranking, and evaluation.

Consolidates pipeline/models.py + modeling/advanced_modeling.py logic.
Pure functions, no visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelScore:
    """Container for model metrics and metadata."""
    model_type: str
    r2: float
    rmse: float
    mae: float
    cv_mean: float
    cv_std: float
    feature_set: List[str]
    feature_count: int
    ordinality_score: float
    coef: Dict[str, float]
    
    def overall_score(self, weights: Dict[str, float] = None) -> float:
        """
        Compute weighted overall score.
        
        Default: 60% R², 20% CV stability, 20% simplicity
        """
        if weights is None:
            weights = {"fit": 0.6, "stability": 0.2, "simplicity": 0.2}
        
        r2_component = self.r2 * weights["fit"]
        # Higher CV mean, lower CV std = more stable
        cv_component = (self.cv_mean - self.cv_std) * weights["stability"]
        # Prefer simpler models
        simplicity_component = (1.0 / (1.0 + self.feature_count / 10)) * weights["simplicity"]
        
        return r2_component + cv_component + simplicity_component


# =============================================================================
# FEATURE IMPORTANCE & SELECTION
# =============================================================================

def get_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "correlation"
) -> List[str]:
    """
    Rank features by importance.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: "correlation" (default), "variance", "random"
        
    Returns:
        Features sorted by importance (high to low)
    """
    if method == "correlation":
        # Pearson correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        return correlations.index.tolist()
    
    elif method == "variance":
        # Features with highest variance
        variances = X.var().sort_values(ascending=False)
        return variances.index.tolist()
    
    else:  # random
        # Return in random order
        features = X.columns.tolist()
        np.random.shuffle(features)
        return features


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train linear regression with cross-validation.
    
    Returns coefficients, R², RMSE, MAE, and CV metrics.
    """
    try:
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model, X, y,
            cv=kf,
            scoring='r2',
            return_train_score=False
        )
        cv_mean = cv_scores['test_score'].mean()
        cv_std = cv_scores['test_score'].std()
        
        # Coefficients
        coef_dict = {X.columns[i]: float(model.coef_[i]) for i in range(len(model.coef_))}
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'coef': coef_dict,
            'y_pred': y_pred,
        }
    except Exception as e:
        return {'error': str(e)}


def train_ridge_regression(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 1.0,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train Ridge regression (L2 regularization).
    
    Returns coefficients, R², RMSE, MAE, and CV metrics.
    """
    try:
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model, X, y,
            cv=kf,
            scoring='r2'
        )
        cv_mean = cv_scores['test_score'].mean()
        cv_std = cv_scores['test_score'].std()
        
        coef_dict = {X.columns[i]: float(model.coef_[i]) for i in range(len(model.coef_))}
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'coef': coef_dict,
            'y_pred': y_pred,
        }
    except Exception as e:
        return {'error': str(e)}


def train_bayesian_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train Bayesian Ridge (probabilistic regression).
    
    Returns coefficients, R², RMSE, MAE, and CV metrics.
    """
    try:
        model = BayesianRidge()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model, X, y,
            cv=kf,
            scoring='r2'
        )
        cv_mean = cv_scores['test_score'].mean()
        cv_std = cv_scores['test_score'].std()
        
        coef_dict = {X.columns[i]: float(model.coef_[i]) for i in range(len(model.coef_))}
        
        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'coef': coef_dict,
            'y_pred': y_pred,
        }
    except Exception as e:
        return {'error': str(e)}


def train_glm_poisson(
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, Any]:
    """
    Train Poisson GLM (count data).
    """
    try:
        X_with_const = sm.add_constant(X)
        model = sm.GLM(y, X_with_const, family=sm.families.Poisson())
        result = model.fit()
        y_pred = result.predict(X_with_const)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        coef_dict = result.params.to_dict()
        
        return {
            'model': result,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': r2,  # No CV for GLM
            'cv_std': 0,
            'coef': coef_dict,
            'y_pred': y_pred.values,
        }
    except Exception as e:
        return {'error': str(e)}


def train_glm_gamma(
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, Any]:
    """
    Train Gamma GLM (positive continuous data).
    """
    try:
        X_with_const = sm.add_constant(X)
        model = sm.GLM(y, X_with_const, family=sm.families.Gamma())
        result = model.fit()
        y_pred = result.predict(X_with_const)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        coef_dict = result.params.to_dict()
        
        return {
            'model': result,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': r2,
            'cv_std': 0,
            'coef': coef_dict,
            'y_pred': y_pred.values,
        }
    except Exception as e:
        return {'error': str(e)}


# =============================================================================
# MODEL VALIDATION
# =============================================================================

def check_ordinality(
    coef_dict: Dict[str, float],
    important_channels: List[str] = None
) -> Dict[str, Any]:
    """
    Check that important channels have positive coefficients.
    
    Returns ordinality_score (0-1) and violation details.
    """
    if important_channels is None:
        # Default: all channels should be positive
        important_channels = list(coef_dict.keys())
    
    violations = []
    
    for channel in important_channels:
        if channel in coef_dict:
            coef = coef_dict[channel]
            if coef < 0:
                violations.append({
                    'channel': channel,
                    'coefficient': round(coef, 4),
                    'issue': 'Negative (expected positive)',
                })
    
    # Score: 1 if no violations, 0 otherwise
    ordinality_score = 1.0 if len(violations) == 0 else 0.5
    
    return {
        'ordinality_score': ordinality_score,
        'violations': violations,
        'num_violations': len(violations),
    }


# =============================================================================
# INCREMENTAL TRAINING
# =============================================================================

def train_incremental_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_importance_order: List[str] = None,
    max_models: int = 50,
    model_types: List[str] = None,
    cv_folds: int = 5,
) -> Tuple[List[ModelScore], Dict[str, Any]]:
    """
    Train models incrementally, starting with most important features.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        feature_importance_order: Order to add features (important first)
        max_models: Maximum models to train
        model_types: Which model types to try (default: all)
        cv_folds: Cross-validation folds
        
    Returns:
        (ranked_models, metadata)
    """
    if feature_importance_order is None:
        feature_importance_order = get_feature_importance(X, y, method="correlation")
    
    if model_types is None:
        model_types = ["LinearRegression", "Ridge", "BayesianRidge", "Poisson", "Gamma"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    all_models = []
    models_by_type = {mt: [] for mt in model_types}
    
    # Incremental: start with 1 feature, add more
    for num_features in range(1, min(len(feature_importance_order) + 1, max_models + 1)):
        selected_features = feature_importance_order[:num_features]
        X_subset = X_scaled_df[selected_features]
        
        # Try each model type
        for model_type in model_types:
            if model_type == "LinearRegression":
                result = train_linear_regression(X_subset, y, cv_folds)
            elif model_type == "Ridge":
                result = train_ridge_regression(X_subset, y, alpha=1.0, cv_folds=cv_folds)
            elif model_type == "BayesianRidge":
                result = train_bayesian_ridge(X_subset, y, cv_folds)
            elif model_type == "Poisson":
                result = train_glm_poisson(X_subset, y)
            elif model_type == "Gamma":
                result = train_glm_gamma(X_subset, y)
            else:
                continue
            
            if 'error' in result:
                continue
            
            # Check ordinality
            ordinality = check_ordinality(result['coef'], selected_features)
            
            # Create ModelScore
            score = ModelScore(
                model_type=model_type,
                r2=result['r2'],
                rmse=result['rmse'],
                mae=result['mae'],
                cv_mean=result['cv_mean'],
                cv_std=result['cv_std'],
                feature_set=selected_features,
                feature_count=num_features,
                ordinality_score=ordinality['ordinality_score'],
                coef=result['coef'],
            )
            
            all_models.append(score)
            models_by_type[model_type].append(score)
    
    # Rank all models
    ranked_models = sorted(
        all_models,
        key=lambda m: m.overall_score(),
        reverse=True
    )
    
    # Top-10
    top_10 = ranked_models[:10]
    
    return top_10, {
        'total_trained': len(all_models),
        'feature_count': len(X.columns),
        'models_by_type': {k: len(v) for k, v in models_by_type.items()},
    }


# =============================================================================
# RANKING & SELECTION
# =============================================================================

def select_best_model(
    ranked_models: List[ModelScore],
    metric: str = "overall_score"
) -> ModelScore:
    """
    Select best model by specified metric.
    
    Args:
        ranked_models: List of ranked models
        metric: "overall_score", "r2", "cv_mean", "simplicity"
        
    Returns:
        Best model
    """
    if not ranked_models:
        return None
    
    if metric == "overall_score":
        return ranked_models[0]  # Already sorted
    
    elif metric == "r2":
        return max(ranked_models, key=lambda m: m.r2)
    
    elif metric == "cv_mean":
        return max(ranked_models, key=lambda m: m.cv_mean)
    
    elif metric == "simplicity":
        return min(ranked_models, key=lambda m: m.feature_count)
    
    else:
        return ranked_models[0]


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_modeling(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "total_gmv",
    max_models: int = 50,
) -> Dict[str, Any]:
    """
    Run complete modeling pipeline.
    
    Args:
        df: Input data
        feature_columns: Channel/feature columns
        target_column: Target variable name
        max_models: Max models to train
        
    Returns:
        {
            'ranked_models': [...top 10...],
            'best_model': ModelScore,
            'metadata': {...},
            'log': [...],
            'is_valid': bool,
        }
    """
    log = []
    
    # Validate
    if target_column not in df.columns:
        log.append(f"❌ Target column '{target_column}' not found")
        return {
            'ranked_models': [],
            'best_model': None,
            'metadata': {},
            'log': log,
            'is_valid': False,
        }
    
    # Remove NaNs
    valid_features = [f for f in feature_columns if f in df.columns]
    df_clean = df[[*valid_features, target_column]].dropna()
    
    if len(df_clean) < 10:
        log.append(f"❌ Insufficient data: {len(df_clean)} rows (need >= 10)")
        return {
            'ranked_models': [],
            'best_model': None,
            'metadata': {},
            'log': log,
            'is_valid': False,
        }
    
    log.append(f"✅ Data ready: {len(df_clean)} rows, {len(valid_features)} features")
    
    # Features and target
    X = df_clean[valid_features]
    y = df_clean[target_column]
    
    # Train incremental models
    ranked_models, metadata = train_incremental_models(
        X, y,
        max_models=max_models,
    )
    
    log.append(f"✅ Trained {metadata['total_trained']} models")
    
    if ranked_models:
        best = ranked_models[0]
        log.append(f"✅ Best model: {best.model_type} (score={best.overall_score():.3f}, R²={best.r2:.3f})")
        
        return {
            'ranked_models': ranked_models,
            'best_model': best,
            'metadata': metadata,
            'log': log,
            'is_valid': True,
        }
    else:
        log.append("❌ No models trained successfully")
        return {
            'ranked_models': [],
            'best_model': None,
            'metadata': metadata,
            'log': log,
            'is_valid': False,
        }


def model_score_to_dict(score: ModelScore) -> Dict[str, Any]:
    """Convert ModelScore dataclass to dict for serialization."""
    return {
        'model_type': score.model_type,
        'r2': round(score.r2, 4),
        'rmse': round(score.rmse, 4),
        'mae': round(score.mae, 4),
        'cv_mean': round(score.cv_mean, 4),
        'cv_std': round(score.cv_std, 4),
        'feature_set': score.feature_set,
        'feature_count': score.feature_count,
        'ordinality_score': round(score.ordinality_score, 3),
        'overall_score': round(score.overall_score(), 3),
        'coef': {k: round(v, 6) for k, v in score.coef.items()},
    }
