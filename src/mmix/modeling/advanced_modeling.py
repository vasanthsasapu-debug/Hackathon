"""
Advanced modeling pipeline with incremental feature selection, cross-validation,
and comprehensive model comparison across multiple algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge, BayesianRidge, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


@dataclass
class ModelScore:
    """Container for model scoring metrics."""
    model_type: str
    params: Dict[str, Any]
    r2: float
    rmse: float
    mae: float
    cv_mean: float
    cv_std: float
    feature_set: List[str]
    feature_count: int
    
    def overall_score(self) -> float:
        """Compute weighted overall score: 60% R², 20% CV stability, 20% simplicity."""
        r2_weight = self.r2 * 0.6
        cv_weight = (1.0 - self.cv_std) * 0.2  # Lower std dev = more stable
        simplicity_weight = (1.0 / (1.0 + self.feature_count / 10)) * 0.2  # Prefer simpler models
        return r2_weight + cv_weight + simplicity_weight


class AdvancedModelingEngine:
    """Advanced modeling with incremental feature selection and cross-validation."""
    
    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.ranked_models: List[ModelScore] = []
        self.model_histories: Dict[str, List[ModelScore]] = {}  # Track incremental progress
        
    def train_incremental_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importance_order: List[str] = None,
        max_models: int = 50
    ) -> Tuple[List[ModelScore], Dict[str, Any]]:
        """
        Train models incrementally, starting with most important features.
        
        Args:
            X: Feature DataFrame with channels
            y: Target (sales/GMV)
            feature_importance_order: Order to add features (important first)
            max_models: Maximum number of models to train
            
        Returns:
            Ranked models and training history
        """
        if feature_importance_order is None:
            # Use correlation with target as importance
            feature_importance_order = self._get_feature_importance(X, y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        models_trained = []
        
        # Start with single feature, incrementally add
        for num_features in range(1, min(len(feature_importance_order) + 1, max_models + 1)):
            selected_features = feature_importance_order[:num_features]
            X_subset = X_scaled_df[selected_features]
            
            # Train multiple model types
            for model_type in ["Ridge", "BayesianRidge", "LinearRegression"]:
                score = self._train_and_evaluate(
                    X_subset, y, model_type, selected_features
                )
                models_trained.append(score)
        
        # Rank all models
        self.ranked_models = sorted(
            models_trained, 
            key=lambda m: m.overall_score(), 
            reverse=True
        )
        
        return self.ranked_models, {"total_trained": len(models_trained)}
    
    def _get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Get feature importance based on correlation with target."""
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        return correlations.index.tolist()
    
    def _train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        feature_set: List[str]
    ) -> ModelScore:
        """Train single model and evaluate with cross-validation."""
        
        # Create and train model
        if model_type == "Ridge":
            model = Ridge(alpha=1.0, random_state=self.random_state)
        elif model_type == "BayesianRidge":
            try:
                model = BayesianRidge(random_state=self.random_state)
            except TypeError:
                model = BayesianRidge()
        else:  # LinearRegression
            model = LinearRegression()
        
        # Fit model
        model.fit(X, y)
        
        # Predictions and metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        return ModelScore(
            model_type=model_type,
            params=self._get_params(model),
            r2=r2,
            rmse=rmse,
            mae=mae,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            feature_set=list(feature_set),
            feature_count=len(feature_set)
        )
    
    def _get_params(self, model) -> Dict[str, Any]:
        """Extract model parameters."""
        if hasattr(model, 'alpha'):
            return {"alpha": model.alpha}
        return {}
    
    def get_top_models(self, n: int = 10) -> List[ModelScore]:
        """Get top N models by overall score."""
        return self.ranked_models[:n]
    
    def validate_ordinality(
        self,
        model: Any,
        feature_names: List[str],
        key_channels: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate that important channels have proper ordinality (correct sign/magnitude).
        
        Args:
            model: Trained sklearn model
            feature_names: Names of features in order
            key_channels: List of channels that should be important
        """
        if key_channels is None:
            key_channels = ["calls", "speaker_programs", "rep_email", "hq_email"]
        
        if not hasattr(model, 'coef_'):
            return {"status": "N/A - model type doesn't support coef_"}
        
        coef_dict = {name: coef for name, coef in zip(feature_names, model.coef_)}
        
        ordinality_check = {}
        for channel in key_channels:
            matching = [k for k in coef_dict if channel.lower() in k.lower()]
            if matching:
                ordinality_check[matching[0]] = {
                    "coefficient": coef_dict[matching[0]],
                    "magnitude": abs(coef_dict[matching[0]])
                }
        
        return ordinality_check
    
    def get_model_comparison_data(self, top_n: int = 10) -> pd.DataFrame:
        """Get comparison data for top N models."""
        top_models = self.get_top_models(top_n)
        
        return pd.DataFrame([
            {
                "Rank": i + 1,
                "Model Type": m.model_type,
                "R²": round(m.r2, 4),
                "RMSE": round(m.rmse, 2),
                "MAE": round(m.mae, 2),
                "CV Mean": round(m.cv_mean, 4),
                "CV Std": round(m.cv_std, 4),
                "Features": m.feature_count,
                "Overall Score": round(m.overall_score(), 4),
            }
            for i, m in enumerate(top_models)
        ])


class CrossValidationAnalyzer:
    """Analyze model robustness through cross-validation."""
    
    @staticmethod
    def detailed_cv_analysis(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform detailed cross-validation analysis."""
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = cross_validate(
            model, X, y, cv=kf,
            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
            return_train_score=True
        )
        
        return {
            "test_r2_mean": cv_results['test_r2'].mean(),
            "test_r2_std": cv_results['test_r2'].std(),
            "train_r2_mean": cv_results['train_r2'].mean(),
            "test_rmse_mean": np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()),
            "test_rmse_std": np.sqrt(-cv_results['test_neg_mean_squared_error']).std(),
            "overfitting_gap": cv_results['train_r2'].mean() - cv_results['test_r2'].mean(),
            "fold_scores": cv_results['test_r2'].tolist(),
        }


def create_model_comparison_table(top_models: List[ModelScore]) -> str:
    """Create formatted text table for model comparison."""
    
    table = "\n" + "="*120 + "\n"
    table += f"{'Rank':<6} {'Model Type':<18} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'CV Mean':<10} {'CV Std':<10} {'Features':<10} {'Score':<10}\n"
    table += "-"*120 + "\n"
    
    for i, model in enumerate(top_models[:10], 1):
        table += f"{i:<6} {model.model_type:<18} {model.r2:<8.4f} {model.rmse:<10.2f} {model.mae:<10.2f} {model.cv_mean:<10.4f} {model.cv_std:<10.4f} {model.feature_count:<10} {model.overall_score():<10.4f}\n"
    
    table += "="*120 + "\n"
    return table
