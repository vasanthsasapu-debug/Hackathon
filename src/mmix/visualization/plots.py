"""
Visualization module for MMIX analysis with plots for models, elasticities, and scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from matplotlib.figure import Figure
import warnings

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class MixModelVisualizer:
    """Create visualizations for MMIX analysis."""
    
    @staticmethod
    def plot_model_comparison(
        models_data: pd.DataFrame,
        output_path: str = None
    ) -> Figure:
        """Plot top 10 models comparison."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Top 10 Models Comparison', fontsize=16, fontweight='bold')
        
        # R² Comparison
        ax1 = axes[0, 0]
        colors = ['#2ecc71' if x == models_data['R²'].max() else '#3498db' 
                  for x in models_data['R²']]
        ax1.barh(models_data['Rank'].astype(str), models_data['R²'], color=colors)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Model Fit (R²)')
        ax1.invert_yaxis()
        
        # RMSE Comparison
        ax2 = axes[0, 1]
        colors = ['#e74c3c' if x == models_data['RMSE'].min() else '#e67e22' 
                  for x in models_data['RMSE']]
        ax2.barh(models_data['Rank'].astype(str), models_data['RMSE'], color=colors)
        ax2.set_xlabel('RMSE')
        ax2.set_title('Model Error (RMSE - Lower is Better)')
        ax2.invert_yaxis()
        
        # CV Stability (CV Std)
        ax3 = axes[1, 0]
        colors = ['#27ae60' if x == models_data['CV Std'].min() else '#95a5a6' 
                  for x in models_data['CV Std']]
        ax3.barh(models_data['Rank'].astype(str), models_data['CV Std'], color=colors)
        ax3.set_xlabel('CV Std Dev')
        ax3.set_title('Model Stability (CV Std - Lower is More Stable)')
        ax3.invert_yaxis()
        
        # Overall Score
        ax4 = axes[1, 1]
        colors = ['#9b59b6' if x == models_data['Overall Score'].max() else '#8e44ad' 
                  for x in models_data['Overall Score']]
        ax4.barh(models_data['Rank'].astype(str), models_data['Overall Score'], color=colors)
        ax4.set_xlabel('Overall Score')
        ax4.set_title('Weighted Overall Score (60% R², 20% Stability, 20% Simplicity)')
        ax4.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_elasticities(
        elasticities: Dict[str, float],
        output_path: str = None
    ) -> Figure:
        """Plot channel elasticities."""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        channels = list(elasticities.keys())
        values = list(elasticities.values())
        colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
        
        # Sort by absolute value for better visualization
        sorted_data = sorted(zip(channels, values, colors), key=lambda x: abs(x[1]), reverse=True)
        channels, values, colors = zip(*sorted_data)
        
        bars = ax.barh(range(len(channels)), values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels)
        ax.set_xlabel('Elasticity (% change in sales per % change in spend)', fontsize=11)
        ax.set_title('Channel Elasticities', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + (0.2 if val > 0 else -0.2), i, f'{val:.2f}', 
                   va='center', ha='left' if val > 0 else 'right', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_optimization_scenarios(
        scenarios: Dict[str, Dict[str, Any]],
        output_path: str = None
    ) -> Figure:
        """Plot budget allocation scenarios."""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Expected uplift comparison
        ax1 = axes[0]
        scenario_names = list(scenarios.keys())
        uplifts = [scenarios[s].get('expected_uplift', 0) * 100 for s in scenario_names]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        bars1 = ax1.bar(range(len(scenario_names)), uplifts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenario_names], rotation=15, ha='right')
        ax1.set_ylabel('Expected Uplift (%)', fontsize=11)
        ax1.set_title('Expected Revenue Uplift by Scenario', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, uplifts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Scenario descriptions
        ax2 = axes[1]
        ax2.axis('off')
        
        descriptions = {
            'base_case': 'Historical allocation\n(Baseline for comparison)',
            'budget_neutral': 'Reallocate spend by\nchannel elasticity\n(No budget increase)',
            'max_profit': 'Boost high-elasticity\nchannels by 20%\n(ROI optimized)',
            'blue_sky': '20% budget increase\nacross all channels\n(Growth focused)'
        }
        
        y_pos = 0.9
        for scenario, desc in descriptions.items():
            display_name = scenario.replace('_', ' ').title()
            expected_uplift = scenarios[scenario].get('expected_uplift', 0) * 100
            
            ax2.text(0.05, y_pos, f"• {display_name}", fontsize=12, fontweight='bold',
                    transform=ax2.transAxes)
            ax2.text(0.05, y_pos - 0.08, f"  {desc}", fontsize=10, transform=ax2.transAxes,
                    style='italic', color='gray')
            ax2.text(0.05, y_pos - 0.15, f"  Expected Uplift: {expected_uplift:.1f}%",
                    fontsize=10, fontweight='bold', color='#2c3e50',
                    transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            y_pos -= 0.28
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_response_curves(
        elasticities: Dict[str, float],
        baseline_spend: Optional[Dict[str, float]] = None,
        output_path: str = None
    ) -> Figure:
        """Plot response curves for each channel."""
        
        num_channels = len(elasticities)
        cols = 3
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if num_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Generate spend range for each channel
        spend_range = np.linspace(0.5, 1.5, 50)  # 50% to 150% of baseline
        
        for idx, (channel, elasticity) in enumerate(elasticities.items()):
            ax = axes[idx]
            
            # Response curve: sales_impact = elasticity * (% change in spend)
            sales_impact = elasticity * (spend_range - 1.0) * 100  # as % change
            
            ax.plot(spend_range * 100, sales_impact, linewidth=2.5, color='#3498db', marker='o', markersize=4)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.axvline(x=100, color='red', linestyle='--', alpha=0.3, label='Current Spend')
            ax.fill_between(spend_range * 100, 0, sales_impact, alpha=0.2, color='#3498db')
            
            ax.set_xlabel('Spend Level (% of current)', fontsize=10)
            ax.set_ylabel('Sales Impact (%)', fontsize=10)
            ax.set_title(f'{channel}\n(Elasticity: {elasticity:.2f})', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if elasticity > 0:
                ax.text(0.98, 0.05, '↑ Positive', transform=ax.transAxes, 
                       fontsize=10, ha='right', color='#27ae60', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.7))
            else:
                ax.text(0.98, 0.95, '↓ Negative', transform=ax.transAxes, 
                       fontsize=10, ha='right', va='top', color='#e74c3c', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(num_channels, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Channel Response Curves - Sales Impact by Spend Level', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_feature_importance(
        feature_importance: Dict[str, float],
        output_path: str = None
    ) -> Figure:
        """Plot feature importance."""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
        bars = ax.barh(range(len(features)), importance, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score (Absolute Correlation with Sales)', fontsize=11)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_cv_fold_performance(
        cv_scores: List[float],
        model_name: str = "Model",
        output_path: str = None
    ) -> Figure:
        """Plot cross-validation fold performance."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fold_nums = range(1, len(cv_scores) + 1)
        mean_score = np.mean(cv_scores)
        
        colors = ['#2ecc71' if s >= mean_score else '#e74c3c' for s in cv_scores]
        bars = ax.bar(fold_nums, cv_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.axhline(y=mean_score, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
        
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title(f'{model_name} - Cross-Validation Performance by Fold', fontsize=12, fontweight='bold')
        ax.set_xticks(fold_nums)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, cv_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
