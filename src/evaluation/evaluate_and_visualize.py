"""
F1 Model Evaluation and Visualization
Creates comprehensive visualizations and analysis of model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class F1ModelEvaluator:
    """Evaluates and visualizes F1 prediction models"""
    
    def __init__(self):
        self.load_data()
        self.load_models()
        self.load_results()
    
    def load_data(self):
        """Load test data"""
        print("Loading data...")
        df = pd.read_csv("data/processed/f1_ml_dataset.csv")
        
        feature_cols = [
            'driver_races_count', 'driver_avg_position', 'driver_best_position',
            'driver_worst_position', 'driver_podiums', 'driver_wins',
            'driver_top5', 'driver_top10', 'driver_dnf_rate', 'driver_recent_form',
            'circuit_driver_races', 'circuit_driver_avg_position',
            'circuit_driver_best_position', 'air_temperature', 'track_temperature',
            'humidity', 'wind_speed', 'rainfall', 'pit_stop_count', 'avg_pit_duration'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        from sklearn.model_selection import train_test_split
        X = df[available_features]
        y = df['final_position']
        
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.feature_names = available_features
        print(f"Loaded {len(self.X_test)} test samples")
    
    def load_models(self):
        """Load trained models"""
        print("Loading models...")
        self.models = {}
        
        model_names = [
            'rf_regressor', 'xgb_regressor', 'lgb_regressor',
            'rf_classifier_podium', 'xgb_classifier_podium', 'lgb_classifier_podium'
        ]
        
        for name in model_names:
            try:
                with open(f"results/models/{name}.pkl", 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"  Loaded: {name}")
            except:
                print(f"  Failed to load: {name}")
    
    def load_results(self):
        """Load training results"""
        with open("results/models/training_results.json", 'r') as f:
            self.results = json.load(f)
    
    def create_performance_comparison(self):
        """Create model performance comparison chart"""
        print("\nCreating performance comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Regression models comparison
        reg_models = ['rf_regressor', 'xgb_regressor', 'lgb_regressor']
        reg_names = ['Random Forest', 'XGBoost', 'LightGBM']
        
        mae_scores = [self.results[m]['mae'] for m in reg_models]
        r2_scores = [self.results[m]['r2'] for m in reg_models]
        
        x = np.arange(len(reg_names))
        width = 0.35
        
        ax1 = axes[0]
        ax1.bar(x - width/2, mae_scores, width, label='MAE', color='#e74c3c')
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, r2_scores, width, label='R²', color='#3498db')
        
        ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MAE (positions)', fontsize=10, color='#e74c3c')
        ax1_twin.set_ylabel('R² Score', fontsize=10, color='#3498db')
        ax1.set_title('Regression Models - Position Prediction', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(reg_names)
        ax1.tick_params(axis='y', labelcolor='#e74c3c')
        ax1_twin.tick_params(axis='y', labelcolor='#3498db')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (mae, r2) in enumerate(zip(mae_scores, r2_scores)):
            ax1.text(i - width/2, mae + 0.1, f'{mae:.2f}', ha='center', va='bottom', fontsize=9)
            ax1_twin.text(i + width/2, r2 + 0.01, f'{r2:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Classification models comparison
        clf_models = ['rf_classifier_podium', 'xgb_classifier_podium', 'lgb_classifier_podium']
        clf_names = ['Random Forest', 'XGBoost', 'LightGBM']
        
        acc_scores = [self.results[m]['accuracy'] for m in clf_models]
        
        ax2 = axes[1]
        bars = ax2.bar(clf_names, acc_scores, color=['#2ecc71', '#f39c12', '#9b59b6'])
        
        ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Classification Models - Podium Prediction', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, acc_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: results/figures/model_performance_comparison.png")
        plt.close()
    
    def create_prediction_analysis(self):
        """Create prediction vs actual analysis"""
        print("\nCreating prediction analysis...")
        
        # Use best regression model
        model = self.models['rf_regressor']
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 1, 20)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot: Predicted vs Actual
        ax1 = axes[0]
        ax1.scatter(self.y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax1.plot([1, 20], [1, 20], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Position', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted Position', fontsize=11, fontweight='bold')
        ax1.set_title('Random Forest: Predicted vs Actual Positions', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 21])
        ax1.set_ylim([0, 21])
        
        # Add metrics text
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        ax1.text(0.05, 0.95, f'MAE: {mae:.2f}\nR²: {r2:.2f}',
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Residuals plot
        ax2 = axes[1]
        residuals = y_pred - self.y_test
        ax2.scatter(self.y_test, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Actual Position', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=11, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/prediction_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: results/figures/prediction_analysis.png")
        plt.close()
    
    def create_feature_importance(self):
        """Create feature importance visualization"""
        print("\nCreating feature importance chart...")
        
        # Get feature importance from Random Forest
        model = self.models['rf_regressor']
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Feature Importance - Random Forest', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        print("  Saved: results/figures/feature_importance.png")
        plt.close()
    
    def create_error_distribution(self):
        """Create error distribution analysis"""
        print("\nCreating error distribution...")
        
        model = self.models['rf_regressor']
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 1, 20)
        
        errors = np.abs(y_pred - self.y_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of absolute errors
        ax1 = axes[0]
        ax1.hist(errors, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax1.axvline(errors.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {errors.mean():.2f}')
        ax1.axvline(errors.median(), color='g', linestyle='--', lw=2, label=f'Median: {errors.median():.2f}')
        ax1.set_xlabel('Absolute Error (positions)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Cumulative error distribution
        ax2 = axes[1]
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax2.plot(sorted_errors, cumulative, linewidth=2, color='#e74c3c')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Absolute Error (positions)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add annotations
        error_50 = sorted_errors[int(len(sorted_errors) * 0.5)]
        error_80 = sorted_errors[int(len(sorted_errors) * 0.8)]
        ax2.text(error_50, 52, f'50%: ±{error_50:.1f} pos', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))
        ax2.text(error_80, 82, f'80%: ±{error_80:.1f} pos', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig('results/figures/error_distribution.png', dpi=300, bbox_inches='tight')
        print("  Saved: results/figures/error_distribution.png")
        plt.close()
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard"""
        print("\nCreating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Model comparison
        ax1 = fig.add_subplot(gs[0, :2])
        reg_models = ['rf_regressor', 'xgb_regressor', 'lgb_regressor']
        reg_names = ['RF', 'XGB', 'LGB']
        mae_scores = [self.results[m]['mae'] for m in reg_models]
        r2_scores = [self.results[m]['r2'] for m in reg_models]
        
        x = np.arange(len(reg_names))
        width = 0.35
        ax1.bar(x - width/2, mae_scores, width, label='MAE', color='#e74c3c')
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, r2_scores, width, label='R²', color='#3498db')
        ax1.set_xticks(x)
        ax1.set_xticklabels(reg_names)
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_ylabel('MAE', color='#e74c3c')
        ax1_twin.set_ylabel('R²', color='#3498db')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Feature importance (top 10)
        ax2 = fig.add_subplot(gs[0, 2])
        model = self.models['rf_regressor']
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        ax2.barh(range(len(importance_df)), importance_df['importance'], color='#2ecc71')
        ax2.set_yticks(range(len(importance_df)))
        ax2.set_yticklabels([f.replace('_', ' ')[:15] for f in importance_df['feature']], fontsize=8)
        ax2.set_title('Top 10 Features', fontweight='bold')
        ax2.invert_yaxis()
        
        # Prediction scatter
        ax3 = fig.add_subplot(gs[1, 0])
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 1, 20)
        ax3.scatter(self.y_test, y_pred, alpha=0.6, s=30)
        ax3.plot([1, 20], [1, 20], 'r--', lw=2)
        ax3.set_xlabel('Actual')
        ax3.set_ylabel('Predicted')
        ax3.set_title('Predictions vs Actual', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Residuals
        ax4 = fig.add_subplot(gs[1, 1])
        residuals = y_pred - self.y_test
        ax4.scatter(self.y_test, residuals, alpha=0.6, s=30)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Actual Position')
        ax4.set_ylabel('Residual')
        ax4.set_title('Residual Plot', fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Error distribution
        ax5 = fig.add_subplot(gs[1, 2])
        errors = np.abs(residuals)
        ax5.hist(errors, bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax5.axvline(errors.mean(), color='r', linestyle='--', lw=2)
        ax5.set_xlabel('Absolute Error')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Error Distribution', fontweight='bold')
        
        # Position-wise accuracy
        ax6 = fig.add_subplot(gs[2, :])
        position_errors = []
        for pos in range(1, 21):
            mask = self.y_test == pos
            if mask.sum() > 0:
                pos_errors = errors[mask]
                position_errors.append({
                    'position': pos,
                    'mean_error': pos_errors.mean(),
                    'count': mask.sum()
                })
        
        if position_errors:
            pos_df = pd.DataFrame(position_errors)
            bars = ax6.bar(pos_df['position'], pos_df['mean_error'], color='#f39c12', edgecolor='black')
            ax6.set_xlabel('Actual Position')
            ax6.set_ylabel('Mean Absolute Error')
            ax6.set_title('Prediction Error by Position', fontweight='bold')
            ax6.grid(axis='y', alpha=0.3)
            ax6.set_xticks(range(1, 21))
        
        plt.suptitle('F1 Race Prediction - Model Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('results/figures/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("  Saved: results/figures/comprehensive_dashboard.png")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "=" * 60)
        print("GENERATING F1 MODEL VISUALIZATIONS")
        print("=" * 60)
        
        self.create_performance_comparison()
        self.create_prediction_analysis()
        self.create_feature_importance()
        self.create_error_distribution()
        self.create_comprehensive_dashboard()
        
        print("\n" + "=" * 60)
        print("VISUALIZATION GENERATION COMPLETE!")
        print("=" * 60)

if __name__ == "__main__":
    evaluator = F1ModelEvaluator()
    evaluator.generate_all_visualizations()

