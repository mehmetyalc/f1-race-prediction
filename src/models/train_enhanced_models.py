"""
Enhanced F1 Model Training
Trains models with improved feature set including qualifying, race pace, and team data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedF1ModelTrainer:
    """Trains enhanced ML models for F1 race prediction"""
    
    def __init__(self, data_path="data/processed/f1_ml_dataset_enhanced.csv"):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare enhanced data"""
        print("Loading enhanced ML dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Remove target from features
        self.X = self.df.drop('final_position', axis=1)
        self.y = self.df['final_position']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Features ({len(self.X.columns)}): {list(self.X.columns)}")
    
    def train_random_forest(self):
        """Train Random Forest with optimized hyperparameters"""
        print("\n[1/3] Training Enhanced Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=200,  # Increased
            max_depth=20,      # Increased
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 1, 20)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
        
        self.models['rf_enhanced'] = model
        self.results['rf_enhanced'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost with optimized hyperparameters"""
        print("\n[2/3] Training Enhanced XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 1, 20)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
        
        self.models['xgb_enhanced'] = model
        self.results['xgb_enhanced'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model
    
    def train_lightgbm(self):
        """Train LightGBM with optimized hyperparameters"""
        print("\n[3/3] Training Enhanced LightGBM...")
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred = np.clip(y_pred, 1, 20)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
        
        self.models['lgb_enhanced'] = model
        self.results['lgb_enhanced'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return model
    
    def get_feature_importance(self, model_name='rf_enhanced'):
        """Get feature importance"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving enhanced models...")
        
        for name, model in self.models.items():
            model_path = f"results/models/{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {model_path}")
    
    def save_results(self):
        """Save training results"""
        print("\nSaving results...")
        
        # Save JSON
        results_path = "results/models/enhanced_training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Saved: {results_path}")
        
        # Save text report
        report_path = "results/models/enhanced_training_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("F1 RACE PREDICTION - ENHANCED MODEL TRAINING RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Samples: {len(self.df)}\n")
            f.write(f"Features: {len(self.X.columns)}\n")
            f.write(f"Training Set: {len(self.X_train)}\n")
            f.write(f"Test Set: {len(self.X_test)}\n\n")
            
            f.write("ENHANCED FEATURES ADDED\n")
            f.write("-" * 70 + "\n")
            f.write("- Qualifying position (grid position)\n")
            f.write("- Grid position gain\n")
            f.write("- Average race pace\n")
            f.write("- Best race pace\n")
            f.write("- Pace consistency\n")
            f.write("- Clean laps count\n")
            f.write("- Team average position\n")
            f.write("- Team best position\n")
            f.write("- Team driver count\n\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 70 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  MAE:  {metrics['mae']:.3f} positions\n")
                f.write(f"  RMSE: {metrics['rmse']:.3f} positions\n")
                f.write(f"  R²:   {metrics['r2']:.3f}\n")
                f.write(f"  CV R² (5-fold): {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}\n\n")
            
            # Feature importance
            f.write("\nTOP 15 FEATURE IMPORTANCE (Random Forest)\n")
            f.write("-" * 70 + "\n")
            importance_df = self.get_feature_importance()
            if importance_df is not None:
                for idx, row in importance_df.head(15).iterrows():
                    f.write(f"{row['feature']:30s}: {row['importance']:.4f}\n")
        
        print(f"  Saved: {report_path}")
    
    def train_all_models(self):
        """Train all enhanced models"""
        print("\n" + "=" * 70)
        print("TRAINING ENHANCED F1 RACE PREDICTION MODELS")
        print("=" * 70)
        
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        
        self.save_models()
        self.save_results()
        
        print("\n" + "=" * 70)
        print("ENHANCED MODEL TRAINING COMPLETE!")
        print("=" * 70)
        
        # Print comparison
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 70)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  MAE:  {metrics['mae']:.3f} positions")
            print(f"  R²:   {metrics['r2']:.3f}")
            print(f"  CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
        
        # Best model
        best_model = min(self.results.items(), key=lambda x: x[1]['mae'])
        print(f"\n🏆 BEST MODEL: {best_model[0].upper()}")
        print(f"   MAE: {best_model[1]['mae']:.3f} positions")
        print(f"   R²:  {best_model[1]['r2']:.3f}")
        
        # Improvement vs baseline
        print("\n📈 IMPROVEMENT vs BASELINE (R² 0.288):")
        for model_name, metrics in self.results.items():
            improvement = ((metrics['r2'] - 0.288) / 0.288) * 100
            print(f"   {model_name}: {improvement:+.1f}%")
        
        return self.models, self.results

if __name__ == "__main__":
    trainer = EnhancedF1ModelTrainer()
    models, results = trainer.train_all_models()

