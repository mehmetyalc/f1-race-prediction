"""
F1 Race Result Prediction - ML Model Training
Trains multiple models to predict final race positions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class F1ModelTrainer:
    """Trains ML models for F1 race prediction"""
    
    def __init__(self, data_path="data/processed/f1_ml_dataset.csv"):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("Loading ML dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Select features for training
        feature_cols = [
            'driver_races_count', 'driver_avg_position', 'driver_best_position',
            'driver_worst_position', 'driver_podiums', 'driver_wins',
            'driver_top5', 'driver_top10', 'driver_dnf_rate', 'driver_recent_form',
            'circuit_driver_races', 'circuit_driver_avg_position',
            'circuit_driver_best_position', 'air_temperature', 'track_temperature',
            'humidity', 'wind_speed', 'rainfall', 'pit_stop_count', 'avg_pit_duration'
        ]
        
        # Check which features exist
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        self.X = self.df[available_features]
        self.y = self.df['final_position']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Features: {list(self.X.columns)}")
    
    def train_random_forest_regressor(self):
        """Train Random Forest Regressor"""
        print("\n[1/6] Training Random Forest Regressor...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Clip predictions to valid range [1, 20]
        y_pred = np.clip(y_pred, 1, 20)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['rf_regressor'] = model
        self.results['rf_regressor'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        
        return model
    
    def train_xgboost_regressor(self):
        """Train XGBoost Regressor"""
        print("\n[2/6] Training XGBoost Regressor...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Clip predictions
        y_pred = np.clip(y_pred, 1, 20)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['xgb_regressor'] = model
        self.results['xgb_regressor'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        
        return model
    
    def train_lightgbm_regressor(self):
        """Train LightGBM Regressor"""
        print("\n[3/6] Training LightGBM Regressor...")
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Clip predictions
        y_pred = np.clip(y_pred, 1, 20)
        
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        self.models['lgb_regressor'] = model
        self.results['lgb_regressor'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        
        return model
    
    def train_random_forest_classifier(self):
        """Train Random Forest Classifier for podium prediction"""
        print("\n[4/6] Training Random Forest Classifier (Podium Prediction)...")
        
        # Create binary target: podium (1-3) vs non-podium
        y_train_class = (self.y_train <= 3).astype(int)
        y_test_class = (self.y_test <= 3).astype(int)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, y_train_class)
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(y_test_class, y_pred)
        f1 = f1_score(y_test_class, y_pred, average='binary')
        
        self.models['rf_classifier_podium'] = model
        self.results['rf_classifier_podium'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        return model
    
    def train_xgboost_classifier(self):
        """Train XGBoost Classifier for podium prediction"""
        print("\n[5/6] Training XGBoost Classifier (Podium Prediction)...")
        
        # Create binary target
        y_train_class = (self.y_train <= 3).astype(int)
        y_test_class = (self.y_test <= 3).astype(int)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, y_train_class)
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(y_test_class, y_pred)
        f1 = f1_score(y_test_class, y_pred, average='binary')
        
        self.models['xgb_classifier_podium'] = model
        self.results['xgb_classifier_podium'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        return model
    
    def train_lightgbm_classifier(self):
        """Train LightGBM Classifier for podium prediction"""
        print("\n[6/6] Training LightGBM Classifier (Podium Prediction)...")
        
        # Create binary target
        y_train_class = (self.y_train <= 3).astype(int)
        y_test_class = (self.y_test <= 3).astype(int)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(self.X_train, y_train_class)
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(y_test_class, y_pred)
        f1 = f1_score(y_test_class, y_pred, average='binary')
        
        self.models['lgb_classifier_podium'] = model
        self.results['lgb_classifier_podium'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        return model
    
    def get_feature_importance(self, model_name):
        """Get feature importance for a model"""
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return None
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        
        for name, model in self.models.items():
            model_path = f"results/models/{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {model_path}")
    
    def save_results(self):
        """Save training results"""
        print("\nSaving results...")
        
        # Save as JSON
        results_path = "results/models/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Saved: {results_path}")
        
        # Save as text report
        report_path = "results/models/training_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("F1 RACE PREDICTION - MODEL TRAINING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("REGRESSION MODELS (Position Prediction)\n")
            f.write("-" * 60 + "\n")
            for model_name in ['rf_regressor', 'xgb_regressor', 'lgb_regressor']:
                if model_name in self.results:
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  MAE:  {self.results[model_name]['mae']:.3f}\n")
                    f.write(f"  RMSE: {self.results[model_name]['rmse']:.3f}\n")
                    f.write(f"  R²:   {self.results[model_name]['r2']:.3f}\n")
            
            f.write("\n\nCLASSIFICATION MODELS (Podium Prediction)\n")
            f.write("-" * 60 + "\n")
            for model_name in ['rf_classifier_podium', 'xgb_classifier_podium', 'lgb_classifier_podium']:
                if model_name in self.results:
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  Accuracy: {self.results[model_name]['accuracy']:.3f}\n")
                    f.write(f"  F1-Score: {self.results[model_name]['f1_score']:.3f}\n")
        
        print(f"  Saved: {report_path}")
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "=" * 60)
        print("TRAINING F1 RACE PREDICTION MODELS")
        print("=" * 60)
        
        # Regression models
        self.train_random_forest_regressor()
        self.train_xgboost_regressor()
        self.train_lightgbm_regressor()
        
        # Classification models
        self.train_random_forest_classifier()
        self.train_xgboost_classifier()
        self.train_lightgbm_classifier()
        
        # Save everything
        self.save_models()
        self.save_results()
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE!")
        print("=" * 60)
        
        # Print summary
        print("\nBEST MODELS:")
        print("-" * 60)
        
        # Best regression model
        reg_models = {k: v for k, v in self.results.items() if 'regressor' in k}
        best_reg = min(reg_models.items(), key=lambda x: x[1]['mae'])
        print(f"\nRegression (Position Prediction):")
        print(f"  Model: {best_reg[0]}")
        print(f"  MAE: {best_reg[1]['mae']:.3f} positions")
        print(f"  R²: {best_reg[1]['r2']:.3f}")
        
        # Best classification model
        clf_models = {k: v for k, v in self.results.items() if 'classifier' in k}
        best_clf = max(clf_models.items(), key=lambda x: x[1]['f1_score'])
        print(f"\nClassification (Podium Prediction):")
        print(f"  Model: {best_clf[0]}")
        print(f"  Accuracy: {best_clf[1]['accuracy']:.3f}")
        print(f"  F1-Score: {best_clf[1]['f1_score']:.3f}")
        
        return self.models, self.results

if __name__ == "__main__":
    trainer = F1ModelTrainer()
    models, results = trainer.train_all_models()

