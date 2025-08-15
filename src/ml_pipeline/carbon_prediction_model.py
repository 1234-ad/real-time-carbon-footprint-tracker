"""
Machine Learning pipeline for carbon emission prediction and optimization.
Includes feature engineering, model training, and real-time inference.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from influxdb_client import InfluxDBClient
import redis
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float

class FeatureEngineer:
    """Advanced feature engineering for carbon emission prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def create_time_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Basic time features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business hours indicator
        df['business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
        """Create lagged features for time series"""
        df = df.copy()
        df = df.sort_values('timestamp')
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        df = df.sort_values('timestamp')
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-related features (simulated)"""
        df = df.copy()
        
        # Simulate weather data based on time patterns
        df['temperature'] = 20 + 10 * np.sin(2 * np.pi * df['month'] / 12) + np.random.normal(0, 3, len(df))
        df['humidity'] = 50 + 20 * np.sin(2 * np.pi * df['month'] / 12 + np.pi/4) + np.random.normal(0, 5, len(df))
        df['wind_speed'] = 5 + 3 * np.random.exponential(1, len(df))
        
        # Weather impact on energy consumption
        df['heating_degree_days'] = np.maximum(0, 18 - df['temperature'])
        df['cooling_degree_days'] = np.maximum(0, df['temperature'] - 24)
        
        return df
    
    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        df = df.copy()
        
        # Grid-based location encoding
        df['lat_grid'] = (df['lat'] * 100).astype(int) // 10
        df['lng_grid'] = (df['lng'] * 100).astype(int) // 10
        df['location_cluster'] = df['lat_grid'].astype(str) + '_' + df['lng_grid'].astype(str)
        
        # Distance from city center (assuming NYC)
        city_center_lat, city_center_lng = 40.7128, -74.0060
        df['distance_from_center'] = np.sqrt(
            (df['lat'] - city_center_lat)**2 + (df['lng'] - city_center_lng)**2
        )
        
        # Urban density proxy
        df['urban_density'] = 1 / (1 + df['distance_from_center'])
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        # Time features
        df = self.create_time_features(df, 'timestamp')
        
        # Lag features
        df = self.create_lag_features(df, 'calculated_emissions', [1, 2, 3, 6, 12, 24])
        
        # Rolling features
        df = self.create_rolling_features(df, 'calculated_emissions', [3, 6, 12, 24])
        
        # Weather features
        df = self.create_weather_features(df)
        
        # Location features
        df = self.create_location_features(df)
        
        # Categorical encoding
        categorical_cols = ['emission_type', 'consumption_unit', 'device_type']
        df = self.encode_categorical_features(df, categorical_cols)
        
        # Store feature names
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = [col for col in numeric_cols if col not in ['calculated_emissions', 'lat', 'lng']]
        
        return df

class CarbonPredictionModel:
    """Advanced carbon emission prediction model"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different model types"""
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        }
    
    def load_data_from_influxdb(self, hours_back: int = 168) -> pd.DataFrame:
        """Load training data from InfluxDB"""
        client = InfluxDBClient(url="http://localhost:8086", token="your-token", org="carbon-tracker")
        
        query = f'''
        from(bucket: "carbon-emissions")
        |> range(start: -{hours_back}h)
        |> filter(fn: (r) => r["_measurement"] == "carbon_events")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            result = client.query_api().query_data_frame(query)
            
            # Process the data
            df = result.copy()
            df['timestamp'] = pd.to_datetime(df['_time'])
            
            # Extract metadata fields
            if 'metadata' in df.columns:
                metadata_df = pd.json_normalize(df['metadata'])
                df = pd.concat([df, metadata_df], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from InfluxDB: {e}")
            # Return simulated data for demo
            return self._generate_simulated_data(hours_back)
    
    def _generate_simulated_data(self, hours_back: int) -> pd.DataFrame:
        """Generate simulated training data"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Generate hourly data points
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        
        data = []
        for ts in timestamps:
            # Simulate different device types and emission patterns
            for device_type in ['smart_meter', 'hvac', 'industrial', 'vehicle']:
                for i in range(np.random.randint(5, 15)):  # 5-15 devices per type
                    
                    # Base emission with time patterns
                    hour_factor = 1 + 0.3 * np.sin(2 * np.pi * ts.hour / 24)
                    day_factor = 0.8 if ts.weekday() >= 5 else 1.0  # Weekend reduction
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * ts.month / 12)
                    
                    if device_type == 'smart_meter':
                        base_emission = 25 * hour_factor * day_factor * seasonal_factor
                        emission_type = 'electricity_grid'
                        unit = 'kWh'
                    elif device_type == 'hvac':
                        base_emission = 45 * hour_factor * seasonal_factor
                        emission_type = 'electricity_grid'
                        unit = 'kWh'
                    elif device_type == 'industrial':
                        base_emission = 150 * day_factor
                        emission_type = 'electricity_grid'
                        unit = 'kWh'
                    else:  # vehicle
                        base_emission = 8 * day_factor
                        emission_type = 'gasoline'
                        unit = 'liters'
                    
                    # Add noise
                    consumption = base_emission * np.random.normal(1, 0.2)
                    carbon_factor = 0.233 if emission_type == 'electricity_grid' else 2.31
                    calculated_emissions = consumption * carbon_factor
                    
                    data.append({
                        'timestamp': ts,
                        'device_id': f'{device_type}_{i}',
                        'device_type': device_type,
                        'emission_type': emission_type,
                        'consumption_value': consumption,
                        'consumption_unit': unit,
                        'carbon_factor': carbon_factor,
                        'calculated_emissions': calculated_emissions,
                        'lat': 40.7128 + np.random.normal(0, 0.1),
                        'lng': -74.0060 + np.random.normal(0, 0.1)
                    })
        
        return pd.DataFrame(data)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Feature engineering
        df_processed = self.feature_engineer.fit_transform(df)
        
        # Remove rows with NaN values (from lag features)
        df_processed = df_processed.dropna()
        
        # Prepare features and target
        X = df_processed[self.feature_engineer.feature_names].values
        y = df_processed['calculated_emissions'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, ModelMetrics]:
        """Train all models and return performance metrics"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            
            for name, model in self.models.items():
                logging.info(f"Training {name} model...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                metrics = ModelMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape)
                results[name] = metrics
                
                # Log to MLflow
                mlflow.log_metrics({
                    f"{name}_mae": mae,
                    f"{name}_mse": mse,
                    f"{name}_rmse": rmse,
                    f"{name}_r2": r2,
                    f"{name}_mape": mape
                })
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
                mlflow.log_metric(f"{name}_cv_mae", -cv_scores.mean())
                
                logging.info(f"{name} - MAE: {mae:.3f}, R²: {r2:.3f}, MAPE: {mape:.2f}%")
        
        self.is_fitted = True
        return results
    
    def create_ensemble_model(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """Create ensemble model with the best performing models"""
        # Train individual models first
        results = self.train_models(X, y)
        
        # Select top 3 models based on R² score
        top_models = sorted(results.items(), key=lambda x: x[1].r2, reverse=True)[:3]
        
        logging.info(f"Creating ensemble with: {[name for name, _ in top_models]}")
        
        # Create ensemble weights based on performance
        weights = []
        total_r2 = sum(metrics.r2 for _, metrics in top_models)
        
        for name, metrics in top_models:
            weight = metrics.r2 / total_r2
            weights.append(weight)
            logging.info(f"{name} weight: {weight:.3f}")
        
        # Store ensemble configuration
        self.ensemble_models = [(name, self.models[name]) for name, _ in top_models]
        self.ensemble_weights = weights
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self, 'ensemble_models'):
            # Ensemble prediction
            predictions = []
            for (name, model), weight in zip(self.ensemble_models, self.ensemble_weights):
                pred = model.predict(X)
                predictions.append(pred * weight)
            
            return np.sum(predictions, axis=0)
        else:
            # Single model prediction
            return self.models[self.model_type].predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'feature_engineer': self.feature_engineer,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'model_type': self.model_type
        }
        
        if hasattr(self, 'ensemble_models'):
            model_data['ensemble_models'] = self.ensemble_models
            model_data['ensemble_weights'] = self.ensemble_weights
        
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_engineer = model_data['feature_engineer']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.model_type = model_data['model_type']
        
        if 'ensemble_models' in model_data:
            self.ensemble_models = model_data['ensemble_models']
            self.ensemble_weights = model_data['ensemble_weights']
        
        logging.info(f"Model loaded from {filepath}")

def main():
    """Main training pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model
    model = CarbonPredictionModel(model_type='ensemble')
    
    # Load data
    logging.info("Loading training data...")
    df = model.load_data_from_influxdb(hours_back=168)  # 1 week of data
    logging.info(f"Loaded {len(df)} records")
    
    # Prepare data
    logging.info("Preparing features...")
    X, y = model.prepare_data(df)
    logging.info(f"Feature matrix shape: {X.shape}")
    
    # Train models
    logging.info("Training models...")
    results = model.train_models(X, y)
    
    # Create ensemble
    logging.info("Creating ensemble model...")
    model.create_ensemble_model(X, y)
    
    # Save model
    model.save_model('models/carbon_prediction_model.pkl')
    
    # Print final results
    print("\\n=== Model Performance Summary ===")
    for name, metrics in results.items():
        print(f"{name:15} - MAE: {metrics.mae:6.3f}, R²: {metrics.r2:6.3f}, MAPE: {metrics.mape:6.2f}%")

if __name__ == "__main__":
    main()