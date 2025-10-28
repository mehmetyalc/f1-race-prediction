"""
F1 Data Integration and Feature Engineering
Combines multiple data sources and creates features for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1FeatureEngineer:
    """Prepares F1 data for machine learning"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """Load all F1 data files"""
        print("Loading F1 data...")
        self.meetings = pd.read_csv(f"{self.data_dir}/meetings.csv")
        self.sessions = pd.read_csv(f"{self.data_dir}/sessions.csv")
        self.drivers = pd.read_csv(f"{self.data_dir}/drivers.csv")
        self.laps = pd.read_csv(f"{self.data_dir}/laps.csv")
        self.positions = pd.read_csv(f"{self.data_dir}/positions.csv")
        self.pit_stops = pd.read_csv(f"{self.data_dir}/pit_stops.csv")
        self.weather = pd.read_csv(f"{self.data_dir}/weather.csv")
        print("Data loaded successfully!")
    
    def prepare_race_results(self):
        """Extract final race results from positions data"""
        print("\nPreparing race results...")
        
        # Get race sessions only
        race_sessions = self.sessions[self.sessions['session_type'] == 'Race'].copy()
        
        # Get final positions for each driver in each race
        race_results = []
        
        for session_key in race_sessions['session_key'].unique():
            session_positions = self.positions[self.positions['session_key'] == session_key]
            
            if len(session_positions) == 0:
                continue
            
            # Get last recorded position for each driver
            final_positions = session_positions.sort_values('date').groupby('driver_number').last().reset_index()
            
            for _, row in final_positions.iterrows():
                race_results.append({
                    'session_key': session_key,
                    'driver_number': row['driver_number'],
                    'final_position': row['position'],
                    'date': row['date']
                })
        
        race_results_df = pd.DataFrame(race_results)
        
        # Merge with session and meeting info
        race_results_df = race_results_df.merge(
            race_sessions[['session_key', 'meeting_key', 'circuit_short_name', 'country_name', 'date_start']],
            on='session_key',
            how='left'
        )
        
        race_results_df = race_results_df.merge(
            self.meetings[['meeting_key', 'meeting_name', 'year', 'location']],
            on='meeting_key',
            how='left'
        )
        
        print(f"Extracted {len(race_results_df)} race results")
        return race_results_df
    
    def calculate_driver_stats(self, race_results_df):
        """Calculate historical driver performance statistics"""
        print("\nCalculating driver statistics...")
        
        # Sort by date
        race_results_df = race_results_df.sort_values('date_start')
        
        driver_stats = []
        
        for idx, row in race_results_df.iterrows():
            driver_num = row['driver_number']
            current_date = row['date_start']
            
            # Get historical races for this driver before current race
            historical = race_results_df[
                (race_results_df['driver_number'] == driver_num) &
                (race_results_df['date_start'] < current_date)
            ]
            
            if len(historical) == 0:
                # No history - use neutral values
                stats = {
                    'driver_races_count': 0,
                    'driver_avg_position': 10.5,
                    'driver_best_position': 20,
                    'driver_worst_position': 20,
                    'driver_podiums': 0,
                    'driver_wins': 0,
                    'driver_top5': 0,
                    'driver_top10': 0,
                    'driver_dnf_rate': 0.0,
                    'driver_recent_form': 10.5  # Last 3 races avg
                }
            else:
                recent = historical.tail(3)
                
                stats = {
                    'driver_races_count': len(historical),
                    'driver_avg_position': historical['final_position'].mean(),
                    'driver_best_position': historical['final_position'].min(),
                    'driver_worst_position': historical['final_position'].max(),
                    'driver_podiums': (historical['final_position'] <= 3).sum(),
                    'driver_wins': (historical['final_position'] == 1).sum(),
                    'driver_top5': (historical['final_position'] <= 5).sum(),
                    'driver_top10': (historical['final_position'] <= 10).sum(),
                    'driver_dnf_rate': (historical['final_position'] > 18).mean(),
                    'driver_recent_form': recent['final_position'].mean()
                }
            
            driver_stats.append({
                'session_key': row['session_key'],
                'driver_number': driver_num,
                **stats
            })
        
        driver_stats_df = pd.DataFrame(driver_stats)
        print(f"Calculated stats for {len(driver_stats_df)} driver-race combinations")
        return driver_stats_df
    
    def calculate_circuit_stats(self, race_results_df):
        """Calculate circuit-specific statistics"""
        print("\nCalculating circuit statistics...")
        
        circuit_stats = []
        
        for idx, row in race_results_df.iterrows():
            circuit = row['circuit_short_name']
            driver_num = row['driver_number']
            current_date = row['date_start']
            
            # Get historical performance at this circuit for this driver
            circuit_history = race_results_df[
                (race_results_df['circuit_short_name'] == circuit) &
                (race_results_df['driver_number'] == driver_num) &
                (race_results_df['date_start'] < current_date)
            ]
            
            if len(circuit_history) == 0:
                stats = {
                    'circuit_driver_races': 0,
                    'circuit_driver_avg_position': 10.5,
                    'circuit_driver_best_position': 20
                }
            else:
                stats = {
                    'circuit_driver_races': len(circuit_history),
                    'circuit_driver_avg_position': circuit_history['final_position'].mean(),
                    'circuit_driver_best_position': circuit_history['final_position'].min()
                }
            
            circuit_stats.append({
                'session_key': row['session_key'],
                'driver_number': driver_num,
                **stats
            })
        
        circuit_stats_df = pd.DataFrame(circuit_stats)
        print(f"Calculated circuit stats for {len(circuit_stats_df)} combinations")
        return circuit_stats_df
    
    def add_lap_statistics(self, race_results_df):
        """Add lap time statistics from qualifying/practice"""
        print("\nAdding lap statistics...")
        
        lap_stats = []
        
        for session_key in race_results_df['session_key'].unique():
            # Get meeting key for this race
            meeting_key = race_results_df[race_results_df['session_key'] == session_key]['meeting_key'].iloc[0]
            
            # Get all sessions for this meeting
            meeting_sessions = self.sessions[self.sessions['meeting_key'] == meeting_key]
            
            # Get qualifying session
            quali_session = meeting_sessions[meeting_sessions['session_type'] == 'Qualifying']
            
            if len(quali_session) > 0:
                quali_key = quali_session.iloc[0]['session_key']
                quali_laps = self.laps[self.laps['session_key'] == quali_key]
                
                # Calculate best lap time for each driver
                if len(quali_laps) > 0:
                    # Filter out invalid laps
                    valid_laps = quali_laps[quali_laps['lap_duration'].notna()]
                    
                    if len(valid_laps) > 0:
                        best_laps = valid_laps.groupby('driver_number')['lap_duration'].min().reset_index()
                        best_laps.columns = ['driver_number', 'quali_best_lap']
                        
                        # Calculate gap to fastest
                        fastest_lap = best_laps['quali_best_lap'].min()
                        best_laps['quali_gap_to_fastest'] = best_laps['quali_best_lap'] - fastest_lap
                        
                        best_laps['session_key'] = session_key
                        lap_stats.append(best_laps)
        
        if len(lap_stats) > 0:
            lap_stats_df = pd.concat(lap_stats, ignore_index=True)
            print(f"Added lap stats for {len(lap_stats_df)} driver-race combinations")
            return lap_stats_df
        else:
            print("No lap statistics available")
            return pd.DataFrame()
    
    def add_weather_features(self, race_results_df):
        """Add weather conditions"""
        print("\nAdding weather features...")
        
        weather_features = []
        
        for session_key in race_results_df['session_key'].unique():
            session_weather = self.weather[self.weather['session_key'] == session_key]
            
            if len(session_weather) > 0:
                # Get average weather conditions
                avg_weather = {
                    'session_key': session_key,
                    'air_temperature': session_weather['air_temperature'].mean(),
                    'track_temperature': session_weather['track_temperature'].mean(),
                    'humidity': session_weather['humidity'].mean(),
                    'wind_speed': session_weather['wind_speed'].mean(),
                    'rainfall': session_weather['rainfall'].mean() if 'rainfall' in session_weather.columns else 0
                }
                weather_features.append(avg_weather)
        
        if len(weather_features) > 0:
            weather_df = pd.DataFrame(weather_features)
            print(f"Added weather for {len(weather_df)} races")
            return weather_df
        else:
            print("No weather data available")
            return pd.DataFrame()
    
    def add_pit_stop_features(self, race_results_df):
        """Add pit stop statistics"""
        print("\nAdding pit stop features...")
        
        pit_features = []
        
        for session_key in race_results_df['session_key'].unique():
            session_pits = self.pit_stops[self.pit_stops['session_key'] == session_key]
            
            if len(session_pits) > 0:
                # Count pit stops per driver
                pit_counts = session_pits.groupby('driver_number').size().reset_index()
                pit_counts.columns = ['driver_number', 'pit_stop_count']
                
                # Average pit duration
                pit_durations = session_pits.groupby('driver_number')['pit_duration'].mean().reset_index()
                pit_durations.columns = ['driver_number', 'avg_pit_duration']
                
                # Merge
                pit_stats = pit_counts.merge(pit_durations, on='driver_number', how='left')
                pit_stats['session_key'] = session_key
                pit_features.append(pit_stats)
        
        if len(pit_features) > 0:
            pit_df = pd.concat(pit_features, ignore_index=True)
            print(f"Added pit stop stats for {len(pit_df)} driver-race combinations")
            return pit_df
        else:
            print("No pit stop data available")
            return pd.DataFrame()
    
    def create_ml_dataset(self):
        """Create final ML-ready dataset"""
        print("\n" + "=" * 50)
        print("Creating ML Dataset")
        print("=" * 50)
        
        # Step 1: Get race results (target variable)
        race_results = self.prepare_race_results()
        
        # Step 2: Calculate driver statistics
        driver_stats = self.calculate_driver_stats(race_results)
        ml_data = race_results.merge(driver_stats, on=['session_key', 'driver_number'], how='left')
        
        # Step 3: Calculate circuit statistics
        circuit_stats = self.calculate_circuit_stats(race_results)
        ml_data = ml_data.merge(circuit_stats, on=['session_key', 'driver_number'], how='left')
        
        # Step 4: Add lap statistics
        lap_stats = self.add_lap_statistics(race_results)
        if len(lap_stats) > 0:
            ml_data = ml_data.merge(lap_stats, on=['session_key', 'driver_number'], how='left')
        
        # Step 5: Add weather features
        weather_features = self.add_weather_features(race_results)
        if len(weather_features) > 0:
            ml_data = ml_data.merge(weather_features, on='session_key', how='left')
        
        # Step 6: Add pit stop features
        pit_features = self.add_pit_stop_features(race_results)
        if len(pit_features) > 0:
            ml_data = ml_data.merge(pit_features, on=['session_key', 'driver_number'], how='left')
        
        # Fill missing values
        ml_data = ml_data.fillna({
            'quali_best_lap': ml_data['quali_best_lap'].median() if 'quali_best_lap' in ml_data.columns else 90,
            'quali_gap_to_fastest': 2.0,
            'pit_stop_count': 2,
            'avg_pit_duration': 25.0,
            'air_temperature': 25.0,
            'track_temperature': 35.0,
            'humidity': 50.0,
            'wind_speed': 5.0,
            'rainfall': 0.0
        })
        
        # Save processed dataset
        output_path = "data/processed/f1_ml_dataset.csv"
        ml_data.to_csv(output_path, index=False)
        
        print("\n" + "=" * 50)
        print("ML Dataset Created Successfully!")
        print("=" * 50)
        print(f"\nDataset shape: {ml_data.shape}")
        print(f"Saved to: {output_path}")
        print(f"\nFeatures: {ml_data.columns.tolist()}")
        print(f"\nTarget variable: final_position")
        print(f"Target distribution:\n{ml_data['final_position'].value_counts().sort_index().head(10)}")
        
        return ml_data

if __name__ == "__main__":
    engineer = F1FeatureEngineer()
    ml_dataset = engineer.create_ml_dataset()

