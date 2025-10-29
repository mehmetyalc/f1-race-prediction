"""
Enhanced F1 ML Dataset Preparation
Integrates qualifying, race pace, and team performance features
"""

import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedF1DatasetBuilder:
    """Builds enhanced ML dataset with critical features"""
    
    def __init__(self):
        self.load_all_data()
    
    def load_all_data(self):
        """Load all data files"""
        print("Loading all F1 data...")
        
        # Original data
        self.meetings = pd.read_csv('data/raw/meetings.csv')
        self.sessions = pd.read_csv('data/raw/sessions.csv')
        self.drivers = pd.read_csv('data/raw/drivers.csv')
        self.laps = pd.read_csv('data/raw/laps.csv')
        self.positions = pd.read_csv('data/raw/positions.csv')
        self.pit_stops = pd.read_csv('data/raw/pit_stops.csv')
        self.weather = pd.read_csv('data/raw/weather.csv')
        
        # Enhanced data
        self.qualifying = pd.read_csv('data/raw/qualifying_results.csv')
        self.race_pace = pd.read_csv('data/raw/race_pace.csv')
        self.team_performance = pd.read_csv('data/raw/team_performance.csv')
        
        print("✓ All data loaded successfully")
    
    def get_race_results(self):
        """Extract race results with final positions"""
        print("\n[1/8] Extracting race results...")
        
        # Get race sessions only
        race_sessions = self.sessions[self.sessions['session_name'] == 'Race']
        
        results = []
        
        for _, session in race_sessions.iterrows():
            session_key = session['session_key']
            
            # Get drivers in this session
            session_drivers = self.drivers[self.drivers['session_key'] == session_key]
            
            for _, driver in session_drivers.iterrows():
                driver_num = driver['driver_number']
                
                # Get final position from positions data
                driver_positions = self.positions[
                    (self.positions['session_key'] == session_key) &
                    (self.positions['driver_number'] == driver_num)
                ]
                
                if len(driver_positions) > 0:
                    # Get last recorded position
                    final_position = driver_positions.iloc[-1]['position']
                    
                    results.append({
                        'session_key': session_key,
                        'driver_number': driver_num,
                        'final_position': final_position,
                        'date': session['date_start'],
                        'meeting_key': session['meeting_key']
                    })
        
        df = pd.DataFrame(results)
        print(f"  ✓ Extracted {len(df)} race results")
        
        return df
    
    def add_qualifying_features(self, df):
        """Add qualifying position features"""
        print("\n[2/8] Adding qualifying features...")
        
        # Merge qualifying results
        df = df.merge(
            self.qualifying[['session_key', 'driver_number', 'qualifying_position']],
            on=['session_key', 'driver_number'],
            how='left'
        )
        
        # Calculate grid position advantage (qualifying vs final)
        df['grid_position_gain'] = df['qualifying_position'] - df['final_position']
        
        # Fill missing qualifying positions with median
        df['qualifying_position'].fillna(df['qualifying_position'].median(), inplace=True)
        df['grid_position_gain'].fillna(0, inplace=True)
        
        print(f"  ✓ Added qualifying features")
        print(f"    - qualifying_position")
        print(f"    - grid_position_gain")
        
        return df
    
    def add_race_pace_features(self, df):
        """Add race pace features"""
        print("\n[3/8] Adding race pace features...")
        
        # Merge race pace data
        df = df.merge(
            self.race_pace[['session_key', 'driver_number', 'avg_race_pace', 
                           'best_race_pace', 'pace_consistency', 'clean_laps_count']],
            on=['session_key', 'driver_number'],
            how='left'
        )
        
        # Fill missing values
        df['avg_race_pace'].fillna(df['avg_race_pace'].median(), inplace=True)
        df['best_race_pace'].fillna(df['best_race_pace'].median(), inplace=True)
        df['pace_consistency'].fillna(df['pace_consistency'].median(), inplace=True)
        df['clean_laps_count'].fillna(0, inplace=True)
        
        print(f"  ✓ Added race pace features")
        print(f"    - avg_race_pace")
        print(f"    - best_race_pace")
        print(f"    - pace_consistency")
        print(f"    - clean_laps_count")
        
        return df
    
    def add_team_features(self, df):
        """Add team performance features"""
        print("\n[4/8] Adding team features...")
        
        # Get team name for each driver
        driver_teams = self.drivers[['session_key', 'driver_number', 'team_name']].drop_duplicates()
        
        df = df.merge(
            driver_teams,
            on=['session_key', 'driver_number'],
            how='left'
        )
        
        # Calculate team statistics per session
        team_stats = []
        
        for session_key in df['session_key'].unique():
            session_df = df[df['session_key'] == session_key]
            
            for team in session_df['team_name'].unique():
                if pd.isna(team):
                    continue
                
                team_drivers = session_df[session_df['team_name'] == team]
                
                team_stats.append({
                    'session_key': session_key,
                    'team_name': team,
                    'team_avg_position': team_drivers['final_position'].mean(),
                    'team_best_position': team_drivers['final_position'].min(),
                    'team_driver_count': len(team_drivers)
                })
        
        team_stats_df = pd.DataFrame(team_stats)
        
        # Merge team stats
        df = df.merge(
            team_stats_df,
            on=['session_key', 'team_name'],
            how='left'
        )
        
        # Fill missing values
        df['team_avg_position'].fillna(10, inplace=True)
        df['team_best_position'].fillna(10, inplace=True)
        df['team_driver_count'].fillna(2, inplace=True)
        
        print(f"  ✓ Added team features")
        print(f"    - team_avg_position")
        print(f"    - team_best_position")
        print(f"    - team_driver_count")
        
        return df
    
    def add_driver_historical_features(self, df):
        """Add driver historical performance features"""
        print("\n[5/8] Adding driver historical features...")
        
        # Sort by date
        df = df.sort_values('date')
        
        driver_stats = []
        
        for idx, row in df.iterrows():
            driver_num = row['driver_number']
            current_date = row['date']
            
            # Get all previous races for this driver
            prev_races = df[
                (df['driver_number'] == driver_num) &
                (df['date'] < current_date)
            ]
            
            if len(prev_races) > 0:
                stats = {
                    'driver_races_count': len(prev_races),
                    'driver_avg_position': prev_races['final_position'].mean(),
                    'driver_best_position': prev_races['final_position'].min(),
                    'driver_worst_position': prev_races['final_position'].max(),
                    'driver_podiums': (prev_races['final_position'] <= 3).sum(),
                    'driver_wins': (prev_races['final_position'] == 1).sum(),
                    'driver_top5': (prev_races['final_position'] <= 5).sum(),
                    'driver_top10': (prev_races['final_position'] <= 10).sum(),
                }
                
                # Recent form (last 3 races)
                recent_races = prev_races.tail(3)
                stats['driver_recent_form'] = recent_races['final_position'].mean()
            else:
                stats = {
                    'driver_races_count': 0,
                    'driver_avg_position': 10,
                    'driver_best_position': 10,
                    'driver_worst_position': 10,
                    'driver_podiums': 0,
                    'driver_wins': 0,
                    'driver_top5': 0,
                    'driver_top10': 0,
                    'driver_recent_form': 10
                }
            
            driver_stats.append(stats)
        
        driver_stats_df = pd.DataFrame(driver_stats)
        df = pd.concat([df.reset_index(drop=True), driver_stats_df], axis=1)
        
        print(f"  ✓ Added driver historical features")
        
        return df
    
    def add_circuit_features(self, df):
        """Add circuit-specific features"""
        print("\n[6/8] Adding circuit features...")
        
        # Get circuit info from meetings
        circuit_info = self.meetings[['meeting_key', 'circuit_short_name', 'country_name', 'location']]
        
        df = df.merge(circuit_info, on='meeting_key', how='left')
        
        # Calculate driver performance at each circuit
        circuit_stats = []
        
        for idx, row in df.iterrows():
            driver_num = row['driver_number']
            circuit = row['circuit_short_name']
            current_date = row['date']
            
            # Get previous races at this circuit
            prev_circuit_races = df[
                (df['driver_number'] == driver_num) &
                (df['circuit_short_name'] == circuit) &
                (df['date'] < current_date)
            ]
            
            if len(prev_circuit_races) > 0:
                stats = {
                    'circuit_driver_races': len(prev_circuit_races),
                    'circuit_driver_avg_position': prev_circuit_races['final_position'].mean(),
                    'circuit_driver_best_position': prev_circuit_races['final_position'].min()
                }
            else:
                stats = {
                    'circuit_driver_races': 0,
                    'circuit_driver_avg_position': row['driver_avg_position'],
                    'circuit_driver_best_position': row['driver_best_position']
                }
            
            circuit_stats.append(stats)
        
        circuit_stats_df = pd.DataFrame(circuit_stats)
        df = pd.concat([df.reset_index(drop=True), circuit_stats_df], axis=1)
        
        print(f"  ✓ Added circuit features")
        
        return df
    
    def add_weather_features(self, df):
        """Add weather features"""
        print("\n[7/8] Adding weather features...")
        
        # Calculate average weather per session
        weather_stats = self.weather.groupby('session_key').agg({
            'air_temperature': 'mean',
            'track_temperature': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'rainfall': 'max'  # Max rainfall probability
        }).reset_index()
        
        df = df.merge(weather_stats, on='session_key', how='left')
        
        # Fill missing weather data
        df['air_temperature'].fillna(df['air_temperature'].median(), inplace=True)
        df['track_temperature'].fillna(df['track_temperature'].median(), inplace=True)
        df['humidity'].fillna(df['humidity'].median(), inplace=True)
        df['wind_speed'].fillna(df['wind_speed'].median(), inplace=True)
        df['rainfall'].fillna(0, inplace=True)
        
        print(f"  ✓ Added weather features")
        
        return df
    
    def add_pit_stop_features(self, df):
        """Add pit stop strategy features"""
        print("\n[8/8] Adding pit stop features...")
        
        # Calculate pit stop stats per driver per session
        pit_stats = self.pit_stops.groupby(['session_key', 'driver_number']).agg({
            'pit_duration': ['count', 'mean']
        }).reset_index()
        
        pit_stats.columns = ['session_key', 'driver_number', 'pit_stop_count', 'avg_pit_duration']
        
        df = df.merge(pit_stats, on=['session_key', 'driver_number'], how='left')
        
        # Fill missing pit stop data
        df['pit_stop_count'].fillna(0, inplace=True)
        df['avg_pit_duration'].fillna(df['avg_pit_duration'].median(), inplace=True)
        
        print(f"  ✓ Added pit stop features")
        
        return df
    
    def build_enhanced_dataset(self):
        """Build complete enhanced ML dataset"""
        print("\n" + "=" * 60)
        print("BUILDING ENHANCED ML DATASET")
        print("=" * 60)
        
        # Start with race results
        df = self.get_race_results()
        
        # Add all feature groups
        df = self.add_qualifying_features(df)
        df = self.add_race_pace_features(df)
        df = self.add_team_features(df)
        df = self.add_driver_historical_features(df)
        df = self.add_circuit_features(df)
        df = self.add_weather_features(df)
        df = self.add_pit_stop_features(df)
        
        # Select only numeric features for ML
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifiers, keep only features
        exclude_cols = ['session_key', 'driver_number', 'meeting_key']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        ml_df = df[feature_cols]
        
        # Save dataset
        ml_df.to_csv('data/processed/f1_ml_dataset_enhanced.csv', index=False)
        
        print("\n" + "=" * 60)
        print("ENHANCED ML DATASET CREATED!")
        print("=" * 60)
        print(f"\nDataset shape: {ml_df.shape}")
        print(f"Features: {len(ml_df.columns) - 1}")  # Exclude target
        print(f"Samples: {len(ml_df)}")
        print(f"\nTarget variable: final_position")
        print(f"\nFeature categories:")
        print(f"  - Qualifying: 2 features")
        print(f"  - Race Pace: 4 features")
        print(f"  - Team Performance: 3 features")
        print(f"  - Driver Historical: 9 features")
        print(f"  - Circuit-Specific: 3 features")
        print(f"  - Weather: 5 features")
        print(f"  - Pit Strategy: 2 features")
        print(f"\nTotal: {len(ml_df.columns)} columns")
        
        return ml_df

if __name__ == "__main__":
    builder = EnhancedF1DatasetBuilder()
    dataset = builder.build_enhanced_dataset()

