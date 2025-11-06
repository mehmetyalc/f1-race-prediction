"""
F1 Race Position Prediction Tool
Predict driver finishing position based on qualifying, team, and conditions
"""

import pickle
import pandas as pd
import numpy as np

class F1RacePredictor:
    """Predict F1 race finishing positions"""
    
    def __init__(self, model_path='results/models/rf_enhanced.pkl'):
        """Initialize predictor with trained model and historical data"""
        print("Loading F1 prediction model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("  Model loaded successfully!")
        
        print("\nLoading historical F1 data...")
        self.drivers = pd.read_csv('data/raw/drivers.csv')
        self.meetings = pd.read_csv('data/raw/meetings.csv')
        self.positions = pd.read_csv('data/raw/positions.csv')
        self.qualifying = pd.read_csv('data/raw/qualifying_results.csv')
        self.race_pace = pd.read_csv('data/raw/race_pace.csv')
        self.weather = pd.read_csv('data/raw/weather.csv')
        self.team_perf = pd.read_csv('data/raw/team_performance.csv')
        
        print(f"  Loaded {len(self.drivers)} driver records")
        print(f"  Loaded {len(self.meetings)} race meetings")
        
        # Get unique drivers and circuits
        self.driver_names = sorted(self.drivers['full_name'].unique())
        self.circuits = sorted(self.meetings['meeting_name'].unique())
        
        print(f"  Found {len(self.driver_names)} drivers")
        print(f"  Found {len(self.circuits)} circuits")
    
    def get_driver_stats(self, driver_name):
        """Get driver historical statistics"""
        driver_data = self.drivers[self.drivers['full_name'] == driver_name]
        
        if len(driver_data) == 0:
            return None
        
        # Get driver's team
        team = driver_data.iloc[0]['team_name'] if 'team_name' in driver_data.columns else 'Unknown'
        
        # Calculate driver statistics from positions data
        driver_positions = self.positions[self.positions['driver_number'].isin(driver_data['driver_number'])]
        
        if len(driver_positions) == 0:
            # Default stats for new drivers
            return {
                'team': team,
                'races_count': 0,
                'avg_position': 10.0,
                'best_position': 10,
                'worst_position': 20,
                'podiums': 0,
                'wins': 0,
                'top5': 0,
                'top10': 0,
                'recent_form': 10.0
            }
        
        stats = {
            'team': team,
            'races_count': len(driver_positions),
            'avg_position': driver_positions['position'].mean(),
            'best_position': driver_positions['position'].min(),
            'worst_position': driver_positions['position'].max(),
            'podiums': (driver_positions['position'] <= 3).sum(),
            'wins': (driver_positions['position'] == 1).sum(),
            'top5': (driver_positions['position'] <= 5).sum(),
            'top10': (driver_positions['position'] <= 10).sum(),
            'recent_form': driver_positions.tail(5)['position'].mean() if len(driver_positions) >= 5 else driver_positions['position'].mean()
        }
        
        return stats
    
    def get_circuit_driver_stats(self, driver_name, circuit_name):
        """Get driver statistics at specific circuit"""
        driver_data = self.drivers[self.drivers['full_name'] == driver_name]
        
        if len(driver_data) == 0:
            return 0, 10.0, 10
        
        # Get meetings at this circuit
        circuit_meetings = self.meetings[self.meetings['meeting_name'] == circuit_name]
        
        if len(circuit_meetings) == 0:
            return 0, 10.0, 10
        
        # Get driver positions at this circuit
        circuit_positions = self.positions[
            (self.positions['driver_number'].isin(driver_data['driver_number'])) &
            (self.positions['meeting_key'].isin(circuit_meetings['meeting_key']))
        ]
        
        if len(circuit_positions) == 0:
            return 0, 10.0, 10
        
        return (
            len(circuit_positions),
            circuit_positions['position'].mean(),
            circuit_positions['position'].min()
        )
    
    def get_team_stats(self, team_name):
        """Get team performance statistics"""
        if len(self.team_perf) == 0:
            return 10.0, 10, 2
        
        team_data = self.team_perf[self.team_perf['team_name'] == team_name]
        
        if len(team_data) == 0:
            return 10.0, 10, 2
        
        return (
            team_data['avg_position'].mean() if 'avg_position' in team_data.columns else 10.0,
            team_data['best_position'].min() if 'best_position' in team_data.columns else 10,
            team_data['driver_count'].iloc[0] if 'driver_count' in team_data.columns else 2
        )
    
    def get_weather_defaults(self):
        """Get default weather conditions"""
        if len(self.weather) == 0:
            return 25.0, 35.0, 50.0, 5.0, 0.0
        
        return (
            self.weather['air_temperature'].mean(),
            self.weather['track_temperature'].mean(),
            self.weather['humidity'].mean(),
            self.weather['wind_speed'].mean(),
            0.0  # No rain by default
        )
    
    def prepare_race_features(self, driver_name, circuit_name, qualifying_pos=10, 
                             air_temp=None, track_temp=None, humidity=None, 
                             wind_speed=None, rainfall=0.0):
        """Prepare features for race prediction"""
        print(f"\nPreparing features for: {driver_name} at {circuit_name}")
        
        # Check if driver exists
        if driver_name not in self.driver_names:
            raise ValueError(f"Driver '{driver_name}' not found. Use list_drivers() to see available drivers.")
        
        # Check if circuit exists
        if circuit_name not in self.circuits:
            raise ValueError(f"Circuit '{circuit_name}' not found. Use list_circuits() to see available circuits.")
        
        # Get driver stats
        driver_stats = self.get_driver_stats(driver_name)
        if driver_stats is None:
            raise ValueError(f"No data found for driver '{driver_name}'")
        
        # Get circuit-specific stats
        circuit_races, circuit_avg, circuit_best = self.get_circuit_driver_stats(driver_name, circuit_name)
        
        # Get team stats
        team_avg, team_best, team_drivers = self.get_team_stats(driver_stats['team'])
        
        # Get weather defaults if not provided
        if air_temp is None or track_temp is None or humidity is None or wind_speed is None:
            def_air, def_track, def_hum, def_wind, _ = self.get_weather_defaults()
            air_temp = air_temp or def_air
            track_temp = track_temp or def_track
            humidity = humidity or def_hum
            wind_speed = wind_speed or def_wind
        
        # Build features in exact training order
        features = {
            'qualifying_position': qualifying_pos,
            'grid_position_gain': 0.0,  # Unknown before race
            'avg_race_pace': 90.0,  # Default
            'best_race_pace': 88.0,  # Default
            'pace_consistency': 2.0,  # Default
            'clean_laps_count': 50.0,  # Default
            'team_avg_position': team_avg,
            'team_best_position': team_best,
            'team_driver_count': team_drivers,
            'driver_races_count': driver_stats['races_count'],
            'driver_avg_position': driver_stats['avg_position'],
            'driver_best_position': driver_stats['best_position'],
            'driver_worst_position': driver_stats['worst_position'],
            'driver_podiums': driver_stats['podiums'],
            'driver_wins': driver_stats['wins'],
            'driver_top5': driver_stats['top5'],
            'driver_top10': driver_stats['top10'],
            'driver_recent_form': driver_stats['recent_form'],
            'circuit_driver_races': circuit_races,
            'circuit_driver_avg_position': circuit_avg,
            'circuit_driver_best_position': circuit_best,
            'air_temperature': air_temp,
            'track_temperature': track_temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'rainfall': rainfall,
            'pit_stop_count': 2.0,  # Default 2 stops
            'avg_pit_duration': 24.0  # Default ~24s
        }
        
        return pd.DataFrame([features]), driver_stats
    
    def predict_position(self, driver_name, circuit_name, qualifying_pos=10,
                        air_temp=None, track_temp=None, humidity=None,
                        wind_speed=None, rainfall=0.0, show_details=True):
        """Predict race finishing position"""
        
        # Prepare features
        features_df, driver_stats = self.prepare_race_features(
            driver_name, circuit_name, qualifying_pos,
            air_temp, track_temp, humidity, wind_speed, rainfall
        )
        
        # Make prediction
        predicted_position = self.model.predict(features_df)[0]
        predicted_position = max(1, min(20, round(predicted_position)))  # Clamp to 1-20
        
        # Display results
        print("\n" + "=" * 70)
        print("F1 RACE POSITION PREDICTION")
        print("=" * 70)
        print(f"\nDriver: {driver_name}")
        print(f"Team: {driver_stats['team']}")
        print(f"Circuit: {circuit_name}")
        print(f"Qualifying Position: P{qualifying_pos}")
        
        print(f"\nPredicted Finishing Position: P{predicted_position}")
        
        position_change = qualifying_pos - predicted_position
        if position_change > 0:
            print(f"Expected Position Change: +{position_change} (gain {position_change} positions)")
        elif position_change < 0:
            print(f"Expected Position Change: {position_change} (lose {abs(position_change)} positions)")
        else:
            print(f"Expected Position Change: 0 (maintain position)")
        
        if show_details:
            print(f"\nDriver Statistics:")
            print(f"  Career Races: {driver_stats['races_count']}")
            print(f"  Average Position: P{driver_stats['avg_position']:.1f}")
            print(f"  Best Position: P{driver_stats['best_position']}")
            print(f"  Podiums: {driver_stats['podiums']}")
            print(f"  Wins: {driver_stats['wins']}")
            print(f"  Recent Form (last 5): P{driver_stats['recent_form']:.1f}")
            
            circuit_races, circuit_avg, circuit_best = self.get_circuit_driver_stats(driver_name, circuit_name)
            if circuit_races > 0:
                print(f"\nCircuit Performance ({circuit_name}):")
                print(f"  Races: {circuit_races}")
                print(f"  Average Position: P{circuit_avg:.1f}")
                print(f"  Best Position: P{circuit_best}")
            else:
                print(f"\nCircuit Performance: No previous races at {circuit_name}")
            
            print(f"\nConditions:")
            print(f"  Air Temperature: {features_df.iloc[0]['air_temperature']:.1f}°C")
            print(f"  Track Temperature: {features_df.iloc[0]['track_temperature']:.1f}°C")
            print(f"  Humidity: {features_df.iloc[0]['humidity']:.1f}%")
            print(f"  Wind Speed: {features_df.iloc[0]['wind_speed']:.1f} m/s")
            print(f"  Rainfall: {rainfall} mm")
        
        print("\n" + "=" * 70)
        
        return predicted_position
    
    def list_drivers(self, limit=None):
        """List all available drivers"""
        print("\nAvailable Drivers:")
        print("=" * 70)
        
        # Group by team
        driver_teams = {}
        for _, driver in self.drivers.iterrows():
            team = driver.get('team_name', 'Unknown')
            name = driver['full_name']
            if team not in driver_teams:
                driver_teams[team] = []
            if name not in driver_teams[team]:
                driver_teams[team].append(name)
        
        count = 0
        for team in sorted(driver_teams.keys()):
            print(f"\n{team}:")
            for driver in sorted(driver_teams[team]):
                count += 1
                print(f"  {count}. {driver}")
                if limit and count >= limit:
                    print(f"\n... and {len(self.driver_names) - count} more drivers")
                    print("=" * 70)
                    return
        
        print("=" * 70)
        print(f"Total: {len(self.driver_names)} drivers")
    
    def list_circuits(self):
        """List all available circuits"""
        print("\nAvailable Circuits:")
        print("=" * 70)
        for i, circuit in enumerate(self.circuits, 1):
            location = self.meetings[self.meetings['meeting_name'] == circuit].iloc[0]
            country = location.get('country_name', 'Unknown')
            print(f"{i:2d}. {circuit} ({country})")
        print("=" * 70)
        print(f"Total: {len(self.circuits)} circuits")

def main():
    """Main function for interactive prediction"""
    print("=" * 70)
    print("F1 RACE POSITION PREDICTION TOOL")
    print("=" * 70)
    
    # Initialize predictor
    predictor = F1RacePredictor()
    
    # Example predictions
    print("\n\nEXAMPLE PREDICTIONS:")
    print("=" * 70)
    
    # Example 1: Max Verstappen at Monaco
    predictor.predict_position('Max VERSTAPPEN', 'Monaco Grand Prix', qualifying_pos=1)
    
    # Example 2: Lewis Hamilton at Silverstone
    predictor.predict_position('Lewis HAMILTON', 'British Grand Prix', qualifying_pos=3)
    
    # List available options
    print("\n\n")
    predictor.list_drivers(limit=30)
    
    print("\n\n")
    predictor.list_circuits()
    
    # Interactive mode
    print("\n\nINTERACTIVE MODE")
    print("=" * 70)
    print("Enter driver and circuit to predict race position (or 'quit' to exit)")
    
    while True:
        try:
            driver = input("\nDriver name: ").strip()
            if driver.lower() == 'quit':
                break
            
            circuit = input("Circuit/Grand Prix: ").strip()
            if circuit.lower() == 'quit':
                break
            
            qual_pos = input("Qualifying position (default 10): ").strip()
            qual_pos = int(qual_pos) if qual_pos else 10
            
            predictor.predict_position(driver, circuit, qual_pos)
            
        except ValueError as e:
            print(f"\nError: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\nThank you for using F1 Race Position Predictor!")

if __name__ == "__main__":
    main()
