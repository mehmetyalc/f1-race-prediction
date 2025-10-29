"""
Enhanced F1 Data Collection
Collects additional critical features: qualifying times, race pace, team performance
"""

import requests
import pandas as pd
import time
from datetime import datetime
import numpy as np

class EnhancedF1DataCollector:
    """Collects enhanced F1 data from OpenF1 API"""
    
    def __init__(self):
        self.base_url = "https://api.openf1.org/v1"
        self.session = requests.Session()
    
    def get_qualifying_results(self):
        """Get qualifying session results for grid positions"""
        print("\n[1/3] Collecting Qualifying Results...")
        
        # Get all qualifying sessions
        sessions_url = f"{self.base_url}/sessions"
        params = {"session_type": "Qualifying"}
        
        response = self.session.get(sessions_url, params=params)
        sessions = response.json()
        
        print(f"  Found {len(sessions)} qualifying sessions")
        
        qualifying_data = []
        
        for session in sessions:
            session_key = session['session_key']
            meeting_key = session['meeting_key']
            date = session['date_start']
            
            # Get positions at end of qualifying
            positions_url = f"{self.base_url}/position"
            params = {
                "session_key": session_key
            }
            
            try:
                response = self.session.get(positions_url, params=params)
                positions = response.json()
                
                if positions:
                    # Get final qualifying positions (last recorded position for each driver)
                    driver_final_pos = {}
                    for pos in positions:
                        driver_num = pos['driver_number']
                        position = pos['position']
                        date_time = pos['date']
                        
                        if driver_num not in driver_final_pos:
                            driver_final_pos[driver_num] = {
                                'position': position,
                                'date': date_time
                            }
                        else:
                            # Keep the latest position
                            if date_time > driver_final_pos[driver_num]['date']:
                                driver_final_pos[driver_num] = {
                                    'position': position,
                                    'date': date_time
                                }
                    
                    # Save qualifying results
                    for driver_num, data in driver_final_pos.items():
                        qualifying_data.append({
                            'session_key': session_key,
                            'meeting_key': meeting_key,
                            'driver_number': driver_num,
                            'qualifying_position': data['position'],
                            'date': date
                        })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"  Error getting positions for session {session_key}: {e}")
                continue
        
        df = pd.DataFrame(qualifying_data)
        df.to_csv('data/raw/qualifying_results.csv', index=False)
        print(f"  ✓ Saved {len(df)} qualifying results")
        
        return df
    
    def get_race_pace_data(self):
        """Calculate race pace from lap times"""
        print("\n[2/3] Calculating Race Pace Data...")
        
        # Load existing lap data
        laps_df = pd.read_csv('data/raw/laps.csv')
        
        print(f"  Loaded {len(laps_df)} lap records")
        
        # Calculate clean air pace (laps without traffic)
        # Filter out slow laps (pit laps, traffic, etc.)
        race_pace_data = []
        
        for session_key in laps_df['session_key'].unique():
            session_laps = laps_df[laps_df['session_key'] == session_key]
            
            for driver_num in session_laps['driver_number'].unique():
                driver_laps = session_laps[session_laps['driver_number'] == driver_num]
                
                # Filter valid laps (not pit laps, not outliers)
                valid_laps = driver_laps[driver_laps['lap_duration'] > 0]
                
                if len(valid_laps) > 0:
                    # Remove outliers (laps > 1.5x median)
                    median_time = valid_laps['lap_duration'].median()
                    clean_laps = valid_laps[valid_laps['lap_duration'] < median_time * 1.5]
                    
                    if len(clean_laps) >= 3:
                        # Calculate statistics
                        avg_pace = clean_laps['lap_duration'].mean()
                        best_pace = clean_laps['lap_duration'].min()
                        pace_consistency = clean_laps['lap_duration'].std()
                        
                        race_pace_data.append({
                            'session_key': session_key,
                            'driver_number': driver_num,
                            'avg_race_pace': avg_pace,
                            'best_race_pace': best_pace,
                            'pace_consistency': pace_consistency,
                            'clean_laps_count': len(clean_laps)
                        })
        
        df = pd.DataFrame(race_pace_data)
        df.to_csv('data/raw/race_pace.csv', index=False)
        print(f"  ✓ Saved {len(df)} race pace records")
        
        return df
    
    def get_team_performance_data(self):
        """Calculate team performance metrics"""
        print("\n[3/3] Calculating Team Performance Data...")
        
        # Load existing data
        drivers_df = pd.read_csv('data/raw/drivers.csv')
        
        print(f"  Loaded {len(drivers_df)} driver records")
        
        # Calculate team statistics
        team_performance = []
        
        for session_key in drivers_df['session_key'].unique():
            session_drivers = drivers_df[drivers_df['session_key'] == session_key]
            
            # Group by team
            for team_name in session_drivers['team_name'].unique():
                if pd.isna(team_name):
                    continue
                
                team_drivers = session_drivers[session_drivers['team_name'] == team_name]
                
                # Get driver numbers for this team
                driver_numbers = team_drivers['driver_number'].tolist()
                
                team_performance.append({
                    'session_key': session_key,
                    'team_name': team_name,
                    'driver_count': len(driver_numbers),
                    'driver_numbers': ','.join(map(str, driver_numbers))
                })
        
        df = pd.DataFrame(team_performance)
        df.to_csv('data/raw/team_performance.csv', index=False)
        print(f"  ✓ Saved {len(df)} team performance records")
        
        return df
    
    def collect_all_enhanced_data(self):
        """Collect all enhanced data"""
        print("=" * 60)
        print("COLLECTING ENHANCED F1 DATA")
        print("=" * 60)
        
        qualifying_df = self.get_qualifying_results()
        race_pace_df = self.get_race_pace_data()
        team_perf_df = self.get_team_performance_data()
        
        print("\n" + "=" * 60)
        print("ENHANCED DATA COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Qualifying Results: {len(qualifying_df)} records")
        print(f"  Race Pace Data: {len(race_pace_df)} records")
        print(f"  Team Performance: {len(team_perf_df)} records")
        
        return qualifying_df, race_pace_df, team_perf_df

if __name__ == "__main__":
    collector = EnhancedF1DataCollector()
    collector.collect_all_enhanced_data()

