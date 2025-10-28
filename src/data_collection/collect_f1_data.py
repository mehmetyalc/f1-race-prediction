"""
Formula 1 Data Collection from OpenF1 API
Collects historical race data including meetings, sessions, drivers, laps, and results
"""

import requests
import pandas as pd
import time
from datetime import datetime
import json
import os

class F1DataCollector:
    """Collects F1 data from OpenF1 API"""
    
    def __init__(self, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
        self.data_dir = "data/raw"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_data(self, endpoint, params=None):
        """Generic method to fetch data from API"""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            time.sleep(0.5)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return None
    
    def collect_meetings(self, years=None):
        """Collect Grand Prix meetings data"""
        if years is None:
            years = list(range(2023, 2026))  # 2023-2025
        
        all_meetings = []
        for year in years:
            print(f"Collecting meetings for {year}...")
            data = self.get_data("meetings", params={"year": year})
            if data:
                all_meetings.extend(data)
        
        df = pd.DataFrame(all_meetings)
        df.to_csv(f"{self.data_dir}/meetings.csv", index=False)
        print(f"Collected {len(df)} meetings")
        return df
    
    def collect_sessions(self, meeting_keys=None):
        """Collect session data for meetings"""
        if meeting_keys is None:
            meetings_df = pd.read_csv(f"{self.data_dir}/meetings.csv")
            meeting_keys = meetings_df['meeting_key'].unique()
        
        all_sessions = []
        for meeting_key in meeting_keys:
            print(f"Collecting sessions for meeting {meeting_key}...")
            data = self.get_data("sessions", params={"meeting_key": meeting_key})
            if data:
                all_sessions.extend(data)
        
        df = pd.DataFrame(all_sessions)
        df.to_csv(f"{self.data_dir}/sessions.csv", index=False)
        print(f"Collected {len(df)} sessions")
        return df
    
    def collect_drivers(self, session_keys=None):
        """Collect driver data for sessions"""
        if session_keys is None:
            sessions_df = pd.read_csv(f"{self.data_dir}/sessions.csv")
            # Only get race sessions
            race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
            session_keys = race_sessions['session_key'].unique()
        
        all_drivers = []
        for session_key in session_keys[:50]:  # Limit for initial collection
            print(f"Collecting drivers for session {session_key}...")
            data = self.get_data("drivers", params={"session_key": session_key})
            if data:
                all_drivers.extend(data)
        
        df = pd.DataFrame(all_drivers)
        df.to_csv(f"{self.data_dir}/drivers.csv", index=False)
        print(f"Collected {len(df)} driver records")
        return df
    
    def collect_laps(self, session_keys=None, limit=10):
        """Collect lap data for race sessions"""
        if session_keys is None:
            sessions_df = pd.read_csv(f"{self.data_dir}/sessions.csv")
            race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
            session_keys = race_sessions['session_key'].unique()[:limit]
        
        all_laps = []
        for session_key in session_keys:
            print(f"Collecting laps for session {session_key}...")
            data = self.get_data("laps", params={"session_key": session_key})
            if data:
                all_laps.extend(data)
        
        df = pd.DataFrame(all_laps)
        df.to_csv(f"{self.data_dir}/laps.csv", index=False)
        print(f"Collected {len(df)} lap records")
        return df
    
    def collect_positions(self, session_keys=None, limit=10):
        """Collect position data for race sessions"""
        if session_keys is None:
            sessions_df = pd.read_csv(f"{self.data_dir}/sessions.csv")
            race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
            session_keys = race_sessions['session_key'].unique()[:limit]
        
        all_positions = []
        for session_key in session_keys:
            print(f"Collecting positions for session {session_key}...")
            data = self.get_data("position", params={"session_key": session_key})
            if data:
                all_positions.extend(data)
        
        df = pd.DataFrame(all_positions)
        df.to_csv(f"{self.data_dir}/positions.csv", index=False)
        print(f"Collected {len(df)} position records")
        return df
    
    def collect_pit_stops(self, session_keys=None, limit=10):
        """Collect pit stop data"""
        if session_keys is None:
            sessions_df = pd.read_csv(f"{self.data_dir}/sessions.csv")
            race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
            session_keys = race_sessions['session_key'].unique()[:limit]
        
        all_pits = []
        for session_key in session_keys:
            print(f"Collecting pit stops for session {session_key}...")
            data = self.get_data("pit", params={"session_key": session_key})
            if data:
                all_pits.extend(data)
        
        df = pd.DataFrame(all_pits)
        df.to_csv(f"{self.data_dir}/pit_stops.csv", index=False)
        print(f"Collected {len(df)} pit stop records")
        return df
    
    def collect_weather(self, session_keys=None, limit=10):
        """Collect weather data"""
        if session_keys is None:
            sessions_df = pd.read_csv(f"{self.data_dir}/sessions.csv")
            race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
            session_keys = race_sessions['session_key'].unique()[:limit]
        
        all_weather = []
        for session_key in session_keys:
            print(f"Collecting weather for session {session_key}...")
            data = self.get_data("weather", params={"session_key": session_key})
            if data:
                all_weather.extend(data)
        
        df = pd.DataFrame(all_weather)
        df.to_csv(f"{self.data_dir}/weather.csv", index=False)
        print(f"Collected {len(df)} weather records")
        return df
    
    def collect_starting_grid(self, session_keys=None, limit=10):
        """Collect starting grid positions"""
        if session_keys is None:
            sessions_df = pd.read_csv(f"{self.data_dir}/sessions.csv")
            race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
            session_keys = race_sessions['session_key'].unique()[:limit]
        
        all_grids = []
        for session_key in session_keys:
            print(f"Collecting starting grid for session {session_key}...")
            data = self.get_data("starting_grid", params={"session_key": session_key})
            if data:
                all_grids.extend(data)
        
        df = pd.DataFrame(all_grids)
        df.to_csv(f"{self.data_dir}/starting_grid.csv", index=False)
        print(f"Collected {len(df)} starting grid records")
        return df
    
    def collect_all_data(self, years=None):
        """Collect all F1 data"""
        print("=" * 50)
        print("Starting F1 Data Collection")
        print("=" * 50)
        
        # Step 1: Collect meetings
        print("\n[1/8] Collecting meetings...")
        meetings_df = self.collect_meetings(years)
        
        # Step 2: Collect sessions
        print("\n[2/8] Collecting sessions...")
        sessions_df = self.collect_sessions()
        
        # Get race sessions only
        race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
        session_keys = race_sessions['session_key'].unique()[:20]  # Limit to 20 races
        
        # Step 3: Collect drivers
        print("\n[3/8] Collecting drivers...")
        self.collect_drivers(session_keys)
        
        # Step 4: Collect laps
        print("\n[4/8] Collecting laps...")
        self.collect_laps(session_keys, limit=20)
        
        # Step 5: Collect positions
        print("\n[5/8] Collecting positions...")
        self.collect_positions(session_keys, limit=20)
        
        # Step 6: Collect pit stops
        print("\n[6/8] Collecting pit stops...")
        self.collect_pit_stops(session_keys, limit=20)
        
        # Step 7: Collect weather
        print("\n[7/8] Collecting weather...")
        self.collect_weather(session_keys, limit=20)
        
        # Step 8: Collect starting grid
        print("\n[8/8] Collecting starting grid...")
        self.collect_starting_grid(session_keys, limit=20)
        
        print("\n" + "=" * 50)
        print("Data Collection Complete!")
        print("=" * 50)
        
        # Print summary
        print("\nData Summary:")
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(f"{self.data_dir}/{file}")
                print(f"  {file}: {len(df)} records")

if __name__ == "__main__":
    collector = F1DataCollector()
    collector.collect_all_data(years=[2023, 2024, 2025])

