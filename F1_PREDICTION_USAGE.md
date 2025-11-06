# F1 Race Position Prediction - Usage Guide

## Quick Start

### 1. Automatic Prediction (Easiest)

Run the interactive prediction tool:

```bash
python predict_race.py
```

The tool will:
- Show example predictions (Max Verstappen at Monaco, Lewis Hamilton at Silverstone)
- List all available drivers (22 drivers)
- List all available circuits (25 Grand Prix)
- Enter interactive mode for custom predictions

### 2. Interactive Mode

```
Driver name: Max VERSTAPPEN
Circuit/Grand Prix: Monaco Grand Prix
Qualifying position (default 10): 1
```

**Output:**
```
======================================================================
F1 RACE POSITION PREDICTION
======================================================================

Driver: Max VERSTAPPEN
Team: Red Bull Racing
Circuit: Monaco Grand Prix
Qualifying Position: P1

Predicted Finishing Position: P10
Expected Position Change: -9 (lose 9 positions)

Driver Statistics:
  Career Races: 190
  Average Position: P5.6
  Best Position: P1
  Podiums: 82
  Wins: 32
  Recent Form (last 5): P4.0

Circuit Performance (Monaco Grand Prix):
  Races: 1
  Average Position: P1.0
  Best Position: P1

Conditions:
  Air Temperature: 23.6°C
  Track Temperature: 34.2°C
  Humidity: 55.5%
  Wind Speed: 1.6 m/s
  Rainfall: 0.0 mm

======================================================================
```

### 3. Programmatic Usage

```python
from predict_race import F1RacePredictor

# Initialize predictor
predictor = F1RacePredictor()

# Predict race position
predicted_pos = predictor.predict_position(
    driver_name='Max VERSTAPPEN',
    circuit_name='Monaco Grand Prix',
    qualifying_pos=1,
    show_details=True
)

print(f"Predicted position: P{predicted_pos}")
```

### 4. Custom Weather Conditions

```python
from predict_race import F1RacePredictor

predictor = F1RacePredictor()

# Predict with specific weather
predicted_pos = predictor.predict_position(
    driver_name='Lewis HAMILTON',
    circuit_name='British Grand Prix',
    qualifying_pos=3,
    air_temp=18.0,        # Cooler conditions
    track_temp=22.0,
    humidity=75.0,        # Higher humidity
    wind_speed=8.0,       # Windy
    rainfall=2.5,         # Light rain
    show_details=True
)
```

### 5. Batch Predictions

```python
from predict_race import F1RacePredictor

predictor = F1RacePredictor()

# Predict multiple drivers at same race
drivers_qualifying = [
    ('Max VERSTAPPEN', 1),
    ('Sergio PEREZ', 2),
    ('Lewis HAMILTON', 3),
    ('Charles LECLERC', 4),
    ('Carlos SAINZ', 5),
]

circuit = 'Monaco Grand Prix'

print(f"Predictions for {circuit}:")
for driver, qual_pos in drivers_qualifying:
    pred_pos = predictor.predict_position(
        driver, circuit, qual_pos, show_details=False
    )
    print(f"  {driver}: P{qual_pos} -> P{pred_pos}")
```

## Example Predictions

### Example 1: Max Verstappen at Monaco

```python
predictor.predict_position('Max VERSTAPPEN', 'Monaco Grand Prix', qualifying_pos=1)
```

**Result:** P10 (loses 9 positions)
- Monaco is notoriously difficult to overtake
- Model considers historical data and track characteristics

### Example 2: Lewis Hamilton at Silverstone

```python
predictor.predict_position('Lewis HAMILTON', 'British Grand Prix', qualifying_pos=3)
```

**Result:** P10 (loses 7 positions)
- Silverstone is Hamilton's home circuit
- 9 previous races at this circuit with P6.3 average

### Example 3: Charles Leclerc at Monza

```python
predictor.predict_position('Charles LECLERC', 'Italian Grand Prix', qualifying_pos=2)
```

**Analysis:** Ferrari's home race, high-speed circuit favoring their car

## Available Drivers (22)

**Red Bull Racing**
- Max VERSTAPPEN
- Sergio PEREZ

**Mercedes**
- Lewis HAMILTON
- George RUSSELL

**Ferrari**
- Charles LECLERC
- Carlos SAINZ

**McLaren**
- Lando NORRIS
- Oscar PIASTRI

**Aston Martin**
- Fernando ALONSO
- Lance STROLL

**Alpine**
- Pierre GASLY
- Esteban OCON

**Williams**
- Alexander ALBON
- Logan SARGEANT

**AlphaTauri**
- Yuki TSUNODA
- Daniel RICCIARDO
- Nyck DE VRIES
- Liam LAWSON

**Alfa Romeo**
- Valtteri BOTTAS
- ZHOU Guanyu

**Haas F1 Team**
- Kevin MAGNUSSEN
- Nico HULKENBERG

Run `predictor.list_drivers()` to see all drivers with their teams.

## Available Circuits (25)

1. Bahrain Grand Prix (Bahrain)
2. Saudi Arabian Grand Prix (Saudi Arabia)
3. Australian Grand Prix (Australia)
4. Azerbaijan Grand Prix (Azerbaijan)
5. Miami Grand Prix (United States)
6. Monaco Grand Prix (Monaco)
7. Spanish Grand Prix (Spain)
8. Canadian Grand Prix (Canada)
9. Austrian Grand Prix (Austria)
10. British Grand Prix (Great Britain)
11. Hungarian Grand Prix (Hungary)
12. Belgian Grand Prix (Belgium)
13. Dutch Grand Prix (Netherlands)
14. Italian Grand Prix (Italy)
15. Singapore Grand Prix (Singapore)
16. Japanese Grand Prix (Japan)
17. Qatar Grand Prix (Qatar)
18. United States Grand Prix (United States)
19. Mexico City Grand Prix (Mexico)
20. São Paulo Grand Prix (Brazil)
21. Las Vegas Grand Prix (United States)
22. Abu Dhabi Grand Prix (United Arab Emirates)
23. Chinese Grand Prix (China)
24. Emilia Romagna Grand Prix (Italy)
25. Pre-Season Testing (Bahrain)

Run `predictor.list_circuits()` to see all circuits with countries.

## Understanding the Output

### Predicted Finishing Position
- **P1-P3**: Podium finish
- **P4-P10**: Points finish
- **P11-P20**: Outside points

### Expected Position Change
- **Positive (+)**: Driver expected to gain positions during race
- **Negative (-)**: Driver expected to lose positions during race
- **Zero (0)**: Driver expected to maintain qualifying position

### Driver Statistics
- **Career Races**: Total F1 races in dataset
- **Average Position**: Career average finishing position
- **Best Position**: Best career finish
- **Podiums**: Total podium finishes (P1-P3)
- **Wins**: Total race wins
- **Recent Form**: Average position in last 5 races

### Circuit Performance
Shows driver's historical performance at the specific circuit:
- **Races**: Number of races at this circuit
- **Average Position**: Average finish at this circuit
- **Best Position**: Best finish at this circuit

If no previous races: "No previous races at [circuit]"

### Conditions
Weather and track conditions used for prediction:
- **Air Temperature**: Ambient air temperature (°C)
- **Track Temperature**: Track surface temperature (°C)
- **Humidity**: Relative humidity (%)
- **Wind Speed**: Wind speed (m/s)
- **Rainfall**: Precipitation (mm)

## Tips for Best Predictions

1. **Qualifying Position**: Most important factor - use actual qualifying result
2. **Weather**: Specify conditions if known (rain significantly affects results)
3. **Driver Names**: Use exact format (e.g., "Max VERSTAPPEN" not "Verstappen")
4. **Circuit Names**: Use full Grand Prix name (e.g., "British Grand Prix" not "Silverstone")
5. **Historical Data**: Predictions most accurate for drivers/circuits with more history

## Model Performance

- **MAE (Mean Absolute Error)**: 2.24 positions
- **R² Score**: 0.613
- **Typical Accuracy**: ±2-3 positions

**What this means:**
- On average, predictions are within 2.24 positions of actual result
- Model explains 61.3% of variance in race results
- Better than baseline/random predictions

## Limitations

The model does NOT account for:
- Race incidents (crashes, mechanical failures)
- Safety car periods
- Pit stop strategy variations
- Tire degradation specifics
- Team orders
- Driver mistakes
- Weather changes during race
- Grid penalties
- Sprint race results

**Use predictions as guidance, not certainty.**

## Error Handling

**Driver not found:**
```
ValueError: Driver 'XYZ' not found. Use list_drivers() to see available drivers.
```
Solution: Check spelling or use `predictor.list_drivers()`

**Circuit not found:**
```
ValueError: Circuit 'XYZ' not found. Use list_circuits() to see available circuits.
```
Solution: Use full Grand Prix name or `predictor.list_circuits()`

**Model file missing:**
```
FileNotFoundError: results/models/rf_enhanced.pkl
```
Solution: Ensure model files are in correct directory

## Advanced Usage

### View Driver Details

```python
# Get driver statistics
stats = predictor.get_driver_stats('Max VERSTAPPEN')
print(f"Team: {stats['team']}")
print(f"Wins: {stats['wins']}")
print(f"Podiums: {stats['podiums']}")
```

### Circuit-Specific Performance

```python
# Get driver's history at specific circuit
races, avg_pos, best_pos = predictor.get_circuit_driver_stats(
    'Lewis HAMILTON', 
    'British Grand Prix'
)
print(f"Races: {races}, Avg: P{avg_pos:.1f}, Best: P{best_pos}")
```

### Team Performance

```python
# Get team statistics
team_avg, team_best, team_drivers = predictor.get_team_stats('Red Bull Racing')
print(f"Team avg position: P{team_avg:.1f}")
print(f"Team best: P{team_best}")
```

## Requirements

```
pandas
numpy
scikit-learn
pickle (built-in)
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

## Questions?

For issues or questions, please open an issue on GitHub:
https://github.com/mehmetyalc/f1-race-prediction/issues
