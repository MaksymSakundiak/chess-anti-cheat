# HDF5 Format for Chess Anti-Cheat Detection

## 🎯 Overview

This document explains the new HDF5 data format that enables **player-level cheating detection** instead of game-level detection. The system now analyzes individual players (white/black) separately, allowing for more granular and accurate cheating detection.

## 📊 Data Format

### HDF5 Structure
The data is stored in HDF5 format with the following structure:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `game_id` | string | Unique game identifier | `"001_000001"` |
| `player_color` | string | Player's color | `"white"` or `"black"` |
| `player_id` | string | Unique player identifier | `"player_001_000001_white"` |
| `opponent_id` | string | Opponent's identifier | `"player_001_000001_black"` |
| `moves` | string | Comma-separated moves | `"e4,Nf3,Bb5,Nc6,O-O"` |
| `move_times` | string | Comma-separated move times | `"2.1,1.9,2.3,2.0,1.8"` |
| `result` | string | Game result for this player | `"win"`, `"loss"`, or `"draw"` |
| `label` | int | Cheating label (0=honest, 1=cheater) | `0` or `1` |

### Example Data
```python
# Sample HDF5 record
{
    'game_id': '001_000001',
    'player_color': 'white',
    'player_id': 'player_001_000001_white',
    'opponent_id': 'player_001_000001_black',
    'moves': 'e4,Nf3,Bb5,Nc6,O-O',
    'move_times': '2.1,1.9,2.3,2.0,1.8',
    'result': 'win',
    'label': 1  # Cheater
}
```

## 🔄 Conversion Process

### From CSV to HDF5
The conversion process transforms game-level data to player-level data:

1. **Parse Games**: Split each game into white and black perspectives
2. **Extract Moves**: Separate moves by player color
3. **Generate Times**: Create realistic move times based on complexity
4. **Determine Results**: Calculate game results for each player
5. **Create Records**: Generate separate records for each player

### Conversion Script
```bash
python convert_to_hdf5.py
```

This script:
- Converts `data/cheater_games.csv` → `data/cheater_games.h5`
- Converts `data/honest_games.csv` → `data/honest_games.h5`
- Combines into `data/combined_games.h5`

## 🎯 Key Advantages

### 1. **Player-Level Granularity**
- **Before**: Model analyzed entire games (both players combined)
- **After**: Model analyzes individual players separately
- **Benefit**: Can identify which specific player is cheating

### 2. **Improved Accuracy**
- **Focused Analysis**: Model sees only one player's moves at a time
- **Cleaner Patterns**: No mixing of different playing styles
- **Better Training**: More precise training data

### 3. **Enhanced Detection**
- **White vs Black**: Can detect if white or black player is cheating
- **Individual Assessment**: Each player gets their own cheating score
- **Pattern Recognition**: Better identification of player-specific patterns

### 4. **Real-World Application**
- **Live Games**: Can analyze individual players in real-time
- **Tournament Monitoring**: Track specific players across multiple games
- **Player Profiling**: Build individual player risk profiles

## 🚀 Usage

### Training with HDF5 Data
```bash
python train_hdf5.py
```

### Prediction with HDF5 Model
```bash
python predict_hdf5.py
```

### Interactive Analysis
```python
from predict_hdf5 import HDF5ChessPredictor

# Initialize predictor
predictor = HDF5ChessPredictor()

# Analyze player moves
result = predictor.predict_player_moves("e4,Nf3,Bb5,Nc6,O-O")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📈 Data Statistics

### Current Dataset
- **Total Player Records**: 138,234
- **Cheaters**: 64,914 (47%)
- **Honest Players**: 73,320 (53%)
- **White Players**: 69,117
- **Black Players**: 69,117
- **Unique Games**: 69,117

### Game Results Distribution
- **Draws**: 100,270 (72.5%)
- **Wins**: 18,982 (13.7%)
- **Losses**: 18,982 (13.7%)

### Move Statistics
- **Average Moves per Player**: 47.6
- **Median Moves per Player**: 43.0
- **Min Moves**: 5
- **Max Moves**: 300

### Time Statistics
- **Average Move Time**: 3.11 seconds
- **Median Move Time**: 3.15 seconds
- **Min Move Time**: 0.80 seconds
- **Max Move Time**: 5.40 seconds

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **CSV Loading**: Load original game data
2. **Move Parsing**: Split moves by player color
3. **Time Generation**: Create realistic move times
4. **Record Creation**: Generate player-level records
5. **HDF5 Storage**: Save in efficient HDF5 format

### Model Architecture
- **Input**: Player moves (comma-separated)
- **Processing**: Transformer-LSTM hybrid
- **Output**: Cheating probability (0-1)
- **Features**: Move patterns, timing, complexity

### Training Process
- **Dataset Split**: 70% train, 10% validation, 20% test
- **Loss Function**: Focal Loss (handles class imbalance)
- **Optimization**: Adam with learning rate scheduling
- **Regularization**: Early stopping, gradient clipping

## 🎯 Detection Capabilities

### Player-Level Analysis
- **Move Complexity**: Analyze individual move sophistication
- **Tactical Accuracy**: Assess tactical precision
- **Timing Patterns**: Examine move timing consistency
- **Style Consistency**: Check for player-specific patterns

### Risk Factors
- **High Tactical Accuracy**: Suspicious engine-like precision
- **Consistent Timing**: Unrealistic move time consistency
- **Complex Move Patterns**: Unusually sophisticated play
- **Low Move Variety**: Limited repertoire

### Confidence Levels
- **Very High**: >80% confidence
- **High**: 60-80% confidence
- **Medium**: 40-60% confidence
- **Low**: 30-40% confidence
- **Very Low**: <30% confidence

## 📁 File Structure

```
chess-anti-cheat/
├── convert_to_hdf5.py          # Convert CSV to HDF5
├── hdf5_preprocess.py          # HDF5 data preprocessing
├── train_hdf5.py              # Train model on HDF5 data
├── predict_hdf5.py            # Predict with HDF5 model
├── data/
│   ├── cheater_games.h5       # Cheater data (HDF5)
│   ├── honest_games.h5        # Honest data (HDF5)
│   └── combined_games.h5      # Combined dataset (HDF5)
├── hdf5_vocab.pkl            # Vocabulary for HDF5 data
├── hdf5_chess_classifier.pt  # Trained HDF5 model
└── HDF5_FORMAT_GUIDE.md      # This guide
```

## 🔄 Migration from Old Format

### Old Format (Game-Level)
```python
# Old CSV format
{
    'moves': 'e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O',
    'label': 1  # Entire game labeled as cheater
}
```

### New Format (Player-Level)
```python
# New HDF5 format
{
    'game_id': '001_000001',
    'player_color': 'white',
    'player_id': 'player_001_000001_white',
    'moves': 'e4,Nf3,Bb5,Ba4,O-O',
    'label': 1  # Only white player labeled as cheater
}
```

## 🎯 Benefits Summary

1. **Granular Detection**: Identify specific cheating players
2. **Improved Accuracy**: Better pattern recognition
3. **Real-Time Analysis**: Analyze individual players live
4. **Enhanced Features**: Player-specific move and timing analysis
5. **Scalable Architecture**: Efficient HDF5 storage and processing
6. **Better Training**: Cleaner, more focused training data

This new HDF5 format represents a significant advancement in chess anti-cheat detection, enabling more precise and actionable cheating detection at the individual player level. 