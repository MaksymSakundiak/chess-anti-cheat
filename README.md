# Chess Anti-Cheat Detection with TensorFlow

Advanced chess anti-cheat detection system using TensorFlow/Keras for better pattern recognition.

## 🚀 Features

- **Advanced TensorFlow Model**: Bidirectional LSTM + Conv1D layers for superior pattern recognition
- **Multi-Modal Input**: Processes moves, timing, and player color simultaneously
- **Real-time Analysis**: Interactive prediction for both white and black players
- **Comprehensive Risk Analysis**: Detects timing patterns, tactical accuracy, and move complexity
- **Clean Data Pipeline**: Optimized HDF5 data format with essential features only

## 📊 Model Architecture

```
Input Layers:
├── Moves (Embedding + Bidirectional LSTM)
├── Move Times (Dense + Normalization)
└── Player Color (Dense)

Feature Extraction:
├── Conv1D Layers (3, 5, 7 kernel sizes)
├── Batch Normalization
└── Global Max Pooling

Classification:
├── Dense Layers (512 → 256 → 128)
├── Dropout (0.4 → 0.3 → 0.2)
└── Sigmoid Output
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/MaksymSakundiak/chess-anti-cheat.git
cd chess-anti-cheat

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
chess-anti-cheat/
├── data/
│   ├── combined_games.h5          # Clean dataset
│   ├── cheater_games.h5           # Cheater games
│   └── honest_games.h5            # Honest games
├── improved_tensorflow_model.py    # Advanced TensorFlow model
├── train_tensorflow.py            # Training script
├── predict_tensorflow.py          # Prediction script
├── fetch_games.py                 # Data fetching
├── convert_to_hdf5.py            # Data conversion
├── hdf5_preprocess.py            # Data preprocessing
└── requirements.txt               # Dependencies
```

## 🎯 Quick Start

### 1. Fetch Data

```bash
python fetch_games.py
```

### 2. Train TensorFlow Model

```bash
python train_tensorflow.py
```

### 3. Make Predictions

```bash
python predict_tensorflow.py
```

## 📈 Model Performance

The TensorFlow model provides:

- **Better Pattern Recognition**: Conv1D layers detect local patterns
- **Improved Timing Analysis**: Normalized time features
- **Enhanced Architecture**: Bidirectional LSTM + Attention mechanisms
- **Advanced Callbacks**: Early stopping, learning rate reduction, TensorBoard

## 🔍 Usage Examples

### Interactive Prediction

```bash
python predict_tensorflow.py

# Enter game data:
# White moves: e4,e5,Nf3,Nc6
# Black moves: d5,d6,Nf6,Nd7
# White times: 2.1,1.8,3.2,2.9
# Black times: 1.9,2.4,3.1,2.6
# White result: win
# Black result: loss
```

### Programmatic Usage

```python
from improved_tensorflow_model import AdvancedChessClassifier

# Load model
classifier = AdvancedChessClassifier()
classifier.load_model('advanced_chess_model.h5')

# Make prediction
result = classifier.predict_player_moves(
    moves="e4,e5,Nf3,Nc6",
    move_times="2.1,1.8,3.2,2.9",
    player_color="white"
)
```

## 📊 Data Format

The cleaned dataset contains only essential features:

| Column         | Description                      | Example           |
| -------------- | -------------------------------- | ----------------- |
| `player_color` | "white" or "black"               | "white"           |
| `moves`        | Comma-separated chess moves      | "e4,e5,Nf3,Nc6"   |
| `move_times`   | Comma-separated times in seconds | "2.1,1.8,3.2,2.9" |
| `result`       | "win", "loss", or "draw"         | "win"             |
| `label`        | 0 (honest) or 1 (cheater)        | 1                 |

## 🎯 Key Improvements

### TensorFlow Advantages

- **Better Pattern Recognition**: Conv1D layers detect local move patterns
- **Improved Timing Analysis**: Normalized time features with statistical analysis
- **Enhanced Architecture**: Bidirectional LSTM + Attention mechanisms
- **Advanced Training**: Early stopping, learning rate reduction, TensorBoard logging

### Data Pipeline Improvements

- **Clean Dataset**: Removed useless columns (game_id, player_id, opponent_id)
- **Optimized Format**: Only essential features for cheating detection
- **Better Preprocessing**: Normalized timing data and improved feature engineering

### Model Architecture

- **Multi-Modal Input**: Processes moves, timing, and player color simultaneously
- **Advanced Layers**: Conv1D, Bidirectional LSTM, Batch Normalization
- **Comprehensive Analysis**: Risk factors, timing patterns, tactical accuracy

## 🔧 Configuration

### Model Parameters

```python
# In improved_tensorflow_model.py
class AdvancedChessClassifier:
    def __init__(self, vocab_size: int, max_len: int = 300, embed_dim: int = 128):
        # Adjust these parameters for different performance
```

### Training Parameters

```python
# In train_tensorflow.py
history = classifier.train(
    df=train_df,
    vocab=vocab,
    epochs=30,        # Increase for better performance
    batch_size=64,    # Adjust based on memory
    validation_split=0.2
)
```

## 📈 Performance Metrics

The TensorFlow model tracks:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to cheaters
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## 🚀 Advanced Features

### Risk Factor Analysis

- **Timing Patterns**: Detects suspiciously fast or consistent move times
- **Tactical Accuracy**: Identifies unusually high tactical move ratios
- **Move Complexity**: Analyzes move complexity patterns
- **Move Variety**: Detects limited move variety (engine-like behavior)

### Real-time Analysis

- **Interactive Interface**: Easy-to-use command-line interface
- **Comprehensive Reports**: Detailed analysis for both players
- **Risk Assessment**: Multiple risk factors with confidence levels
- **Game Summary**: Overall assessment of the game

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the TensorFlow model
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for advanced neural network capabilities
- Lichess API for chess game data
- Chess community for feedback and testing
