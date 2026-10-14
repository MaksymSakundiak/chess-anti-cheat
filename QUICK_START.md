# Quick Start Guide - TensorFlow Chess Anti-Cheat

## 🚀 Fast Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train TensorFlow Model
```bash
python train_tensorflow.py
```

### 3. Make Predictions
```bash
python predict_tensorflow.py
```

## 📊 What's New in TensorFlow Version

### ✅ **Improved Architecture**
- **Bidirectional LSTM**: Better sequence understanding
- **Conv1D Layers**: Local pattern detection
- **Batch Normalization**: Stable training
- **Advanced Callbacks**: TensorBoard, early stopping

### ✅ **Better Pattern Recognition**
- **Multi-Modal Input**: Moves + timing + color
- **Real Move Times**: Taken from Lichess clock data
- **Risk Factor Analysis**: Comprehensive cheating detection

### ✅ **Clean Data Pipeline**
- **Essential Features Only**: Removed useless columns
- **Optimized Format**: Faster processing
- **Player-Level Split**: Test players are never seen in training

## 🎯 Example Usage

### Interactive Prediction
```bash
python predict_tensorflow.py

# Enter game data:
White moves: e4,e5,Nf3,Nc6,Bb5
Black moves: d5,d6,Nf6,Nd7,Bd7
White times: 2.1,1.8,3.2,2.9,4.1
Black times: 1.9,2.4,3.1,2.6,3.8
White result: win
Black result: loss
```

### Expected Output
```
============================================================
WHITE PLAYER ANALYSIS
============================================================
Player Color: white
PREDICTION: Honest
Confidence: 0.2345 (Low)
Move Count: 5
Average Move Time: 2.82s

Explanation: The TensorFlow model shows low confidence in this prediction. 
Analyzed 5 moves. Average move time: 2.8s. Found 2 tactical moves. 
Player color: white. 1 medium-risk factors detected.

Risk Factors:
  1. High move complexity (Risk: MEDIUM)
     Value: 0.800
     Description: Unusually complex moves, which might indicate engine assistance
============================================================
```

## 📈 Model Performance

### Training Metrics
Training prints test-set accuracy next to the always-"honest" baseline, plus
balanced accuracy, cheater precision/recall/F1 and AUC. With ~13% cheater
games, accuracy alone is misleading: judge the model by balanced accuracy,
cheater recall and AUC on the player-level test split.

### Detection Capabilities
- **Timing Patterns**: Suspiciously fast/consistent moves
- **Tactical Accuracy**: Unrealistic precision
- **Move Complexity**: Engine-like sophistication
- **Move Variety**: Limited variety detection

## 🔧 Configuration Options

### Model Parameters
```python
# In improved_tensorflow_model.py
class AdvancedChessClassifier:
    def __init__(self, vocab_size: int, max_len: int = 300, embed_dim: int = 128):
        # Adjust for different performance
```

### Training Parameters
```python
# In train_tensorflow.py
history = classifier.train(
    df=train_df, 
    vocab=vocab, 
    epochs=10,        # More epochs = better performance
    batch_size=64,    # Adjust based on memory
    val_df=val_df     # Validation players, split by player
)
```

## 🛠️ Troubleshooting

### Common Issues

**1. TensorFlow not found**
```bash
pip install tensorflow
```

**2. Model file not found**
```bash
# Train the model first
python train_tensorflow.py
```

**3. Memory issues**
```bash
# Reduce batch size in train_tensorflow.py
batch_size=32  # Instead of 64
```

**4. Training too slow**
```bash
# Use GPU if available
# Or reduce epochs
epochs=5  # Instead of 10
```

## 📊 Data Requirements

### Input Format
- **Moves**: Comma-separated chess notation
- **Times**: Comma-separated seconds
- **Color**: "white" or "black"
- **Result**: "win", "loss", or "draw"

### Example Data
```
White moves: e4,e5,Nf3,Nc6,Bb5
White times: 2.1,1.8,3.2,2.9,4.1
Player color: white
Result: win
```

## 🎯 Key Features

### ✅ **Advanced TensorFlow Model**
- Bidirectional LSTM + Conv1D layers
- Multi-modal input processing
- Batch normalization and dropout

### ✅ **Comprehensive Analysis**
- Risk factor detection
- Timing pattern analysis
- Tactical accuracy assessment
- Move complexity evaluation

### ✅ **Real-time Prediction**
- Interactive command-line interface
- Detailed player analysis
- Game summary assessment
- Confidence level reporting

## 🚀 Next Steps

1. **Train the model**: `python train_tensorflow.py`
2. **Test predictions**: `python predict_tensorflow.py`
3. **Analyze results**: Review risk factors and confidence levels
4. **Fine-tune parameters**: Adjust model configuration as needed

## 📈 Performance Tips

### For Better Accuracy
- Increase training epochs (30-50)
- Use larger batch sizes (if memory allows)
- Add more training data
- Fine-tune hyperparameters

### For Faster Training
- Reduce batch size
- Use GPU if available
- Reduce model complexity
- Use fewer epochs

## 🎯 Success Metrics

### Good Model Performance
- **Balanced Accuracy**: clearly above 0.5
- **AUC**: clearly above 0.5 on unseen test players
- **Accuracy**: above the always-"honest" baseline printed during training

### Expected Detection Rates
- **High Confidence**: >80% confidence
- **Medium Confidence**: 60-80% confidence
- **Low Confidence**: <60% confidence

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify TensorFlow installation
3. Ensure data format is correct
4. Review error messages carefully

---

**Ready to detect chess cheating with TensorFlow! 🎯**
