#!/usr/bin/env python3
"""
Advanced TensorFlow/Keras Model for Chess Anti-Cheat Detection
Simplified architecture with better pattern recognition capabilities
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Bidirectional, Dropout, Embedding, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Input, Concatenate, BatchNormalization, Flatten,
    Reshape, Lambda
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedChessClassifier:
    """Advanced TensorFlow/Keras model for chess anti-cheat detection"""
    
    def __init__(self, vocab_size: int, max_len: int = 300, embed_dim: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self) -> Model:
        """Build advanced neural network architecture"""
        
        # Input layers
        moves_input = Input(shape=(self.max_len,), name='moves_input')
        times_input = Input(shape=(self.max_len,), name='times_input')
        color_input = Input(shape=(1,), name='color_input')
        
        # Move embedding and processing
        move_embedding = Embedding(self.vocab_size, self.embed_dim, 
                                 input_length=self.max_len, 
                                 mask_zero=True)(moves_input)
        
        # Bidirectional LSTM for move sequence
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(move_embedding)
        lstm_out = Dropout(0.3)(lstm_out)
        
        # Conv1D layers for local pattern detection
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(lstm_out)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(2)(conv1)
        
        conv2 = Conv1D(128, 5, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        
        conv3 = Conv1D(256, 7, activation='relu', padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = GlobalMaxPooling1D()(conv3)
        
        # Time features processing - FIXED: Reshape to 3D for GlobalMaxPooling1D
        times_reshaped = Reshape((self.max_len, 1))(times_input)
        times_dense = Dense(64, activation='relu')(times_reshaped)
        times_dense = Dropout(0.2)(times_dense)
        times_pooled = GlobalMaxPooling1D()(times_dense)
        
        # Color features
        color_dense = Dense(32, activation='relu')(color_input)
        color_dense = Dropout(0.1)(color_dense)
        
        # Combine all features
        combined = Concatenate()([conv3, times_pooled, color_dense])
        
        # Dense layers with residual connections
        dense1 = Dense(512, activation='relu')(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        dense3 = Dense(128, activation='relu')(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = Dropout(0.2)(dense3)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(dense3)
        
        # Create model
        self.model = Model(
            inputs=[moves_input, times_input, color_input],
            outputs=output,
            name='AdvancedChessClassifier'
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with advanced settings"""
        
        optimizer = Adam(learning_rate=learning_rate)
        loss = BinaryCrossentropy(label_smoothing=0.1)
        
        metrics = [
            Accuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"✅ Model compiled successfully")
        print(f"📊 Model parameters: {self.model.count_params():,}")
        
    def get_callbacks(self, model_path: str = 'best_tensorflow_model.keras'):
        """Get advanced callbacks for training"""
        
        callbacks = [
            # Early stopping with patience
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint (native Keras format)
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_auc',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def prepare_data(self, df: pd.DataFrame, vocab: Dict[str, int]) -> Tuple:
        """Prepare data for TensorFlow model"""
        
        print("🔄 Preparing data for TensorFlow model...")
        
        # Prepare move sequences
        moves_sequences = []
        times_sequences = []
        color_sequences = []
        labels = []
        
        for _, row in df.iterrows():
            # Tokenize moves
            moves_str = row['moves']
            moves = moves_str.split(',') if isinstance(moves_str, str) else []
            tokens = [vocab.get(move.strip(), vocab['<unk>']) for move in moves]
            
            # Pad or truncate
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            else:
                tokens = tokens + [vocab['<PAD>']] * (self.max_len - len(tokens))
            
            # Process move times
            times_str = row['move_times']
            if isinstance(times_str, str):
                times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
            else:
                times = times_str if isinstance(times_str, list) else []
            
            if len(times) > self.max_len:
                times = times[:self.max_len]
            else:
                times = times + [0.0] * (self.max_len - len(times))
            
            # Normalize times
            times = np.array(times)
            if np.std(times) > 0:
                times = (times - np.mean(times)) / np.std(times)
            
            # Color encoding (white=1, black=0)
            color = 1.0 if row['player_color'] == 'white' else 0.0
            
            moves_sequences.append(tokens)
            times_sequences.append(times)
            color_sequences.append([color])
            labels.append(row['label'])
        
        # Convert to numpy arrays
        X_moves = np.array(moves_sequences)
        X_times = np.array(times_sequences)
        X_color = np.array(color_sequences)
        y = np.array(labels)
        
        print(f"✅ Data prepared:")
        print(f"   - Moves shape: {X_moves.shape}")
        print(f"   - Times shape: {X_times.shape}")
        print(f"   - Color shape: {X_color.shape}")
        print(f"   - Labels shape: {y.shape}")
        
        return X_moves, X_times, X_color, y
    
    def train(self, df: pd.DataFrame, vocab: Dict[str, int], 
              epochs: int = 50, batch_size: int = 32, 
              validation_split: float = 0.2) -> Dict:
        """Train the advanced model"""
        
        print("🚀 Starting advanced TensorFlow training...")
        
        # Prepare data
        X_moves, X_times, X_color, y = self.prepare_data(df, vocab)
        
        # Build and compile model
        self.build_model()
        self.compile_model()
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        history = self.model.fit(
            [X_moves, X_times, X_color],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def evaluate(self, X_moves: np.ndarray, X_times: np.ndarray, 
                X_color: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate the model"""
        
        print("🔍 Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict([X_moves, X_times, X_color])
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Convert predictions to binary
        y_pred_binary = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        report = classification_report(y, y_pred_binary, output_dict=True)
        cm = confusion_matrix(y, y_pred_binary)
        auc = roc_auc_score(y, predictions)
        
        results = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': predictions.flatten(),
            'binary_predictions': y_pred_binary.flatten()
        }
        
        print("📊 Evaluation Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")
        
        return results
    
    def save_model(self, model_path: str = 'advanced_chess_model.keras'):
        """Save the trained model (native Keras format)"""
        self.model.save(model_path)
        print(f"✅ Model saved to {model_path}")
    
    def load_model(self, model_path: str = 'advanced_chess_model.keras'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")

def main():
    """Test the advanced TensorFlow model"""
    print("🎯 Advanced TensorFlow Chess Anti-Cheat Model")
    print("=" * 50)
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found. Please install: pip install tensorflow")
        return
    
    # Load data
    try:
        import pandas as pd
        df = pd.read_hdf('data/combined_games.h5', key='chess_data')
        print(f"✅ Loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Load vocabulary
    try:
        with open('hdf5_vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        print(f"✅ Loaded vocabulary with {len(vocab)} tokens")
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return
    
    # Create and train model
    classifier = AdvancedChessClassifier(vocab_size=len(vocab))
    
    # Train the model
    history = classifier.train(df, vocab, epochs=20, batch_size=64)
    
    # Save the model
    classifier.save_model()
    
    print("🎉 Advanced TensorFlow training complete!")

if __name__ == "__main__":
    main()
