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
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def build_vocab(df: pd.DataFrame, min_freq: int = 3) -> Dict[str, int]:
    """Build a move vocabulary; moves seen fewer than min_freq times map to <unk>"""
    move_counts = Counter(
        move.strip()
        for moves_str in df['moves'] if isinstance(moves_str, str)
        for move in moves_str.split(',') if move.strip()
    )
    vocab = {'<PAD>': 0, '<unk>': 1}
    for move, count in move_counts.items():
        if count >= min_freq:
            vocab[move] = len(vocab)
    return vocab

def encode_moves(moves_str: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """Tokenize comma-separated moves, truncated / padded to max_len"""
    moves = moves_str.split(',') if isinstance(moves_str, str) else []
    tokens = [vocab.get(move.strip(), vocab['<unk>']) for move in moves if move.strip()]
    tokens = tokens[:max_len]
    return tokens + [vocab['<PAD>']] * (max_len - len(tokens))

def encode_times(times_str, max_len: int) -> np.ndarray:
    """Encode move times (seconds) as log1p, truncated / zero-padded to max_len.

    A fixed transform keeps absolute speed and consistency visible to the model
    (per-game z-scoring erased both) and leaves padding at exactly 0.
    """
    if isinstance(times_str, str):
        times = [float(t) for t in times_str.split(',') if t.strip()]
    else:
        times = list(times_str) if isinstance(times_str, (list, np.ndarray)) else []
    encoded = np.zeros(max_len, dtype=np.float32)
    times = np.log1p(np.clip(times[:max_len], 0, None))
    encoded[:len(times)] = times
    return encoded

# Keras' default (0.99) leaves inference-time BatchNorm statistics lagging far
# behind training on datasets this small, squashing predictions toward the prior
BN_MOMENTUM = 0.9

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
        conv1 = BatchNormalization(momentum=BN_MOMENTUM)(conv1)
        conv1 = MaxPooling1D(2)(conv1)
        
        conv2 = Conv1D(128, 5, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization(momentum=BN_MOMENTUM)(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        
        conv3 = Conv1D(256, 7, activation='relu', padding='same')(conv2)
        conv3 = BatchNormalization(momentum=BN_MOMENTUM)(conv3)
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
        dense1 = BatchNormalization(momentum=BN_MOMENTUM)(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(256, activation='relu')(dense1)
        dense2 = BatchNormalization(momentum=BN_MOMENTUM)(dense2)
        dense2 = Dropout(0.3)(dense2)
        
        dense3 = Dense(128, activation='relu')(dense2)
        dense3 = BatchNormalization(momentum=BN_MOMENTUM)(dense3)
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
            BinaryAccuracy(name='accuracy'),
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
            tokens = encode_moves(row['moves'], vocab, self.max_len)
            times = encode_times(row['move_times'], self.max_len)
            
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
              validation_split: float = 0.2,
              val_df: pd.DataFrame = None) -> Dict:
        """Train the advanced model.

        Pass val_df (split by player) to validate on unseen players; otherwise
        Keras holds out the last validation_split of df.
        """
        
        print("🚀 Starting advanced TensorFlow training...")
        
        # Prepare data
        X_moves, X_times, X_color, y = self.prepare_data(df, vocab)
        if val_df is not None:
            X_moves_val, X_times_val, X_color_val, y_val = self.prepare_data(val_df, vocab)
            validation = {'validation_data': ([X_moves_val, X_times_val, X_color_val], y_val)}
        else:
            validation = {'validation_split': validation_split}
        
        # Weight classes inversely to frequency so the minority (cheaters) isn't ignored
        counts = np.bincount(y.astype(int), minlength=2)
        class_weight = {c: len(y) / (2 * counts[c]) for c in (0, 1) if counts[c] > 0}
        print(f"⚖️ Class weights: {class_weight}")
        
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
            class_weight=class_weight,
            callbacks=callbacks,
            **validation,
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
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, confusion_matrix,
            precision_recall_fscore_support, roc_auc_score
        )
        
        # Convert predictions to binary
        y_pred_binary = (predictions > 0.5).astype(int)
        
        # Precision / recall / F1 for the cheater class: weighted averages are
        # dominated by the honest majority and hide missed cheaters
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred_binary, pos_label=1, average='binary', zero_division=0
        )
        cm = confusion_matrix(y, y_pred_binary, labels=[0, 1])
        auc = roc_auc_score(y, predictions)
        
        results = {
            'accuracy': accuracy_score(y, y_pred_binary),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred_binary),
            'baseline_accuracy': max(np.mean(y), 1 - np.mean(y)),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': predictions.flatten(),
            'binary_predictions': y_pred_binary.flatten()
        }
        
        print("📊 Evaluation Results:")
        print(f"  Accuracy: {results['accuracy']:.4f} (always-majority baseline: {results['baseline_accuracy']:.4f})")
        print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"  Cheater Precision: {results['precision']:.4f}")
        print(f"  Cheater Recall: {results['recall']:.4f}")
        print(f"  Cheater F1 Score: {results['f1']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  Confusion matrix [[TN FP] [FN TP]]: {cm.tolist()}")
        
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
    """Train via train_tensorflow.py, which splits by player and builds the vocabulary"""
    import train_tensorflow
    train_tensorflow.main()

if __name__ == "__main__":
    main()
