#!/usr/bin/env python3
"""
Advanced TensorFlow/Keras Model for Chess Anti-Cheat Detection
Simplified architecture with better pattern recognition capabilities
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Bidirectional, Dropout, Embedding, 
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
    Input, Concatenate, BatchNormalization, Masking
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

# Per-move feature channels fed to the model
MOVE_FEATURES = ['is_move', 'log_time', 'has_eval', 'eval_loss']

def _parse_floats(values, max_len: int) -> List:
    """Comma-separated numbers -> list; empty entries become None"""
    if isinstance(values, str):
        items = values.split(',') if values else []
        return [float(v) if v.strip() else None for v in items][:max_len]
    if isinstance(values, (list, np.ndarray)):
        return list(values)[:max_len]
    return []

def encode_move_features(times_str, losses_str, max_len: int) -> np.ndarray:
    """Encode each move as [is_move, log1p(seconds), has_eval, log1p(win% loss)].

    is_move marks real moves so padding rows are all-zero and can be masked.
    has_eval separates "unanalysed" from "perfect move", which would otherwise
    both look like a 0 loss. Times use a fixed log transform (per-game z-scoring
    erased absolute speed and consistency).
    """
    times = _parse_floats(times_str, max_len)
    losses = _parse_floats(losses_str, max_len)
    features = np.zeros((max_len, len(MOVE_FEATURES)), dtype=np.float32)
    for i, t in enumerate(times):
        features[i, 0] = 1.0
        features[i, 1] = np.log1p(max(t or 0.0, 0.0))
        if i < len(losses) and losses[i] is not None:
            features[i, 2] = 1.0
            features[i, 3] = np.log1p(min(max(losses[i], 0.0), 100.0))
    return features

def as_model_inputs(model: Model, X_moves: np.ndarray, X_features: np.ndarray,
                    X_color: np.ndarray) -> Dict[str, np.ndarray]:
    """Inputs keyed by name, limited to those the model declares (a model
    trained without move tokens has no moves_input)"""
    available = {'moves_input': X_moves, 'features_input': X_features, 'color_input': X_color}
    return {t.name: available[t.name] for t in model.inputs}

# Keras' default (0.99) leaves inference-time BatchNorm statistics lagging far
# behind training on datasets this small, squashing predictions toward the prior
BN_MOMENTUM = 0.9

class AdvancedChessClassifier:
    """Advanced TensorFlow/Keras model for chess anti-cheat detection"""
    
    def __init__(self, vocab_size: int, max_len: int = 300, embed_dim: int = 64,
                 lstm_units: int = 64, dense_units: Tuple[int, ...] = (128, 64),
                 dropout: float = 0.4, use_moves: bool = True):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        # The move-token branch can memorise players' openings; set False to
        # train on timing and engine-accuracy features only
        self.use_moves = use_moves
        self.model = None
        
    def build_model(self) -> Model:
        """Build the network: move tokens + per-move timing/accuracy features + color"""
        
        # Input layers
        moves_input = Input(shape=(self.max_len,), name='moves_input')
        features_input = Input(shape=(self.max_len, len(MOVE_FEATURES)), name='features_input')
        color_input = Input(shape=(1,), name='color_input')
        branches = []
        
        if self.use_moves:
            # Move tokens: embedding -> BiLSTM -> local patterns
            move_embedding = Embedding(self.vocab_size, self.embed_dim, mask_zero=True)(moves_input)
            lstm_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(move_embedding)
            lstm_out = Dropout(self.dropout)(lstm_out)
            conv = Conv1D(64, 5, activation='relu', padding='same')(lstm_out)
            conv = BatchNormalization(momentum=BN_MOMENTUM)(conv)
            branches.append(GlobalMaxPooling1D()(conv))
        
        # Per-move timing and engine-accuracy features. Masking drops padding
        # rows, so the average is over real moves only (e.g. mean accuracy).
        masked = Masking(mask_value=0.0)(features_input)
        per_move = Dense(32, activation='relu')(masked)
        per_move = Dense(32, activation='relu')(per_move)
        local = Conv1D(32, 5, activation='relu', padding='same')(features_input)
        branches += [
            GlobalAveragePooling1D()(per_move),
            GlobalMaxPooling1D()(local),
        ]
        
        # Color features
        branches.append(Dense(8, activation='relu')(color_input))
        
        x = Concatenate()(branches)
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization(momentum=BN_MOMENTUM)(x)
            x = Dropout(self.dropout)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        inputs = {'features_input': features_input, 'color_input': color_input}
        if self.use_moves:
            inputs['moves_input'] = moves_input
        self.model = Model(
            inputs=inputs,
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
            # Stop once validation AUC stops improving; the model starts
            # memorising training players within a few epochs
            EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
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
                histogram_freq=0,
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
        has_losses = 'move_losses' in df.columns
        
        for _, row in df.iterrows():
            tokens = encode_moves(row['moves'], vocab, self.max_len)
            times = encode_move_features(row['move_times'], row['move_losses'] if has_losses else '',
                                         self.max_len)
            
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
        print(f"   - Move features shape: {X_times.shape}")
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
        
        # Build and compile model
        self.build_model()
        self.compile_model()
        
        if val_df is not None:
            X_moves_val, X_times_val, X_color_val, y_val = self.prepare_data(val_df, vocab)
            validation = {'validation_data': (
                as_model_inputs(self.model, X_moves_val, X_times_val, X_color_val), y_val)}
        else:
            validation = {'validation_split': validation_split}
        
        # Weight classes inversely to frequency so the minority (cheaters) isn't ignored
        counts = np.bincount(y.astype(int), minlength=2)
        class_weight = {c: len(y) / (2 * counts[c]) for c in (0, 1) if counts[c] > 0}
        print(f"⚖️ Class weights: {class_weight}")
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        history = self.model.fit(
            as_model_inputs(self.model, X_moves, X_times, X_color),
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
                X_color: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict:
        """Evaluate the model"""
        
        print("🔍 Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict(as_model_inputs(self.model, X_moves, X_times, X_color), verbose=0)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, confusion_matrix,
            precision_recall_fscore_support, roc_auc_score
        )
        
        # Convert predictions to binary
        y_pred_binary = (predictions > threshold).astype(int)
        
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
        
        print(f"📊 Evaluation Results (threshold {threshold:.3f}):")
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
