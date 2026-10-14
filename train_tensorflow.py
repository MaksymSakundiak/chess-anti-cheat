#!/usr/bin/env python3
"""
TensorFlow Training Script for Chess Anti-Cheat Detection
Advanced pattern recognition with TensorFlow/Keras
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Import our advanced model
try:
    from improved_tensorflow_model import AdvancedChessClassifier, build_vocab
except ImportError:
    print("❌ improved_tensorflow_model not found. Make sure improved_tensorflow_model.py is in the current directory.")
    exit(1)

def plot_training_history(history):
    """Plot TensorFlow training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Training AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Validation AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Precision
    axes[1, 1].plot(history.history['precision'], label='Training Precision')
    axes[1, 1].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 1].set_title('Model Precision')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('tensorflow_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def split_by_player(df, test_fraction=0.2, random_state=42):
    """Split so each player's games land in exactly one side, keeping the
    cheater/honest ratio. A row-level split puts the same player's games in
    both sides, letting the model memorize players instead of detecting cheating."""
    n_splits = round(1 / test_fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df['label'], groups=df['username'].str.lower()))
    return df.iloc[train_idx], df.iloc[test_idx]

def main():
    """Main TensorFlow training function"""
    print("🎯 TensorFlow Chess Anti-Cheat Training")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 Found {len(gpus)} GPU(s): {gpus}")
        # Enable memory growth to avoid GPU memory issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("🖥️ Using CPU for training")
    
    # Load data
    hdf5_file = "data/combined_games.h5"
    if not os.path.exists(hdf5_file):
        print(f"❌ {hdf5_file} not found. Please run fetch_games.py first.")
        return
    
    df = pd.read_hdf(hdf5_file, key='chess_data')
    print(f"✅ Loaded {len(df)} records from {hdf5_file}")
    
    if 'username' not in df.columns:
        print("❌ Dataset has no 'username' column (it predates the player-level split). "
              "Delete the files in data/ and re-run fetch_games.py.")
        return
    
    # Split by player: test players and validation players are never seen in training
    print("🔄 Splitting data by player...")
    train_df, test_df = split_by_player(df, test_fraction=0.2)
    train_df, val_df = split_by_player(train_df, test_fraction=0.2)
    
    print(f"📊 Data split:")
    for name, part in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
        players = part.groupby(part['username'].str.lower())['label'].first()
        print(f"   - {name}: {len(part)} games from {len(players)} players "
              f"({int(players.sum())} cheaters, {int(part['label'].sum())} cheater games)")
    
    # Build vocabulary from training players only, so test moves can't leak in
    vocab = build_vocab(train_df)
    vocab_file = "hdf5_vocab.pkl"
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✅ Built vocabulary with {len(vocab)} tokens -> {vocab_file}")
    
    # Create and train model
    print("🏗️ Building advanced TensorFlow model...")
    classifier = AdvancedChessClassifier(vocab_size=len(vocab))
    
    # Train the model
    print("🚀 Starting training...")
    history = classifier.train(
        df=train_df, 
        vocab=vocab, 
        epochs=10,  # Reduced epochs for faster training
        batch_size=64,
        val_df=val_df
    )
    
    # Plot training history
    print("📈 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("🔍 Evaluating on test set...")
    X_moves_test, X_times_test, X_color_test, y_test = classifier.prepare_data(test_df, vocab)
    results = classifier.evaluate(X_moves_test, X_times_test, X_color_test, y_test)
    
    # Save the model (native Keras format)
    out_path = 'advanced_chess_model.keras'
    classifier.save_model(out_path)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"📊 Final Test Results:")
    print(f"   - Accuracy: {results['accuracy']:.4f} (always-majority baseline: {results['baseline_accuracy']:.4f})")
    print(f"   - Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"   - Cheater Precision: {results['precision']:.4f}")
    print(f"   - Cheater Recall: {results['recall']:.4f}")
    print(f"   - Cheater F1 Score: {results['f1']:.4f}")
    print(f"   - AUC: {results['auc']:.4f}")
    print(f"\n💾 Model saved as: {out_path}")
    print(f"📈 Training plot saved as: tensorflow_training_history.png")
    
    # Comparison note
    if os.path.exists('best_hdf5_model.pt'):
        print(f"\n📊 Comparison:")
        print(f"   - TensorFlow model: {results['auc']:.4f} AUC")
        print(f"   - PyTorch model: legacy (best_hdf5_model.pt)")
        print(f"   - Recommendation: Use the model with higher AUC")

if __name__ == "__main__":
    main()
