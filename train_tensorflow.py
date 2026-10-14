#!/usr/bin/env python3
"""
TensorFlow Training Script for Chess Anti-Cheat Detection
Advanced pattern recognition with TensorFlow/Keras
"""

import argparse
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Import our advanced model
try:
    from improved_tensorflow_model import AdvancedChessClassifier, as_model_inputs, build_vocab
except ImportError:
    print("❌ improved_tensorflow_model not found. Make sure improved_tensorflow_model.py is in the current directory.")
    exit(1)

META_FILE = 'model_meta.json'

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
    plt.close(fig)

def split_by_player(df, test_fraction=0.2, random_state=42):
    """Split so each player's games land in exactly one side, keeping the
    cheater/honest ratio. A row-level split puts the same player's games in
    both sides, letting the model memorize players instead of detecting cheating."""
    n_splits = round(1 / test_fraction)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df['label'], groups=df['username'].str.lower()))
    return df.iloc[train_idx], df.iloc[test_idx]

def best_threshold(labels, scores) -> float:
    """Threshold maximising balanced accuracy (Youden's J) on the given scores"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])

def player_scores(df: pd.DataFrame, game_scores: np.ndarray) -> pd.DataFrame:
    """Average game scores per player: accounts are judged on many games, not one"""
    return (df.assign(score=game_scores)
              .groupby(df['username'].str.lower())
              .agg(label=('label', 'first'), score=('score', 'mean'), games=('score', 'size')))

def main():
    """Main TensorFlow training function"""
    parser = argparse.ArgumentParser(description="Train the chess anti-cheat model")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--with-moves', action='store_true',
                        help='Also feed move tokens (BiLSTM branch). Off by default: it '
                             'added no accuracy in testing and is ~20x slower')
    parser.add_argument('--lstm-units', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--all-games', action='store_true',
                        help="Also train on games without engine evals. Off by default: "
                             "whether a game was analysed reflects opponents' suspicion, "
                             "not the player's play, and leaks into the label")
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Seed for the player-level train/validation/test split')
    parser.add_argument('--output', default='advanced_chess_model.keras')
    args = parser.parse_args()
    
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
        print(f"❌ {hdf5_file} not found. Please run download_dataset.py first.")
        return
    
    df = pd.read_hdf(hdf5_file, key='chess_data')
    print(f"✅ Loaded {len(df)} records from {hdf5_file}")
    
    if 'username' not in df.columns:
        print("❌ Dataset has no 'username' column (it predates the player-level split). "
              "Delete the files in data/ and re-run download_dataset.py.")
        return
    if 'move_losses' not in df.columns:
        print("⚠️ Dataset has no engine-eval column; training on timing and moves only.")
    else:
        print(f"✅ {(df['move_losses'] != '').mean():.0%} of games have engine evals")
        if not args.all_games:
            df = df[df['move_losses'] != '']
            print(f"   Keeping only analysed games: {len(df)}")
    
    # Split by player: test players and validation players are never seen in training
    print("🔄 Splitting data by player...")
    train_df, test_df = split_by_player(df, test_fraction=0.2, random_state=args.split_seed)
    train_df, val_df = split_by_player(train_df, test_fraction=0.2, random_state=args.split_seed)
    
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
    print("🏗️ Building TensorFlow model...")
    classifier = AdvancedChessClassifier(
        vocab_size=len(vocab), lstm_units=args.lstm_units,
        dropout=args.dropout, use_moves=args.with_moves
    )
    
    # Train the model
    print("🚀 Starting training...")
    history = classifier.train(
        df=train_df, 
        vocab=vocab, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_df=val_df
    )
    
    # Plot training history
    print("📈 Plotting training history...")
    plot_training_history(history)
    
    # Pick thresholds on validation players, never on the test set
    X_val = classifier.prepare_data(val_df, vocab)
    val_scores = classifier.model.predict(as_model_inputs(classifier.model, *X_val[:3]), verbose=0).ravel()
    val_players = player_scores(val_df, val_scores)
    game_threshold = best_threshold(X_val[3], val_scores)
    player_threshold = best_threshold(val_players['label'], val_players['score'])
    
    # Evaluate on test set
    print("🔍 Evaluating single games on the test set...")
    X_moves_test, X_times_test, X_color_test, y_test = classifier.prepare_data(test_df, vocab)
    results = classifier.evaluate(X_moves_test, X_times_test, X_color_test, y_test,
                                  threshold=game_threshold)
    
    # Player level: average each test player's game scores
    test_players = player_scores(test_df, results['predictions'])
    flagged = (test_players['score'] > player_threshold).astype(int)
    player_auc = roc_auc_score(test_players['label'], test_players['score'])
    player_bal_acc = balanced_accuracy_score(test_players['label'], flagged)
    player_cm = confusion_matrix(test_players['label'], flagged, labels=[0, 1])
    
    # Save the model (native Keras format) and the thresholds the predictor uses
    out_path = args.output
    classifier.save_model(out_path)
    meta = {
        'model_path': out_path,
        'game_threshold': game_threshold,
        'player_threshold': player_threshold,
        'max_len': classifier.max_len,
        'use_moves': classifier.use_moves,
        'analysed_only': not args.all_games,
        'test_game_auc': float(results['auc']),
        'test_player_auc': float(player_auc),
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"📊 Single games (threshold {game_threshold:.3f}, chosen on validation players):")
    print(f"   - Accuracy: {results['accuracy']:.4f} (always-majority baseline: {results['baseline_accuracy']:.4f})")
    print(f"   - Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"   - Cheater Precision: {results['precision']:.4f}")
    print(f"   - Cheater Recall: {results['recall']:.4f}")
    print(f"   - Cheater F1 Score: {results['f1']:.4f}")
    print(f"   - AUC: {results['auc']:.4f}")
    print(f"📊 Players (mean game score, threshold {player_threshold:.3f}):")
    print(f"   - Test players: {len(test_players)} ({int(test_players['label'].sum())} cheaters)")
    print(f"   - AUC: {player_auc:.4f}")
    print(f"   - Balanced Accuracy: {player_bal_acc:.4f}")
    print(f"   - Confusion matrix [[TN FP] [FN TP]]: {player_cm.tolist()}")
    print(f"\n💾 Model saved as: {out_path}, thresholds in {META_FILE}")
    print(f"📈 Training plot saved as: tensorflow_training_history.png")

if __name__ == "__main__":
    main()
