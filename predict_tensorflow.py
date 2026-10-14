#!/usr/bin/env python3
"""
TensorFlow Prediction Script for Chess Anti-Cheat Detection

Two modes:
  1. Score a Lichess account: downloads their recent rated rapid games, scores
     each one and averages the scores. This is the reliable mode; a single game
     carries little evidence.
  2. Score one game from typed-in moves, think times and (optionally) per-move
     engine losses.
"""

import json
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from improved_tensorflow_model import as_model_inputs, encode_move_features, encode_moves, _parse_floats

warnings.filterwarnings('ignore')

META_FILE = 'model_meta.json'
# Thresholds are tuned on players with dozens of games; a handful is too little evidence
MIN_ACCOUNT_GAMES = 10


class TensorFlowChessPredictor:
    """TensorFlow-based predictor for chess anti-cheat detection"""

    def __init__(self, model_path: Optional[str] = None, vocab_path: str = "hdf5_vocab.pkl",
                 meta_path: str = META_FILE):
        # Thresholds chosen on validation players by train_tensorflow.py
        self.meta = {'game_threshold': 0.5, 'player_threshold': 0.5, 'max_len': 300}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta.update(json.load(f))
        else:
            print(f"⚠️ {meta_path} not found; using 0.5 thresholds. Retrain to calibrate them.")

        model_path = model_path or self.meta.get('model_path', 'advanced_chess_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model at {model_path}. Train first: python train_tensorflow.py")

        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        self.model = tf.keras.models.load_model(model_path)
        self.max_len = self.meta['max_len']
        print(f"✅ TensorFlow model loaded from {model_path}")

    def score_games(self, games: List[Dict]) -> np.ndarray:
        """Cheating score (0-1) for each game dict with moves, move_times,
        player_color and optionally move_losses"""
        X_moves = np.array([encode_moves(g['moves'], self.vocab, self.max_len) for g in games])
        X_features = np.array([
            encode_move_features(g.get('move_times', ''), g.get('move_losses', ''), self.max_len)
            for g in games
        ])
        X_color = np.array([[1.0 if g.get('player_color') == 'white' else 0.0] for g in games])
        return self.model.predict(as_model_inputs(self.model, X_moves, X_features, X_color),
                                  verbose=0).ravel()

    def predict_player_moves(self, moves_str: str, move_times_str: str = "",
                             player_color: str = "white", move_losses_str: str = "") -> Dict[str, Any]:
        """Score one game from one player's perspective"""
        moves = [m.strip() for m in moves_str.split(',') if m.strip()] if moves_str else []
        if not moves:
            return {'prediction': 'Invalid', 'confidence': 0.0, 'threshold': None,
                    'explanation': 'No moves provided', 'statistics': {},
                    'player_color': player_color, 'move_count': 0}

        game = {'moves': moves_str, 'move_times': move_times_str,
                'move_losses': move_losses_str, 'player_color': player_color}
        score = float(self.score_games([game])[0])
        threshold = self.meta['game_threshold']
        return {
            'prediction': 'Suspicious' if score > threshold else 'Not suspicious',
            'confidence': score,
            'threshold': threshold,
            'explanation': ("A single game carries little evidence; score the whole "
                            "account for a reliable verdict."),
            'statistics': game_statistics(move_times_str, move_losses_str),
            'move_count': len(moves),
            'player_color': player_color,
        }

    def predict_account(self, username: str, max_games: int = 50) -> Dict[str, Any]:
        """Download a Lichess account's recent rated rapid games and score the account"""
        from fetch_games import LichessHDF5Fetcher

        fetcher = LichessHDF5Fetcher()
        response = fetcher.session.get(
            f'https://lichess.org/api/games/user/{username}',
            params={'perfType': 'rapid', 'rated': 'true', 'max': max_games,
                    'clocks': 'true', 'evals': 'true'},
            headers={'Accept': 'application/x-chess-pgn'}
        )
        if response.status_code == 404:
            return {'error': f"Account '{username}' not found or closed"}
        response.raise_for_status()

        games = [fetcher._parse_pgn_game(pgn, username)
                 for pgn in response.text.strip().split('\n\n\n') if '[Event' in pgn]
        games = [g for g in games if g]
        if self.meta.get('analysed_only'):
            # The model was trained on analysed games only; score like-for-like
            games = [g for g in games if g['move_losses']]
        if not games:
            return {'error': f"No usable rated rapid games for '{username}'"
                             + (" (the model needs games with engine analysis)"
                                if self.meta.get('analysed_only') else "")}

        scores = self.score_games(games)
        account_score = float(scores.mean())
        threshold = self.meta['player_threshold']
        all_times = ','.join(g['move_times'] for g in games)
        all_losses = ','.join(g['move_losses'] for g in games if g['move_losses'])
        return {
            'username': username,
            'games': len(games),
            'account_score': account_score,
            'threshold': threshold,
            'prediction': ('Not enough games for a verdict' if len(games) < MIN_ACCOUNT_GAMES
                           else 'Suspicious' if account_score > threshold else 'Not suspicious'),
            'most_suspicious_games': sorted(
                ({'game_id': g['game_id'], 'score': float(s)} for g, s in zip(games, scores)),
                key=lambda x: x['score'], reverse=True)[:5],
            'statistics': game_statistics(all_times, all_losses),
        }


def game_statistics(move_times_str: str, move_losses_str: str = "") -> Dict[str, float]:
    """Plain timing and accuracy statistics, for context next to the model's score.
    These describe the play; they are not verdicts on their own."""
    times = [t for t in _parse_floats(move_times_str, 100_000) if t is not None]
    losses = [l for l in _parse_floats(move_losses_str, 100_000) if l is not None]
    stats = {}
    if times:
        mean_time = float(np.mean(times))
        stats['mean_think_time_s'] = round(mean_time, 2)
        stats['think_time_variation'] = round(float(np.std(times) / mean_time), 2) if mean_time > 0 else 0.0
        stats['share_moves_under_2s'] = round(float(np.mean(np.array(times) < 2)), 2)
    if losses:
        stats['analysed_moves'] = len(losses)
        stats['mean_win_pct_loss'] = round(float(np.mean(losses)), 2)
        stats['share_near_perfect_moves'] = round(float(np.mean(np.array(losses) < 1)), 2)
        stats['share_mistakes'] = round(float(np.mean(np.array(losses) >= 10)), 2)
    return stats


STAT_LABELS = {
    'mean_think_time_s': 'Mean think time (s)',
    'think_time_variation': 'Think-time variation (std / mean)',
    'share_moves_under_2s': 'Share of moves under 2s',
    'analysed_moves': 'Moves with engine evaluation',
    'mean_win_pct_loss': 'Mean win-chance loss per move (%)',
    'share_near_perfect_moves': 'Share of near-perfect moves (<1% loss)',
    'share_mistakes': 'Share of mistakes (>=10% loss)',
}


def print_statistics(stats: Dict[str, float]):
    if not stats:
        print("Statistics: none available")
        return
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {STAT_LABELS.get(key, key)}: {value}")


def print_account_analysis(result: Dict[str, Any]):
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    print("=" * 60)
    print(f"ACCOUNT: {result['username']}")
    print("=" * 60)
    print(f"PREDICTION: {result['prediction']}")
    print(f"Account score: {result['account_score']:.4f} (threshold {result['threshold']:.4f}, "
          f"mean over {result['games']} games)")
    print("Most suspicious games:")
    for game in result['most_suspicious_games']:
        print(f"  https://lichess.org/{game['game_id']}  score {game['score']:.3f}")
    print_statistics(result['statistics'])
    print("=" * 60)


def print_player_analysis(result: Dict[str, Any], player_name: str):
    print("=" * 60)
    print(f"{player_name.upper()} ANALYSIS")
    print("=" * 60)
    print(f"Player Color: {result['player_color']}")
    print(f"PREDICTION: {result['prediction']}")
    if result['move_count']:
        print(f"Score: {result['confidence']:.4f} (threshold {result['threshold']:.4f})")
    print(f"Move Count: {result['move_count']}")
    print(f"\n{result['explanation']}")
    print_statistics(result['statistics'])
    print("=" * 60)


def analyse_typed_game(predictor: TensorFlowChessPredictor):
    for color in ('white', 'black'):
        print(f"\nEnter {color}'s moves (comma-separated, SAN; blank to skip):")
        moves = input().strip()
        if not moves:
            continue
        print(f"Enter {color}'s think times (comma-separated seconds, optional):")
        times = input().strip()
        print(f"Enter {color}'s per-move win-chance losses (comma-separated, optional):")
        losses = input().strip()
        result = predictor.predict_player_moves(moves, times, color, losses)
        print_player_analysis(result, f"{color} player")


def main():
    print("🎯 TensorFlow Chess Anti-Cheat Prediction")
    print("=" * 50)
    try:
        predictor = TensorFlowChessPredictor()
    except Exception as e:
        print(f"❌ Error loading TensorFlow model: {e}")
        return

    while True:
        print("\n1. Analyse a Lichess account (recommended)")
        print("2. Analyse one game from typed-in moves")
        print("3. Quit")
        choice = input("Choice: ").strip().lower()
        try:
            if choice == '1':
                username = input("Lichess username: ").strip()
                games = int(input("Games to analyse (default 50): ").strip() or "50")
                print(f"📥 Downloading and scoring {username}'s games...")
                print_account_analysis(predictor.predict_account(username, games))
            elif choice == '2':
                analyse_typed_game(predictor)
            elif choice in ('3', 'q', 'quit'):
                break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
