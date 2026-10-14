#!/usr/bin/env python3
"""
TensorFlow Prediction Script for Chess Anti-Cheat Detection
Analyzes both players in a chess game for cheating detection
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Fallback: import the model builder to reconstruct architecture if needed
from improved_tensorflow_model import AdvancedChessClassifier

class TensorFlowChessPredictor:
    """TensorFlow-based predictor for chess anti-cheat detection"""
    
    def __init__(self, model_path: str | None = None, vocab_path="hdf5_vocab.pkl"):
        # Decide model path
        if model_path is None:
            if os.path.exists("advanced_chess_model.keras"):
                model_path = "advanced_chess_model.keras"
            elif os.path.exists("advanced_chess_model.h5"):
                model_path = "advanced_chess_model.h5"
            else:
                raise FileNotFoundError("No TensorFlow model found (.keras or .h5)")
        
        # Load vocabulary
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        
        # Load TensorFlow model
        self.model = None
        load_error = None
        try:
            if model_path.endswith('.h5'):
                import keras
                self.model = keras.saving.load_model(model_path, compile=False, safe_mode=False)
            else:
                self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            load_error = e
        
        # Fallback: rebuild architecture and load weights from legacy .h5
        if self.model is None and model_path.endswith('.h5'):
            try:
                classifier = AdvancedChessClassifier(vocab_size=len(self.vocab))
                classifier.build_model()
                classifier.compile_model()
                classifier.model.load_weights(model_path, skip_mismatch=True)
                self.model = classifier.model
                load_error = None
            except Exception as e2:
                load_error = e2
        
        if self.model is None and load_error is not None:
            raise RuntimeError(f"Failed to load model '{model_path}': {load_error}")
        
        print(f"✅ TensorFlow model loaded successfully from {model_path}")
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.low_confidence_threshold = 0.3
        
    def predict_player_moves(self, moves_str: str, move_times_str: str = "", player_color: str = "white", max_len: int = 300) -> Dict[str, Any]:
        """Predict if a player's moves indicate cheating using TensorFlow model"""
        
        if not moves_str or moves_str.strip() == '':
            return {
                'prediction': 'Invalid',
                'confidence': 0.0,
                'explanation': 'No moves provided',
                'risk_factors': [],
                'player_color': player_color
            }
        
        # Tokenize moves
        moves = moves_str.split(',') if isinstance(moves_str, str) else []
        tokens = [self.vocab.get(move.strip(), self.vocab['<unk>']) for move in moves]
        
        # Pad or truncate
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (max_len - len(tokens))
        
        # Process move times
        if move_times_str:
            if isinstance(move_times_str, str):
                times = [float(t.strip()) for t in move_times_str.split(',') if t.strip()]
            else:
                times = move_times_str if isinstance(move_times_str, list) else []
            
            if len(times) > max_len:
                times = times[:max_len]
            else:
                times = times + [0.0] * (max_len - len(times))
        else:
            times = [0.0] * max_len
        
        # Normalize times
        times = np.array(times)
        if np.std(times) > 0:
            times = (times - np.mean(times)) / np.std(times)
        
        # Color encoding (white=1, black=0)
        color = 1.0 if player_color == 'white' else 0.0
        
        # Prepare input for TensorFlow model
        X_moves = np.array([tokens])
        X_times = np.array([times])
        X_color = np.array([[color]])
        
        # Get prediction
        prediction = self.model.predict([X_moves, X_times, X_color], verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine prediction
        prediction_label = "Cheater" if confidence > 0.5 else "Honest"
        
        # Analyze risk factors including timing
        risk_factors = self._analyze_risk_factors_with_timing(moves, times, confidence, player_color)
        
        # Generate explanation
        explanation = self._generate_explanation_with_context(moves, times, confidence, risk_factors, player_color)
        
        # Calculate confidence level
        confidence_level = self._get_confidence_level(confidence)
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'explanation': explanation,
            'risk_factors': risk_factors,
            'move_count': len(moves),
            'player_color': player_color
        }
    
    def _analyze_risk_factors_with_timing(self, moves: List[str], times: np.ndarray, confidence: float, player_color: str) -> List[Dict]:
        """Analyze risk factors based purely on pattern recognition"""
        risk_factors = []
        
        if not moves:
            return risk_factors
        
        # Pure pattern recognition - no derived statistics
        if len(moves) > 10:
            # Look for repetitive move patterns (3-move sequences)
            move_chunks = [moves[i:i+3] for i in range(len(moves)-2)]
            unique_chunks = len(set(tuple(chunk) for chunk in move_chunks))
            pattern_variety = unique_chunks / len(move_chunks)
            
            if pattern_variety < 0.3:
                risk_factors.append({
                    'factor': 'Repetitive move patterns',
                    'value': pattern_variety,
                    'risk_level': 'medium',
                    'description': f'Low move pattern variety ({pattern_variety:.2f}), might indicate engine usage'
                })
        
        # Check for move length consistency patterns
        if len(moves) > 5:
            move_lengths = [len(m) for m in moves]
            length_std = np.std(move_lengths)
            if length_std < 0.5:
                risk_factors.append({
                    'factor': 'Consistent move lengths',
                    'value': length_std,
                    'risk_level': 'medium',
                    'description': f'Very consistent move lengths (std: {length_std:.2f}), typical of engines'
                })
        
        # Move variety pattern
        unique_moves = len(set(moves))
        variety_ratio = unique_moves / len(moves) if len(moves) else 0.0
        if variety_ratio < 0.3:
            risk_factors.append({
                'factor': 'Low move variety',
                'value': variety_ratio,
                'risk_level': 'medium',
                'description': 'Limited move variety, might indicate engine usage'
            })
        
        return risk_factors
    
    def _generate_explanation_with_context(self, moves: List[str], times: np.ndarray, confidence: float, 
                                         risk_factors: List[Dict], player_color: str) -> str:
        """Generate human-readable explanation with context"""
        if confidence > 0.8:
            base_explanation = "The TensorFlow model is highly confident in this prediction."
        elif confidence > 0.6:
            base_explanation = "The TensorFlow model shows moderate confidence in this prediction."
        else:
            base_explanation = "The TensorFlow model shows low confidence in this prediction."
        
        if moves:
            base_explanation += f" Analyzed {len(moves)} moves."
            # Pattern analysis summary
            if len(moves) > 10:
                move_chunks = [moves[i:i+3] for i in range(len(moves)-2)]
                unique_chunks = len(set(tuple(chunk) for chunk in move_chunks))
                pattern_variety = unique_chunks / len(move_chunks)
                base_explanation += f" Move pattern variety: {pattern_variety:.2f}."
        
        base_explanation += f" Player color: {player_color}."
        if risk_factors:
            high_risk = [rf for rf in risk_factors if rf['risk_level'] == 'high']
            medium_risk = [rf for rf in risk_factors if rf['risk_level'] == 'medium']
            if high_risk:
                base_explanation += f" {len(high_risk)} high-risk factors detected."
            if medium_risk:
                base_explanation += f" {len(medium_risk)} medium-risk factors detected."
        return base_explanation
    
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence > self.high_confidence_threshold:
            return "Very High"
        elif confidence > 0.6:
            return "High"
        elif confidence > 0.4:
            return "Medium"
        elif confidence > self.low_confidence_threshold:
            return "Low"
        else:
            return "Very Low"

def print_player_analysis(result: Dict[str, Any], player_name: str):
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    print("=" * 60)
    print(f"{player_name.upper()} ANALYSIS")
    print("=" * 60)
    print(f"Player Color: {result['player_color']}")
    print(f"PREDICTION: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence_level']})")
    print(f"Move Count: {result['move_count']}")
    print(f"\nExplanation: {result['explanation']}")
    if result['risk_factors']:
        print(f"\nRisk Factors:")
        for i, factor in enumerate(result['risk_factors'], 1):
            print(f"  {i}. {factor['factor']} (Risk: {factor['risk_level'].upper()})")
            print(f"     Value: {factor['value']:.3f}")
            print(f"     Description: {factor['description']}")
    else:
        print(f"\nRisk Factors: None detected")
    print("=" * 60)

def main():
    print("🎯 TensorFlow Chess Anti-Cheat Prediction")
    print("=" * 50)
    
    # Determine model path preference
    model_path = None
    if os.path.exists("advanced_chess_model.keras"):
        model_path = "advanced_chess_model.keras"
    elif os.path.exists("advanced_chess_model.h5"):
        model_path = "advanced_chess_model.h5"
    else:
        print("❌ No model file found. Train the model first: python train_tensorflow.py")
        return
    
    vocab_path = "hdf5_vocab.pkl"
    if not os.path.exists(vocab_path):
        print(f"❌ {vocab_path} not found. Please run training first.")
        return
    
    try:
        predictor = TensorFlowChessPredictor(model_path, vocab_path)
    except Exception as e:
        print(f"❌ Error loading TensorFlow model: {e}")
        return
    
    print("\n" + "=" * 60)
    print("INTERACTIVE GAME ANALYSIS (TensorFlow)")
    print("=" * 60)
    print("Enter game data in the following format:")
    print("1. White moves (comma-separated)")
    print("2. Black moves (comma-separated)")
    print("3. White move times (comma-separated seconds)")
    print("4. Black move times (comma-separated seconds)")
    print("5. White result (win/loss/draw)")
    print("6. Black result (win/loss/draw)")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            print("\n" + "-" * 40)
            print("Enter game data:")
            print("-" * 40)
            print("Enter white moves (comma-separated):")
            white_moves = input().strip()
            if white_moves.lower() == 'quit':
                break
            print("Enter black moves (comma-separated):")
            black_moves = input().strip()
            if black_moves.lower() == 'quit':
                break
            print("Enter white move times (comma-separated seconds):")
            white_times = input().strip()
            if white_times.lower() == 'quit':
                break
            print("Enter black move times (comma-separated seconds):")
            black_times = input().strip()
            if black_times.lower() == 'quit':
                break
            print("Enter white result (win/loss/draw):")
            white_result = input().strip()
            if white_result.lower() == 'quit':
                break
            print("Enter black result (win/loss/draw):")
            black_result = input().strip()
            if black_result.lower() == 'quit':
                break
            if not white_moves or not black_moves:
                print("❌ Both white and black moves are required!")
                continue
            print("\n" + "🔍" * 20)
            print("ANALYZING WHITE PLAYER (TensorFlow)")
            print("🔍" * 20)
            white_result_analysis = predictor.predict_player_moves(white_moves, white_times, "white")
            print_player_analysis(white_result_analysis, "WHITE PLAYER")
            print("\n" + "🔍" * 20)
            print("ANALYZING BLACK PLAYER (TensorFlow)")
            print("🔍" * 20)
            black_result_analysis = predictor.predict_player_moves(black_moves, black_times, "black")
            print_player_analysis(black_result_analysis, "BLACK PLAYER")
            print("\n" + "📊" * 20)
            print("GAME SUMMARY (TensorFlow)")
            print("📊" * 20)
            print(f"White Player: {white_result_analysis['prediction']} (Confidence: {white_result_analysis['confidence']:.3f})")
            print(f"Black Player: {black_result_analysis['prediction']} (Confidence: {black_result_analysis['confidence']:.3f})")
            print(f"White Result: {white_result}")
            print(f"Black Result: {black_result}")
            white_cheating = white_result_analysis['prediction'] == 'Cheater'
            black_cheating = black_result_analysis['prediction'] == 'Cheater'
            if white_cheating and black_cheating:
                game_assessment = "BOTH PLAYERS SUSPICIOUS"
            elif white_cheating:
                game_assessment = "WHITE PLAYER SUSPICIOUS"
            elif black_cheating:
                game_assessment = "BLACK PLAYER SUSPICIOUS"
            else:
                game_assessment = "BOTH PLAYERS APPEAR HONEST"
            print(f"\n🎯 GAME ASSESSMENT: {game_assessment}")
            print("📊" * 20)
        except Exception as e:
            print(f"Error analyzing game: {e}")
            print("Please check your data format and try again.")

if __name__ == "__main__":
    main()
