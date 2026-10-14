#!/usr/bin/env python3
"""
Convert Chess Data to HDF5 Format
Converts current CSV format to HDF5 with player-level granularity
"""

import pandas as pd
import numpy as np
import h5py
import chess
import chess.pgn
from typing import List, Dict, Tuple
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ChessDataConverter:
    """Convert chess data to HDF5 format with player-level granularity"""
    
    def __init__(self):
        self.game_counter = 0
        
    def parse_moves_to_players(self, moves_str: str) -> Tuple[List[str], List[str]]:
        """Parse moves string into white and black moves"""
        if not moves_str or moves_str.strip() == '':
            return [], []
            
        moves = moves_str.strip().split()
        white_moves = []
        black_moves = []
        
        for i, move in enumerate(moves):
            if i % 2 == 0:  # White's move
                white_moves.append(move)
            else:  # Black's move
                black_moves.append(move)
                
        return white_moves, black_moves
    
    def determine_game_result(self, moves: List[str]) -> str:
        """Determine game result based on final move"""
        if not moves:
            return "draw"
            
        last_move = moves[-1]
        
        if last_move.endswith('#'):
            return "win"  # Checkmate
        elif last_move.endswith('+'):
            return "win"  # Check (likely leading to mate)
        elif last_move in ['1-0', '0-1', '1/2-1/2']:
            return "win" if last_move == '1-0' else "loss" if last_move == '0-1' else "draw"
        else:
            return "draw"  # Default to draw for incomplete games
    
    def create_player_records(self, game_id: str, moves_str: str, label: int) -> List[Dict]:
        """Create player-level records from a game"""
        white_moves, black_moves = self.parse_moves_to_players(moves_str)
        
        records = []
        
        # White player record
        if white_moves:
            white_result = self.determine_game_result(white_moves + black_moves)
            
            records.append({
                'game_id': game_id,
                'player_color': 'white',
                'player_id': f'player_{game_id}_white',
                'opponent_id': f'player_{game_id}_black',
                'moves': ','.join(white_moves),
                'move_times': [],  # CSV input has no clock data; use fetch_games.py for real times
                'result': white_result,
                'label': label
            })
        
        # Black player record
        if black_moves:
            black_result = self.determine_game_result(white_moves + black_moves)
            # Invert result for black
            if black_result == 'win':
                black_result = 'loss'
            elif black_result == 'loss':
                black_result = 'win'
            
            records.append({
                'game_id': game_id,
                'player_color': 'black',
                'player_id': f'player_{game_id}_black',
                'opponent_id': f'player_{game_id}_white',
                'moves': ','.join(black_moves),
                'move_times': [],  # CSV input has no clock data; use fetch_games.py for real times
                'result': black_result,
                'label': label
            })
        
        return records
    
    def convert_csv_to_hdf5(self, csv_file: str, output_file: str, label: int):
        """Convert CSV file to HDF5 format"""
        print(f"Converting {csv_file} to {output_file}...")
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        all_records = []
        
        for idx, row in df.iterrows():
            game_id = f"{label:03d}_{idx:06d}"  # Format: 001_000001
            moves_str = row['moves']
            
            records = self.create_player_records(game_id, moves_str, label)
            all_records.extend(records)
            
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} games...")
        
        # Create DataFrame
        result_df = pd.DataFrame(all_records)
        
        # Convert move_times lists to strings for HDF5 compatibility
        result_df['move_times'] = result_df['move_times'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))
        
        # Save to HDF5
        result_df.to_hdf(output_file, key='chess_data', mode='w', format='table')
        
        print(f"✅ Converted {len(df)} games to {len(result_df)} player records")
        print(f"📊 Statistics:")
        print(f"   - Total player records: {len(result_df)}")
        print(f"   - White players: {len(result_df[result_df['player_color'] == 'white'])}")
        print(f"   - Black players: {len(result_df[result_df['player_color'] == 'black'])}")
        print(f"   - Cheaters (label=1): {len(result_df[result_df['label'] == 1])}")
        print(f"   - Honest players (label=0): {len(result_df[result_df['label'] == 0])}")
        
        return result_df
    
    def combine_datasets(self, cheater_file: str, honest_file: str, output_file: str):
        """Combine cheater and honest datasets"""
        print("Combining datasets...")
        
        # Read both datasets
        cheater_df = pd.read_hdf(cheater_file, key='chess_data')
        honest_df = pd.read_hdf(honest_file, key='chess_data')
        
        # Combine
        combined_df = pd.concat([cheater_df, honest_df], ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_df.to_hdf(output_file, key='chess_data', mode='w', format='table')
        
        print(f"✅ Combined dataset saved to {output_file}")
        print(f"📊 Combined Statistics:")
        print(f"   - Total player records: {len(combined_df)}")
        print(f"   - Cheaters: {len(combined_df[combined_df['label'] == 1])}")
        print(f"   - Honest players: {len(combined_df[combined_df['label'] == 0])}")
        
        return combined_df
    
    def analyze_hdf5_data(self, hdf5_file: str):
        """Analyze HDF5 dataset"""
        print(f"Analyzing {hdf5_file}...")
        
        df = pd.read_hdf(hdf5_file, key='chess_data')
        
        print("📊 Dataset Analysis:")
        print(f"   - Total records: {len(df)}")
        print(f"   - Unique games: {df['game_id'].nunique()}")
        print(f"   - Unique players: {df['player_id'].nunique()}")
        print(f"   - White players: {len(df[df['player_color'] == 'white'])}")
        print(f"   - Black players: {len(df[df['player_color'] == 'black'])}")
        print(f"   - Cheaters: {len(df[df['label'] == 1])}")
        print(f"   - Honest players: {len(df[df['label'] == 0])}")
        
        print("\n🏆 Game Results:")
        result_counts = df['result'].value_counts()
        for result, count in result_counts.items():
            print(f"   - {result}: {count}")
        
        print("\n⏱️ Move Times Statistics:")
        all_times = []
        for times_str in df['move_times']:
            if isinstance(times_str, str):
                times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
                all_times.extend(times)
            elif isinstance(times_str, list):
                all_times.extend(times_str)
        
        if all_times:
            all_times = [float(t) for t in all_times if isinstance(t, (int, float)) or str(t).replace('.', '').isdigit()]
            if all_times:
                print(f"   - Average move time: {np.mean(all_times):.2f}s")
                print(f"   - Median move time: {np.median(all_times):.2f}s")
                print(f"   - Min move time: {np.min(all_times):.2f}s")
                print(f"   - Max move time: {np.max(all_times):.2f}s")
            else:
                print("   - No valid time data found")
        else:
            print("   - No time data available")
        
        return df

def main():
    """Main conversion process"""
    converter = ChessDataConverter()
    
    # File paths
    cheater_csv = "data/cheater_games.csv"
    honest_csv = "data/honest_games.csv"
    cheater_hdf5 = "data/cheater_games.h5"
    honest_hdf5 = "data/honest_games.h5"
    combined_hdf5 = "data/combined_games.h5"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    print("🔄 Converting Chess Data to HDF5 Format")
    print("=" * 50)
    
    # Convert cheater games
    if os.path.exists(cheater_csv):
        converter.convert_csv_to_hdf5(cheater_csv, cheater_hdf5, label=1)
    else:
        print(f"❌ {cheater_csv} not found")
    
    # Convert honest games
    if os.path.exists(honest_csv):
        converter.convert_csv_to_hdf5(honest_csv, honest_hdf5, label=0)
    else:
        print(f"❌ {honest_csv} not found")
    
    # Combine datasets
    if os.path.exists(cheater_hdf5) and os.path.exists(honest_hdf5):
        converter.combine_datasets(cheater_hdf5, honest_hdf5, combined_hdf5)
        
        # Analyze combined dataset
        print("\n" + "=" * 50)
        converter.analyze_hdf5_data(combined_hdf5)
    
    print("\n✅ Conversion complete!")
    print(f"📁 Output files:")
    print(f"   - {cheater_hdf5}")
    print(f"   - {honest_hdf5}")
    print(f"   - {combined_hdf5}")

if __name__ == "__main__":
    main() 