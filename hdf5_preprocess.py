#!/usr/bin/env python3
"""
HDF5 Preprocessing for Chess Anti-Cheat Detection
Handles player-level data from HDF5 format
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ChessHDF5Dataset(Dataset):
    """Dataset class for HDF5 chess data with player-level granularity"""
    
    def __init__(self, data_df, vocab, max_len=300, include_times=True):
        self.data_df = data_df
        self.vocab = vocab
        self.max_len = max_len
        self.include_times = include_times
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Tokenize moves
        moves = row['moves'].split(',') if isinstance(row['moves'], str) else []
        tokens = [self.vocab.get(move.strip(), self.vocab['<unk>']) for move in moves]
        
        # Pad or truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_len - len(tokens))
        
        # Create side information (all moves are from the same player)
        sides = [0] * self.max_len  # All moves are from the same player
        
        # Move times (if available)
        if self.include_times and 'move_times' in row:
            times_str = row['move_times']
            if isinstance(times_str, str):
                times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
            else:
                times = times_str if isinstance(times_str, list) else []
            
            if len(times) > self.max_len:
                times = times[:self.max_len]
            else:
                times = times + [0.0] * (self.max_len - len(times))
        else:
            times = [0.0] * self.max_len
        
        # Convert to tensors
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        sides_tensor = torch.tensor(sides, dtype=torch.long)
        times_tensor = torch.tensor(times, dtype=torch.float)
        label_tensor = torch.tensor(row['label'], dtype=torch.float)
        
        return {
            'tokens': tokens_tensor,
            'sides': sides_tensor,
            'times': times_tensor,
            'label': label_tensor,
            'player_color': row['player_color']
        }

def load_hdf5_data(file_path: str) -> pd.DataFrame:
    """Load data from HDF5 file"""
    try:
        df = pd.read_hdf(file_path, key='chess_data')
        print(f"✅ Loaded {len(df)} player records from {file_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()

def build_vocab_from_hdf5(df: pd.DataFrame, min_freq: int = 3) -> Dict[str, int]:
    """Build vocabulary from HDF5 data"""
    print("Building vocabulary from HDF5 data...")
    
    move_counts = Counter()
    
    for moves_str in df['moves']:
        if isinstance(moves_str, str):
            moves = moves_str.split(',')
            for move in moves:
                move = move.strip()
                if move:
                    move_counts[move] += 1
    
    # Filter by minimum frequency
    vocab = {'<PAD>': 0, '<unk>': 1}
    for move, count in move_counts.items():
        if count >= min_freq:
            vocab[move] = len(vocab)
    
    print(f"✅ Vocabulary size: {len(vocab)}")
    return vocab

def prepare_hdf5_dataset(df: pd.DataFrame, vocab: Dict[str, int], 
                        test_size: float = 0.2, val_size: float = 0.1,
                        max_len: int = 300, batch_size: int = 32,
                        random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare HDF5 dataset for training"""
    
    print("Preparing HDF5 dataset...")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, 
                                        random_state=random_state, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), 
                                       random_state=random_state, stratify=train_df['label'])
    
    # Create datasets
    train_dataset = ChessHDF5Dataset(train_df, vocab, max_len)
    val_dataset = ChessHDF5Dataset(val_df, vocab, max_len)
    test_dataset = ChessHDF5Dataset(test_df, vocab, max_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Dataset prepared:")
    print(f"   - Training: {len(train_dataset)} samples")
    print(f"   - Validation: {len(val_dataset)} samples")
    print(f"   - Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def analyze_player_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze player patterns in HDF5 data"""
    print("Analyzing player patterns...")
    
    analysis = {}
    
    # Player color distribution
    color_counts = df['player_color'].value_counts()
    analysis['color_distribution'] = color_counts.to_dict()
    
    # Label distribution
    label_counts = df['label'].value_counts()
    analysis['label_distribution'] = label_counts.to_dict()
    
    # Game results
    result_counts = df['result'].value_counts()
    analysis['result_distribution'] = result_counts.to_dict()
    
    # Move statistics
    move_lengths = []
    for moves_str in df['moves']:
        if isinstance(moves_str, str):
            moves = moves_str.split(',')
            move_lengths.append(len(moves))
    
    if move_lengths:
        analysis['move_statistics'] = {
            'avg_moves_per_player': np.mean(move_lengths),
            'median_moves_per_player': np.median(move_lengths),
            'min_moves_per_player': np.min(move_lengths),
            'max_moves_per_player': np.max(move_lengths)
        }
    
    # Time statistics
    all_times = []
    for times_str in df['move_times']:
        if isinstance(times_str, str):
            times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
            all_times.extend(times)
        elif isinstance(times_str, list):
            all_times.extend(times_str)
    
    if all_times:
        analysis['time_statistics'] = {
            'avg_move_time': np.mean(all_times),
            'median_move_time': np.median(all_times),
            'min_move_time': np.min(all_times),
            'max_move_time': np.max(all_times)
        }
    
    print("✅ Analysis complete")
    return analysis

def save_vocab(vocab: Dict[str, int], file_path: str):
    """Save vocabulary to file"""
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✅ Vocabulary saved to {file_path}")

def load_vocab(file_path: str) -> Dict[str, int]:
    """Load vocabulary from file"""
    with open(file_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"✅ Vocabulary loaded from {file_path}")
    return vocab

# def get_player_features(df: pd.DataFrame, player_id: str) -> Dict[str, Any]:
#     """Get features for a specific player - DEPRECATED: player_id no longer exists"""
#     # This function is deprecated since we removed player_id from the dataset
#     # Use player_color and other available features instead
#     return {}

def main():
    """Test HDF5 preprocessing"""
    # Load data
    hdf5_file = "data/combined_games.h5"
    
    if not os.path.exists(hdf5_file):
        print(f"❌ {hdf5_file} not found. Please run fetch_games.py first.")
        return
    
    df = load_hdf5_data(hdf5_file)
    
    if len(df) == 0:
        return
    
    # Build vocabulary
    vocab = build_vocab_from_hdf5(df)
    
    # Save vocabulary
    save_vocab(vocab, "hdf5_vocab.pkl")
    
    # Analyze patterns
    analysis = analyze_player_patterns(df)
    
    print("\n📊 Analysis Results:")
    for key, value in analysis.items():
        print(f"   {key}: {value}")
    
    # Prepare dataset
    train_loader, val_loader, test_loader = prepare_hdf5_dataset(df, vocab)
    
    print("\n✅ HDF5 preprocessing complete!")

if __name__ == "__main__":
    main() 