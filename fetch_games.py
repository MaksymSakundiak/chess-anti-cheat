#!/usr/bin/env python3
"""
Enhanced Lichess Game Fetcher for HDF5 Format
Fetch rated rapid chess games from Lichess API and append to existing HDF5 dataset
"""

import os
import time
import json
import re
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PGN_TOKEN_RE = re.compile(r'\{[^}]*\}|\S+')
MOVE_NUMBER_RE = re.compile(r'\d+\.(\.\.)?')
CLOCK_RE = re.compile(r'\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]')
PGN_RESULTS = {'1-0', '0-1', '1/2-1/2', '*'}

class LichessHDF5Fetcher:
    """Enhanced Lichess data fetcher for HDF5 format with player-level granularity"""
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize the fetcher"""
        load_dotenv()
        
        # Use provided token or load from .env
        self.api_token = api_token or os.getenv('LICHESS_API_TOKEN')
        
        if not self.api_token:
            print("Warning: No Lichess API token found. Using public API (limited rate).")
        
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({'Authorization': f'Bearer {self.api_token}'})
            print("Using authenticated Lichess API")
        else:
            print("Using public Lichess API")
        
        # Track fetched players to avoid duplicates
        self.fetched_players_file = 'fetched_players.json'
        self.fetched_players = self._load_fetched_players()
        
        # Player lists files
        self.cheater_players_file = 'cheater_players.json'
        self.honest_players_file = 'honest_players.json'
        
        # HDF5 file paths
        self.cheater_hdf5 = "data/cheater_games.h5"
        self.honest_hdf5 = "data/honest_games.h5"
        self.combined_hdf5 = "data/combined_games.h5"
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
    
    def _load_fetched_players(self) -> Dict[str, Dict]:
        """Load tracking of previously fetched players"""
        try:
            if os.path.exists(self.fetched_players_file):
                with open(self.fetched_players_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Warning: Could not load fetched players tracking: {e}")
            return {}
    
    def _save_fetched_players(self):
        """Save tracking of fetched players"""
        try:
            with open(self.fetched_players_file, 'w') as f:
                json.dump(self.fetched_players, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save fetched players tracking: {e}")
    
    def _load_player_list(self, file_path: str) -> List[str]:
        """Load list of players from file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Warning: Could not load player list from {file_path}: {e}")
            return []
    
    def _save_player_list(self, players: List[str], file_path: str):
        """Save list of players to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(players, f, indent=2)
            print(f"✅ Saved {len(players)} players to {file_path}")
        except Exception as e:
            print(f"❌ Error saving player list to {file_path}: {e}")
    
    def add_cheater_players(self, usernames: List[str]):
        """Add players to the cheater list"""
        current_cheaters = self._load_player_list(self.cheater_players_file)
        
        # Add new players (avoid duplicates)
        for username in usernames:
            if username not in current_cheaters:
                current_cheaters.append(username)
        
        self._save_player_list(current_cheaters, self.cheater_players_file)
        print(f"✅ Added {len(usernames)} players to cheater list. Total: {len(current_cheaters)}")
    
    def add_honest_players(self, usernames: List[str]):
        """Add players to the honest list"""
        current_honest = self._load_player_list(self.honest_players_file)
        
        # Add new players (avoid duplicates)
        for username in usernames:
            if username not in current_honest:
                current_honest.append(username)
        
        self._save_player_list(current_honest, self.honest_players_file)
        print(f"✅ Added {len(usernames)} players to honest list. Total: {len(current_honest)}")
    
    def remove_cheater_players(self, usernames: List[str]):
        """Remove players from the cheater list"""
        current_cheaters = self._load_player_list(self.cheater_players_file)
        
        # Remove specified players
        for username in usernames:
            if username in current_cheaters:
                current_cheaters.remove(username)
        
        self._save_player_list(current_cheaters, self.cheater_players_file)
        print(f"✅ Removed {len(usernames)} players from cheater list. Total: {len(current_cheaters)}")
    
    def remove_honest_players(self, usernames: List[str]):
        """Remove players from the honest list"""
        current_honest = self._load_player_list(self.honest_players_file)
        
        # Remove specified players
        for username in usernames:
            if username in current_honest:
                current_honest.remove(username)
        
        self._save_player_list(current_honest, self.honest_players_file)
        print(f"✅ Removed {len(usernames)} players from honest list. Total: {len(current_honest)}")
    
    def get_cheater_players(self) -> List[str]:
        """Get list of cheater players"""
        return self._load_player_list(self.cheater_players_file)
    
    def get_honest_players(self) -> List[str]:
        """Get list of honest players"""
        return self._load_player_list(self.honest_players_file)
    
    def show_player_lists(self):
        """Show current player lists"""
        cheaters = self.get_cheater_players()
        honest = self.get_honest_players()
        
        print("\n📊 Current Player Lists:")
        print(f"  Cheater Players ({len(cheaters)}):")
        for i, player in enumerate(cheaters, 1):
            print(f"    {i}. {player}")
        
        print(f"\n  Honest Players ({len(honest)}):")
        for i, player in enumerate(honest, 1):
            print(f"    {i}. {player}")
    
    def _get_player_creation_date(self, username: str) -> datetime:
        """Get the date when a player's account was created"""
        try:
            # Fetch player profile from Lichess API
            url = f"https://lichess.org/api/user/{username}"
            response = self.session.get(url)
            response.raise_for_status()
            
            player_data = response.json()
            
            # Show player info
            title = player_data.get('title', '')
            title_str = f" {title}" if title else ""
            print(f"👤 {username}{title_str}")
            
            # Extract creation date
            created_at = player_data.get('createdAt', None)
            if created_at:
                # Convert from milliseconds to datetime
                creation_date = datetime.fromtimestamp(created_at / 1000)
                print(f"📅 Account created: {creation_date.strftime('%Y-%m-%d')}")
                return creation_date
            else:
                print(f"⚠️ Could not get creation date for {username}, using 2020 as fallback")
                return datetime(2020, 1, 1)
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching profile for {username}: {e}")
            print(f"⚠️ Using 2020 as fallback creation date")
            return datetime(2020, 1, 1)
        except Exception as e:
            print(f"⚠️ Unexpected error getting creation date for {username}: {e}")
            print(f"⚠️ Using 2020 as fallback creation date")
            return datetime(2020, 1, 1)
    
    def _update_player_tracking(self, username: str, games_fetched: int, last_fetch: str):
        """Update tracking for a player"""
        self.fetched_players[username] = {
            'games_fetched': games_fetched,
            'last_fetch': last_fetch,
            'total_games_fetched': self.fetched_players.get(username, {}).get('total_games_fetched', 0) + games_fetched
        }
        self._save_fetched_players()
    
    def fetch_player_games(self, 
                          username: str, 
                          max_games: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          force_refetch: bool = False) -> List[Dict]:
        """Fetch rated rapid games for a specific player"""
        
        print(f"Fetching rated rapid games for player: {username}")
        
        # Check if player was recently fetched
        if not force_refetch and username in self.fetched_players:
            last_fetch = self.fetched_players[username].get('last_fetch', '')
            total_fetched = self.fetched_players[username].get('total_games_fetched', 0)
            print(f"Player {username} was last fetched on {last_fetch} (total: {total_fetched} games)")
            
            # Automatically refetch without asking
            print(f"🔄 Automatically refetching games for {username}")
        
        try:
            # Get player's account creation date
            account_created = self._get_player_creation_date(username)
            
            # Calculate date range
            if since is None:
                since = account_created  # Since account creation
            
            if until is None:
                until = datetime.now()
            
            # Convert to timestamp (milliseconds)
            since_ts = int(since.timestamp() * 1000)
            until_ts = int(until.timestamp() * 1000)
            
            # Fetch games from Lichess API
            url = f"https://lichess.org/api/games/user/{username}"
            params = {
                'perfType': 'rapid',
                'rated': 'true',
                'max': max_games,
                'since': since_ts,
                'until': until_ts,
                'clocks': 'true',
                'evals': 'true'
            }
            
            print(f"📊 Fetching games from account creation ({since.strftime('%Y-%m-%d')}) to {until.strftime('%Y-%m-%d')}")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Parse games (Lichess API returns PGN format, not JSON)
            games_data = response.text.strip()
            games = []
            
            # Parse multiple games from the PGN response
            # The Lichess API returns complete PGN games separated by double newlines
            # Each game has headers followed by moves
            pgn_games = games_data.split('\n\n\n')  # Split by triple newlines
            
            # Filter out empty games and games without moves
            valid_games = []
            for pgn_game in pgn_games:
                if pgn_game.strip() and '[Event' in pgn_game and '1.' in pgn_game:
                    valid_games.append(pgn_game)
            
            # Each valid_game is already a complete PGN game
            combined_games = valid_games
            
            for combined_game in combined_games:
                try:
                    game = self._parse_pgn_game(combined_game, username)
                    if game:
                        games.append(game)
                except Exception as e:
                    print(f"Error parsing PGN game: {e}")
                    continue
            
            print(f"✅ Fetched {len(games)} games for {username}")
            
            # Update tracking
            self._update_player_tracking(username, len(games), datetime.now().isoformat())
            
            return games
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"⚠️ Rate limited for {username}, waiting 5 seconds...")
                time.sleep(5)
                return []
            else:
                print(f"❌ HTTP Error fetching games for {username}: {e}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching games for {username}: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error for {username}: {e}")
            return []
    
    def _determine_player_result(self, game_result: str, player_color: str) -> str:
        """Determine game result for the specific player from a PGN Result header"""
        winner = {'1-0': 'white', '0-1': 'black'}.get(game_result)
        if winner is None:
            return 'draw'
        return 'win' if winner == player_color else 'loss'
    
    def _parse_time_control(self, time_control: str) -> Optional[Tuple[int, int]]:
        """Parse a PGN TimeControl header like '600+5' into (base_seconds, increment_seconds)"""
        try:
            base, increment = time_control.split('+')
            return int(base), int(increment)
        except ValueError:
            return None  # Correspondence ('-') or malformed
    
    def _parse_pgn_game(self, pgn_text: str, username: str) -> Optional[Dict]:
        """Parse a PGN game, extracting this player's moves and real per-move think times"""
        try:
            game_info = {}
            movetext_lines = []
            
            # Split PGN into [Key "Value"] headers and movetext
            for line in pgn_text.strip().split('\n'):
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    key_value = line[1:-1].split(' ', 1)
                    if len(key_value) == 2:
                        game_info[key_value[0]] = key_value[1].strip('"')
                elif line:
                    movetext_lines.append(line)
            
            # Walk the movetext: moves, move numbers ("12." / "12..."), results,
            # NAGs ($1) and comments like { [%eval 0.18] [%clk 0:10:00] }
            moves = []
            clocks = []  # Remaining clock (seconds) after each move
            for token in PGN_TOKEN_RE.findall(' '.join(movetext_lines)):
                if token.startswith('{'):
                    clock = CLOCK_RE.search(token)
                    if clock and moves:
                        hours, minutes, seconds = clock.groups()
                        clocks[-1] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                elif MOVE_NUMBER_RE.fullmatch(token) or token in PGN_RESULTS or token.startswith('$'):
                    continue
                else:
                    # Drop Lichess's engine annotations (?!, ??, !) so tokens are plain SAN
                    moves.append(token.rstrip('?!'))
                    clocks.append(None)
            
            # Determine if the fetched player is white or black
            white_player = game_info.get('White', '').lower()
            black_player = game_info.get('Black', '').lower()
            if white_player == username.lower():
                player_color = 'white'
            elif black_player == username.lower():
                player_color = 'black'
            else:
                return None  # Player not in this game
            
            # Real move times need clock data on every move and a known increment
            time_control = self._parse_time_control(game_info.get('TimeControl', ''))
            if time_control is None or not moves or None in clocks:
                return None
            base_time, increment = time_control
            
            # Extract this player's moves and how long each took:
            # time spent = clock before the move - clock after the move + increment.
            # Lichess doesn't run the clock for each side's first move, so it counts as 0.
            first_index = 0 if player_color == 'white' else 1
            player_moves = []
            move_times = []
            for i in range(first_index, len(moves), 2):
                player_moves.append(moves[i])
                if i < 2:
                    move_times.append(0.0)
                else:
                    spent = clocks[i - 2] - clocks[i] + increment
                    move_times.append(round(max(spent, 0.0), 2))
            
            if not player_moves:
                return None
            
            # Determine game result for this player
            result = game_info.get('Result', '')
            player_result = self._determine_player_result(result, player_color)
            
            # Determine cheating label
            label = self._determine_game_label_from_pgn(game_info, player_moves, username)
            
            return {
                'game_id': game_info.get('Site', '').rsplit('/', 1)[-1],
                'username': username,
                'player_color': player_color,
                'moves': ','.join(player_moves),
                'move_times': ','.join(map(str, move_times)),
                'result': player_result,
                'label': label
            }
            
        except Exception as e:
            print(f"Error parsing PGN game: {e}")
            return None
    
    def _determine_game_label_from_pgn(self, game_info: Dict, player_moves: List[str], username: str) -> int:
        """Determine if the game involves cheating based on PGN data"""
        # Pattern-based heuristics (not tactical chess analysis)
        if len(player_moves) < 5:
            return 0  # Not enough moves to analyze
        
        # Check for repetitive move patterns
        if len(player_moves) > 10:
            move_chunks = [player_moves[i:i+3] for i in range(len(player_moves)-2)]
            unique_chunks = len(set(tuple(chunk) for chunk in move_chunks))
            pattern_variety = unique_chunks / len(move_chunks)
            
            # Low pattern variety might indicate engine usage
            if pattern_variety < 0.3:
                return 1  # Potential cheater
        
        # Check for move length consistency patterns
        if len(player_moves) > 5:
            move_lengths = [len(m) for m in player_moves]
            length_std = np.std(move_lengths)
            if length_std < 0.5:
                return 1  # Potential cheater
        
        return 0  # Honest player
    
    def fetch_multiple_players(self, 
                             usernames: List[str], 
                             games_per_player: int = 50,
                             force_refetch: bool = False,
                             **kwargs) -> pd.DataFrame:
        """Fetch games for multiple players"""
        
        all_games = []
        
        for username in tqdm(usernames, desc="Fetching players"):
            games = self.fetch_player_games(
                username=username,
                max_games=games_per_player, 
                force_refetch=force_refetch,
                **kwargs
            )
            all_games.extend(games)
            
            # Rate limiting - increased to avoid 429 errors
            time.sleep(2)
        
        if all_games:
            df = pd.DataFrame(all_games)
            print(f"✅ Fetched {len(df)} total player records")
            return df
        else:
            print("❌ No games fetched")
            return pd.DataFrame()
    
    def append_to_hdf5(self, new_games_df: pd.DataFrame, label: int, backup: bool = True):
        """Append new games to existing HDF5 files"""
        
        if new_games_df.empty:
            print("No new games to append")
            return
        
        # Determine which HDF5 file to append to
        if label == 1:
            target_file = self.cheater_hdf5
            print(f"Appending {len(new_games_df)} cheater records to {target_file}")
        else:
            target_file = self.honest_hdf5
            print(f"Appending {len(new_games_df)} honest records to {target_file}")
        
        # Backup existing file if requested
        if backup and os.path.exists(target_file):
            backup_file = f"{target_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Creating backup: {backup_file}")
            import shutil
            shutil.copy2(target_file, backup_file)
        
        try:
            # Load existing data
            if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                try:
                    existing_df = pd.read_hdf(target_file, key='chess_data')
                    print(f"Loaded {len(existing_df)} existing records")
                    
                    # Combine with new data
                    combined_df = pd.concat([existing_df, new_games_df], ignore_index=True)
                    print(f"Combined dataset has {len(combined_df)} records")
                except Exception as e:
                    print(f"⚠️ Error reading existing file, creating new one: {e}")
                    combined_df = new_games_df
                    print(f"Creating new file with {len(combined_df)} records")
            else:
                combined_df = new_games_df
                print(f"Creating new file with {len(combined_df)} records")
            
            # Save to HDF5 with error handling
            try:
                combined_df.to_hdf(target_file, key='chess_data', mode='w', format='table')
                print(f"✅ Successfully saved to {target_file}")
                
                # Update combined dataset
                self._update_combined_dataset()
            except Exception as e:
                print(f"❌ Error saving to HDF5: {e}")
                # Try to save to a backup file
                backup_file = f"{target_file}.emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    combined_df.to_hdf(backup_file, key='chess_data', mode='w', format='table')
                    print(f"✅ Saved emergency backup to {backup_file}")
                except Exception as backup_e:
                    print(f"❌ Failed to create emergency backup: {backup_e}")
            
        except Exception as e:
            print(f"❌ Error appending to HDF5: {e}")
    
    def _update_combined_dataset(self):
        """Update the combined dataset with all data"""
        try:
            cheater_df = pd.DataFrame()
            honest_df = pd.DataFrame()
            
            # Load cheater data
            if os.path.exists(self.cheater_hdf5):
                cheater_df = pd.read_hdf(self.cheater_hdf5, key='chess_data')
                print(f"Loaded {len(cheater_df)} cheater records")
            
            # Load honest data
            if os.path.exists(self.honest_hdf5):
                honest_df = pd.read_hdf(self.honest_hdf5, key='chess_data')
                print(f"Loaded {len(honest_df)} honest records")
            
            # Combine and shuffle
            if not cheater_df.empty or not honest_df.empty:
                combined_df = pd.concat([cheater_df, honest_df], ignore_index=True)
                combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Save combined dataset
                combined_df.to_hdf(self.combined_hdf5, key='chess_data', mode='w', format='table')
                print(f"✅ Updated combined dataset: {len(combined_df)} total records")
            
        except Exception as e:
            print(f"❌ Error updating combined dataset: {e}")
    
    def fetch_known_cheaters(self, 
                           cheater_usernames: List[str], 
                           games_per_player: int = 100,
                           force_refetch: bool = False) -> pd.DataFrame:
        """Fetch games from known cheaters"""
        print(f"🎯 Fetching games from {len(cheater_usernames)} known cheaters")
        
        df = self.fetch_multiple_players(
            usernames=cheater_usernames,
            games_per_player=games_per_player,
            force_refetch=force_refetch
        )
        
        if not df.empty:
            # All games from cheaters get label 1
            df['label'] = 1
            self.append_to_hdf5(df, label=1)
        
        return df
    
    def fetch_known_honest_players(self, 
                                 honest_usernames: List[str], 
                                 games_per_player: int = 100,
                                 force_refetch: bool = False) -> pd.DataFrame:
        """Fetch games from known honest players"""
        print(f"🎯 Fetching games from {len(honest_usernames)} known honest players")
        
        df = self.fetch_multiple_players(
            usernames=honest_usernames,
            games_per_player=games_per_player,
            force_refetch=force_refetch
        )
        
        if not df.empty:
            # All games from honest players get label 0
            df['label'] = 0
            self.append_to_hdf5(df, label=0)
        
        return df
    
    def fetch_all_stored_players(self, cheater_games_per_player: int = 50, honest_games_per_player: int = 50, force_refetch: bool = False):
        """Fetch games from all stored players with separate limits for cheaters and honest players"""
        cheaters = self.get_cheater_players()
        honest = self.get_honest_players()
        
        print(f"🎯 Fetching from stored players:")
        print(f"  Cheaters: {len(cheaters)} (games per player: {cheater_games_per_player})")
        print(f"  Honest: {len(honest)} (games per player: {honest_games_per_player})")
        
        if cheaters:
            print(f"\n📥 Fetching cheater games...")
            self.fetch_known_cheaters(cheaters, cheater_games_per_player, force_refetch)
        
        if honest:
            print(f"\n📥 Fetching honest games...")
            self.fetch_known_honest_players(honest, honest_games_per_player, force_refetch)
        
        if not cheaters and not honest:
            print("❌ No stored players found. Add players first.")
    
    def get_fetch_summary(self) -> Dict:
        """Get summary of fetched data"""
        summary = {
            'fetched_players': len(self.fetched_players),
            'total_games_fetched': sum(p.get('total_games_fetched', 0) for p in self.fetched_players.values()),
            'hdf5_files': {},
            'stored_players': {
                'cheaters': len(self.get_cheater_players()),
                'honest': len(self.get_honest_players())
            }
        }
        
        # Check HDF5 files
        for file_path, label_name in [(self.cheater_hdf5, 'cheaters'), 
                                     (self.honest_hdf5, 'honest'), 
                                     (self.combined_hdf5, 'combined')]:
            if os.path.exists(file_path):
                try:
                    df = pd.read_hdf(file_path, key='chess_data')
                    summary['hdf5_files'][label_name] = len(df)
                except Exception as e:
                    summary['hdf5_files'][label_name] = f"Error: {e}"
            else:
                summary['hdf5_files'][label_name] = "File not found"
        
        return summary

def main():
    """Main function for fetching games"""
    print("🎯 Lichess HDF5 Game Fetcher")
    print("=" * 50)
    
    # Initialize fetcher
    fetcher = LichessHDF5Fetcher()
    
    # Main menu
    while True:
        print("\nChoose an option:")
        print("1. Manage player lists")
        print("2. Fetch from known cheaters")
        print("3. Fetch from known honest players")
        print("4. Fetch from custom player list")
        print("5. Fetch from all stored players")
        print("6. Show fetch summary")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            # Player list management
            print("\nPlayer List Management:")
            print("1. Add cheater players")
            print("2. Add honest players")
            print("3. Remove cheater players")
            print("4. Remove honest players")
            print("5. Show current lists")
            print("6. Back to main menu")
            
            sub_choice = input("\nEnter choice (1-6): ").strip()
            
            if sub_choice == '1':
                usernames = input("Enter cheater usernames (comma-separated): ").strip().split(',')
                usernames = [u.strip() for u in usernames if u.strip()]
                if usernames:
                    fetcher.add_cheater_players(usernames)
                else:
                    print("No usernames provided")
            
            elif sub_choice == '2':
                usernames = input("Enter honest usernames (comma-separated): ").strip().split(',')
                usernames = [u.strip() for u in usernames if u.strip()]
                if usernames:
                    fetcher.add_honest_players(usernames)
                else:
                    print("No usernames provided")
            
            elif sub_choice == '3':
                usernames = input("Enter cheater usernames to remove (comma-separated): ").strip().split(',')
                usernames = [u.strip() for u in usernames if u.strip()]
                if usernames:
                    fetcher.remove_cheater_players(usernames)
                else:
                    print("No usernames provided")
            
            elif sub_choice == '4':
                usernames = input("Enter honest usernames to remove (comma-separated): ").strip().split(',')
                usernames = [u.strip() for u in usernames if u.strip()]
                if usernames:
                    fetcher.remove_honest_players(usernames)
                else:
                    print("No usernames provided")
            
            elif sub_choice == '5':
                fetcher.show_player_lists()
            
            elif sub_choice == '6':
                continue
        
        elif choice == '2':
            # Fetch from stored cheaters
            cheaters = fetcher.get_cheater_players()
            if cheaters:
                games_per_player = int(input("Games per player (default 50): ") or "50")
                fetcher.fetch_known_cheaters(cheaters, games_per_player)
            else:
                print("❌ No cheater players stored. Add them first in option 1.")
        
        elif choice == '3':
            # Fetch from stored honest players
            honest = fetcher.get_honest_players()
            if honest:
                games_per_player = int(input("Games per player (default 50): ") or "50")
                fetcher.fetch_known_honest_players(honest, games_per_player)
            else:
                print("❌ No honest players stored. Add them first in option 1.")
        
        elif choice == '4':
            # Fetch from custom list
            usernames = input("Enter usernames (comma-separated): ").strip().split(',')
            usernames = [u.strip() for u in usernames if u.strip()]
            
            if usernames:
                games_per_player = int(input("Games per player (default 50): ") or "50")
                label = int(input("Label (0=honest, 1=cheater): ") or "0")
                
                df = fetcher.fetch_multiple_players(usernames, games_per_player)
                if not df.empty:
                    df['label'] = label
                    fetcher.append_to_hdf5(df, label)
            else:
                print("No usernames provided")
        
        elif choice == '5':
            # Fetch from all stored players
            print("\n🎯 Fetch from all stored players")
            print("Enter game limits for each player type:")
            
            cheater_games = int(input("Games per cheater player (default 50): ") or "50")
            honest_games = int(input("Games per honest player (default 50): ") or "50")
            
            print(f"\n📊 Will fetch {cheater_games} games per cheater and {honest_games} games per honest player")
            fetcher.fetch_all_stored_players(cheater_games, honest_games)
        
        elif choice == '6':
            # Show summary
            summary = fetcher.get_fetch_summary()
            print("\n📊 Fetch Summary:")
            print(f"  Players tracked: {summary['fetched_players']}")
            print(f"  Total games fetched: {summary['total_games_fetched']}")
            print(f"  Stored players:")
            print(f"    Cheaters: {summary['stored_players']['cheaters']}")
            print(f"    Honest: {summary['stored_players']['honest']}")
            print(f"  HDF5 files:")
            for name, count in summary['hdf5_files'].items():
                print(f"    {name}: {count}")
        
        elif choice == '7':
            print("Goodbye!")
            break
    
    else:
            print("Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    main() 