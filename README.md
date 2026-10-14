# Chess Anti-Cheat Detection with TensorFlow

Detects engine-assisted play on Lichess from how a player spends their clock
and how closely their moves match the engine, using a TensorFlow/Keras model
trained on accounts Lichess has flagged for fair-play violations.

## 🚀 Features

- **Real labels**: Cheater examples are human accounts Lichess marked for
  Terms of Service violations; honest examples come from the same opponent pool
- **Real signals**: Per-move think times from Lichess clock data and per-move
  engine accuracy from Lichess's own analysis
- **Account scoring**: Scores a Lichess account over its recent games and lists
  the most suspicious ones
- **Honest evaluation**: Train/validation/test split by player, thresholds
  chosen on validation players, results reported per game and per player

## 📈 Current Performance

768 players (307 flagged, 461 honest), 42,246 rated rapid games. Averaged over
3 player-level splits, about 154 test players each:

| Level | AUC |
|---|---|
| Single game | 0.72 |
| Player (mean score over their analysed games) | 0.84 |

A single game carries little evidence; judge accounts, not games. See
[DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for the full comparison and
known limitations.

## 📊 Model Architecture

```
Per-move features (up to 300 moves)
├── is_move        1 for real moves, 0 for padding (masked)
├── log_time       log1p(seconds spent on the move)
├── has_eval       1 if Lichess's engine analysed the position
└── eval_loss      log1p(winning chance the move gave away, 0-100%)

Feature extraction
├── Dense(32) → Dense(32) per move → masked average over moves
└── Conv1D(32, kernel 5) → global max pooling

Player color → Dense(8)

Classification
└── Dense(128) → Dense(64), BatchNorm + Dropout 0.4 → sigmoid
```

About 20k parameters. An optional branch (`--with-moves`) adds move tokens
through an embedding and a BiLSTM; it added no accuracy in testing and trains
about 20× slower.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/MaksymSakundiak/chess-anti-cheat.git
cd chess-anti-cheat

# Install dependencies
pip install -r requirements.txt
```

A Lichess API token in `.env` (`LICHESS_API_TOKEN=...`) is optional but gives
higher rate limits.

## 📁 Project Structure

```
chess-anti-cheat/
├── data/                          # Downloaded games (git-ignored)
│   ├── combined_games.h5          # All games, used for training
│   ├── cheater_games.h5           # Flagged players' games
│   └── honest_games.h5            # Honest players' games
├── collect_flagged_players.py     # Build the player lists from Lichess
├── download_dataset.py            # Download games for the player lists (resumable)
├── fetch_games.py                 # Lichess download and PGN parsing
├── improved_tensorflow_model.py   # Model and feature encoding
├── train_tensorflow.py            # Training and evaluation
├── predict_tensorflow.py          # Score accounts or single games
├── cheater_players.json           # Flagged accounts
├── honest_players.json            # Unflagged accounts
├── model_meta.json                # Thresholds from the last training run
└── requirements.txt               # Dependencies
```

## 🎯 Quick Start

### 1. Build the player lists (optional, lists are included)

```bash
python collect_flagged_players.py
```

### 2. Download games

```bash
python download_dataset.py                    # all players, ~80 minutes
python download_dataset.py --max-players 50   # 50 per class, ~10 minutes
```

Progress is saved after every 25 players; rerun the same command to resume.

### 3. Train

```bash
python train_tensorflow.py
```

Trains in about a minute on CPU and writes `advanced_chess_model.keras`,
`hdf5_vocab.pkl` and `model_meta.json`.

### 4. Score an account

```bash
python predict_tensorflow.py
# 1. Analyse a Lichess account (recommended)
```

## 🔍 Usage Examples

### Account analysis

```
ACCOUNT: SomePlayer
PREDICTION: Suspicious
Account score: 0.4983 (threshold 0.4773, mean over 21 games)
Most suspicious games:
  https://lichess.org/AbCd1234  score 0.735
  ...
Statistics:
  Mean think time (s): 7.61
  Mean win-chance loss per move (%): 2.81
  Share of near-perfect moves (<1% loss): 0.63
  ...
```

Accounts with fewer than 10 analysed rapid games get no verdict.

### Programmatic usage

```python
from predict_tensorflow import TensorFlowChessPredictor

predictor = TensorFlowChessPredictor()

# Whole account (downloads recent rated rapid games)
result = predictor.predict_account("SomePlayer", max_games=50)
print(result['prediction'], result['account_score'])

# One game, from one player's perspective
result = predictor.predict_player_moves(
    "e4,Nf3,Bb5,O-O",          # this player's moves
    "0.0,3.2,4.1,2.9",         # seconds spent on each move
    player_color="white",
    move_losses_str="0.1,0.4,0.0,1.2"   # winning chance lost per move (%)
)
```

## 📊 Data Format

One row per player per game:

| Column         | Description                                      | Example           |
| -------------- | ------------------------------------------------ | ----------------- |
| `game_id`      | Lichess game id                                  | "AbCd1234"        |
| `username`     | The player this row describes                    | "SomePlayer"      |
| `player_color` | "white" or "black"                               | "white"           |
| `moves`        | The player's moves (SAN), comma-separated        | "e4,Nf3,Bb5"      |
| `move_times`   | Seconds spent per move, from clock data          | "0.0,3.0,36.5"    |
| `move_losses`  | Winning chance lost per move (%); empty if unanalysed | "0.0,0.4,11.8" |
| `result`       | "win", "loss" or "draw"                          | "win"             |
| `label`        | 0 (honest) or 1 (flagged)                        | 1                 |

## 🔧 Configuration

```bash
python train_tensorflow.py --epochs 15 --dropout 0.4 --split-seed 42
python train_tensorflow.py --with-moves   # add the move-token BiLSTM branch
python train_tensorflow.py --all-games    # include games without engine analysis
```

`--all-games` scores higher (player AUC ≈ 0.88), but partly because flagged
players' games are analysed more often, which reflects their opponents'
suspicion rather than their play.

## 📈 Performance Metrics

Training reports, on test players never seen in training:

- **Per game**: accuracy next to the always-"honest" baseline, balanced
  accuracy, cheater precision / recall / F1, AUC
- **Per player**: AUC, balanced accuracy and confusion matrix of the mean
  game score
- **Thresholds**: chosen on validation players (maximum balanced accuracy)
  and saved to `model_meta.json`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the TensorFlow model
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for advanced neural network capabilities
- Lichess API for chess game data
- Chess community for feedback and testing
