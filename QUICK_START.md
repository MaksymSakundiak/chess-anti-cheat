# Quick Start Guide - TensorFlow Chess Anti-Cheat

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Optional: put a Lichess API token in `.env` for higher rate limits:
```
LICHESS_API_TOKEN=your_token_here
```

### 2. Download games
```bash
python download_dataset.py --max-players 50   # quick: 50 per class, ~10 minutes
python download_dataset.py                    # full: all 2,570 players, ~4 hours
```
Progress is saved every 25 players. If it stops, rerun the same command to
resume.

### 3. Train
```bash
python train_tensorflow.py
```
Takes about a minute on CPU.

### 4. Score an account
```bash
python predict_tensorflow.py
```
Choose **1. Analyse a Lichess account** and enter a username.

## 🎯 Example Output

```
ACCOUNT: SomePlayer
============================================================
PREDICTION: Suspicious
Account score: 0.4983 (threshold 0.4773, mean over 21 games)
Most suspicious games:
  https://lichess.org/AbCd1234  score 0.735
  ...
Statistics:
  Mean think time (s): 7.61
  Think-time variation (std / mean): 1.36
  Share of moves under 2s: 0.25
  Moves with engine evaluation: 692
  Mean win-chance loss per move (%): 2.81
  Share of near-perfect moves (<1% loss): 0.63
  Share of mistakes (>=10% loss): 0.08
```

- **Account score** is the mean of the account's game scores. Above the
  threshold, the account is "Suspicious".
- The **threshold** was chosen on validation players during training and
  is stored in `model_meta.json`.
- Accounts with fewer than 10 analysed rapid games get no verdict.
- A "Suspicious" result is a reason to look at the account more closely, not
  proof of cheating. The shipped model flags 9% of honest test players.

## 📈 What the Model Looks At

| Feature                  | Source                                                 |
| ------------------------ | ------------------------------------------------------ |
| Think time per move      | Lichess clock data (`[%clk]`)                          |
| Engine accuracy per move | Winning chance lost per move, from Lichess's `[%eval]` |
| Player color             | Game header                                            |

Only games with Lichess computer analysis are used, because whether a game
was analysed depends on the opponents, not on the player.

## 🔧 Training Options

```bash
python train_tensorflow.py --epochs 15        # maximum epochs (early stopping on)
python train_tensorflow.py --split-seed 7     # different player split
python train_tensorflow.py --with-moves       # add the move-token BiLSTM (slow, no gain so far)
python train_tensorflow.py --all-games        # include unanalysed games (inflated scores)
```

## 🛠️ Troubleshooting

**`data/combined_games.h5 not found`**: run `python download_dataset.py` first.

**`No model at advanced_chess_model.keras`**: run `python train_tensorflow.py` first.

**`TensorBoard is not installed`**: run `pip install -r requirements.txt`
(TensorFlow 2.20+ doesn't include TensorBoard).

**Rate limited by Lichess**: `collect_flagged_players.py` waits and retries.
`download_dataset.py` skips a player whose download fails (under 2% of
players in the 2,570-player download). A token in `.env` raises the limits.

**Download interrupted**: rerun `python download_dataset.py`. It skips
players that are already done.

## 📊 Current Performance

On test players never seen in training, averaged over 3 player splits:

- **Accounts:** AUC 0.88
- **Single games:** AUC 0.75

See [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) for details and next steps.
