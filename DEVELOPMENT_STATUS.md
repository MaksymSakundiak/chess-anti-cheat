# Development Status

_Last updated: 2026-10-14_

## Summary

The pipeline now trains on real data, end to end: games from human Lichess
accounts flagged for fair-play violations versus unflagged players, with real
per-move think times, and an evaluation where test players are never seen in
training.

On a 100-player sample, the model is weak at judging a **single game**
(AUC 0.65). Averaged over a **player's games**, it ranks cheaters well
(AUC 0.90), but that figure comes from only 20 test players, 8 of them
cheaters. The main bottleneck is the amount of data, not the code.

## Current pipeline

| Step                  | Script                           | What it does                                                                                                                                                                                                                                    |
| --------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Build player lists | `collect_flagged_players.py`     | Crawls the opponents of the honest players and looks them up on Lichess. Accounts marked `tosViolation` become cheaters; unflagged opponents from the same pool become honest players. It also removes BOT and closed accounts from both lists. |
| 2. Download games     | `fetch_games.py` (menu option 5) | Downloads rated rapid games per player with clock data, and saves each player's moves and think times to `data/*.h5`.                                                                                                                           |
| 3. Train              | `train_tensorflow.py`            | Splits train/validation/test **by player**, builds the vocabulary from training players only, weights the classes, trains, and reports metrics for the cheater class.                                                                           |
| 4. Predict            | `predict_tensorflow.py`          | Scores one player's moves and times in a single game.                                                                                                                                                                                           |

## Achieved so far

### Data correctness

- **Real move times.** Think time is now computed from Lichess `[%clk]`
  annotations plus the increment. Before, times were generated randomly.
- **PGN parsing fixed.** Black move numbers (`1...`) were being counted as
  moves, which assigned moves to the wrong player. 300 of the 4,732
  vocabulary entries were these artefacts.
- **Game results fixed.** Every game used to be recorded as a loss.
- **Engine annotations removed.** Lichess's own verdicts (`?!`, `??`) are
  stripped from moves, so the model can't learn from them.
- **Real cheaters.** The old "cheater" list was 30 Lichess **BOT** accounts,
  which are engines playing openly, not human cheaters. It has been replaced
  with 309 human accounts flagged for fair-play violations. 9 players on the
  old honest list turned out to be flagged as well.

### Honest evaluation

- **Player-level split.** Before, the same player's games appeared in both
  training and test, so the model could memorise players.
- **Class weighting** during training.
- **Metrics for the cheater class:** precision, recall, F1, balanced accuracy
  and a confusion matrix, all shown next to the accuracy of always answering
  "honest". Before, accuracy was computed with the wrong Keras metric and was
  always 0.
- **Consistent time encoding.** Training and prediction now share the same
  encoders (log-scaled seconds). The old per-game rescaling erased speed and
  consistency.
- **BatchNorm momentum set to 0.9.** With the default, the saved model's
  predictions were squashed toward "honest".

### Project hygiene

- Missing dependencies added: `tensorboard`, `tables`, `python-dotenv`.
- Docs corrected: there's no attention layer, and the "85–90% accuracy" claim
  was removed because nobody measured it.

## Current results

Dataset: 50 flagged and 50 honest players, 5,468 rated rapid games.
Split: 63 training players, 17 validation, 20 test (8 cheaters).

| Level                        | AUC      | Notes                                                                                                                    |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| Single game                  | 0.65     | Balanced accuracy 0.54. At the 0.5 cut-off it flags 86% of honest games, so 0.5 is the wrong threshold for this model.   |
| Player (mean of their games) | **0.90** | The top 5 are all cheaters, and 6 of the 8 cheaters rank in the top 8. Only 20 players, so treat this as a rough figure. |

For comparison, the earlier run on the bot dataset scored 0.78 per game. That
result doesn't mean much, because telling bots from humans is a much easier
problem than catching human cheaters.

The model **memorises training players**: training recall rises to 0.92
while validation AUC drifts from 0.65 down to 0.54. Early stopping keeps the
epoch-4 weights.

## Known limitations

- **Small sample.** Only 100 of the 770 listed players have been downloaded,
  so the test set has 8 cheaters.
- **Labels are noisy.** "Not flagged" doesn't prove someone is honest, and a
  flagged account's older games may come from before it started cheating.
- **Scores are uncalibrated.** Single-game scores sit in a narrow band, so a
  fixed 0.5 threshold doesn't work.
- **Single-game predictions only.** `predict_tensorflow.py` doesn't combine
  scores across a player's games yet.
- **Rule-of-thumb risk factors.** Some of the "risk factors" in
  `predict_tensorflow.py` don't measure anything meaningful. For example,
  "consistent move lengths" measures how many characters the move notation
  has.
- **Model size and data size don't match.** The model has about 1.5M
  parameters but only about 60 training players, which explains the
  memorisation.
- **Outdated docs.** `HDF5_FORMAT_GUIDE.md` still describes the
  `convert_to_hdf5.py` path, and `FIXES_SUMMARY.md` describes earlier fixes.
  Both predate this work.

## Next steps (in priority order)

1. **Download more players.** Lists of 309 flagged and 461 honest players
   are ready. About 300 of each takes roughly 1 hour at about 6 seconds per
   player. `collect_flagged_players.py` can find more flagged accounts: the
   crawl found 897, and it adds 300 by default (`--max-flagged`).
2. **Score at the player level.** Report per-player AUC during training, and
   add a "score this account over N games" mode to the predictor.
3. **Choose the threshold on validation players** instead of using 0.5, for
   example by targeting a fixed false-positive rate for honest players.
4. **Reduce overfitting.** Try a smaller model, stronger dropout, or fewer
   epochs, and compare validation AUC across a few seeds.
5. **Add stronger features.** Engine evaluations (`[%eval]`) are already
   requested from Lichess but thrown away. Comparing a player's moves with
   the engine's choices (centipawn loss) is the standard cheating signal.
6. **Replace the rule-of-thumb risk factors** with real timing statistics,
   such as the spread of think times and the share of fast moves in complex
   positions.
7. **Tidy the docs.** Update or remove `HDF5_FORMAT_GUIDE.md` and
   `FIXES_SUMMARY.md`.

## Reproducing the current results

```bash
pip install -r requirements.txt
python collect_flagged_players.py      # optional: refresh the player lists
python fetch_games.py                  # option 5, or sample players as below
python train_tensorflow.py
```

The current run downloaded a seeded random sample of 50 players from each
list (`random.Random(42)`, 50 games per player). The earlier bot-based data
is kept locally in `data/old_bot_dataset/`, which git ignores.
