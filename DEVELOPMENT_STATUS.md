# Development Status

_Last updated: 2026-10-15_

## Summary

The detector works at the level that matters, whole accounts. On test
players it never saw in training, it ranks flagged accounts above honest ones
with an **AUC of 0.88**, and it judges a single game with an AUC of 0.75.
These results are averaged over 3 player-level splits of 2,526 players, with
about 505 test players per split.

The gains came from engine accuracy as a feature and from more data. Going
from 768 to 2,526 players raised the account AUC from 0.84 to 0.88, so more
data is still paying off. Feeding the move tokens to the model added nothing.
An earlier figure of 0.90 came from only 20 test players and was noise.

## Current pipeline

| Step                  | Script                       | What it does                                                                                                                                                                                                 |
| --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Build player lists | `collect_flagged_players.py` | Crawls the honest players' opponents and looks them up on Lichess. Accounts marked `tosViolation` become cheaters; unflagged opponents from the same pool become honest players. Removes BOT and closed accounts. |
| 2. Download games     | `download_dataset.py`        | Downloads 50 rated rapid games per player with clock data and engine evals. Alternates cheaters and honest players, saves every 25 players, resumes after interruptions, and relabels saved games when the lists change. |
| 3. Train              | `train_tensorflow.py`        | Splits players into train/validation/test, trains on analysed games, chooses game- and player-level thresholds on validation players, and saves them to `model_meta.json`.                                  |
| 4. Predict            | `predict_tensorflow.py`      | Scores a Lichess account over its recent games (mean score), lists the most suspicious games, and reports timing and accuracy statistics. It can also score a single game.                                 |

## Achieved so far

### Data

- **Real move times** from Lichess `[%clk]` annotations. They used to be
  generated randomly.
- **Engine accuracy per move:** the share of winning chance each move gave
  away, computed from Lichess's `[%eval]` annotations. This is the same model
  behind Lichess's accuracy score.
- **PGN parsing fixed:** black move numbers (`1...`) had been counted as moves,
  so moves were attributed to the wrong player. Game results were all
  recorded as losses. Lichess's `?!`/`??` annotations were leaking into the
  move tokens.
- **Real cheaters:** the original 30 "cheaters" were BOT accounts. They've been
  replaced with 1,209 human accounts flagged for fair-play violations, and 9
  players on the old honest list turned out to be flagged too.
- **Larger dataset:** 2,526 players and 140,112 games (73,698 analysed), up
  from 100 players and 5,468 games.

### Evaluation

- Train, validation and test sets are split by player, so no player appears
  in more than one set.
- Each player gets a score: the mean of their game scores.
- Thresholds are chosen on validation players instead of a fixed 0.5.
- Training reports cheater-class metrics, balanced accuracy and the
  always-"honest" baseline, separately for games and for players.
- **Leak found and removed:** flagged players' games are analysed more often
  (64% vs 44% in the current dataset), probably because suspicious opponents
  request analysis. The share of analysed games alone separated the classes
  (AUC 0.63 on the 768-player dataset), so the production model trains and
  scores on analysed games only.

### Model and predictor

- Each move is encoded as 4 features (is_move, log think time, has_eval,
  log eval loss), with a masked average over the moves, so the model can
  learn averages such as mean accuracy.
- The production model has about 20k parameters, down from 1.5M, and trains
  in about a minute.
- Early stopping now watches validation AUC.
- The predictor can score a whole account and refuses to give a verdict on
  fewer than 10 analysed games.
- The rule-of-thumb "risk factors" were replaced with real timing and
  accuracy statistics.

### Project hygiene

- Missing dependencies were added (`tensorboard`, `tables`,
  `python-dotenv`).
- Docs were rewritten to match the code. The unmeasured "85–90% accuracy"
  claim and the nonexistent attention layer were removed.

## Current results

### Production model, 2,526 players

Dataset: 1,182 flagged and 1,344 honest players with games, 140,112 rated
rapid games, of which 73,698 are analysed and used. Each split has about
1,615 training and 505 test players.

| Split seed | Game AUC | Player AUC | Player balanced accuracy | Cheaters caught | Honest flagged |
| ---------- | -------- | ---------- | ------------------------ | --------------- | -------------- |
| 7          | 0.750    | 0.891      | 0.82                     | 192 / 238 (81%) | 47 / 267 (18%) |
| 123        | 0.755    | 0.873      | 0.78                     | 178 / 242 (74%) | 44 / 263 (17%) |
| 42         | 0.759    | 0.872      | 0.79                     | 153 / 229 (67%) | 24 / 276 (9%)  |
| **Mean**   | **0.754**| **0.879 ± 0.009** | **0.80**          | 74%             | 14%            |

The shipped model is the seed-42 run. Its validation-chosen threshold is the
most conservative of the three: it flags 9% of honest test players and
catches 67% of cheaters.

### Configuration comparison, 768 players

The table shows the mean ± standard deviation over 3 player-level splits
(seeds 42, 7 and 123) on the earlier 768-player dataset (307 flagged and 461
honest players, 42,246 games).

| Configuration                        | Game AUC | Player AUC        | Player balanced accuracy | Params | Train time |
| ------------------------------------ | -------- | ----------------- | ------------------------ | ------ | ---------- |
| Features only, analysed games **(production)** | 0.715    | **0.841 ± 0.008** | 0.76                     | 20k    | < 1 min    |
| Features only, all games             | 0.711    | 0.881 ± 0.030     | 0.81                     | 20k    | < 1 min    |
| Moves + features, analysed games     | 0.704    | 0.834 ± 0.011     | 0.77                     | 300k   | ~8 min     |
| Moves + features, all games          | 0.711    | 0.883 ± 0.019     | 0.80                     | 300k   | ~15–40 min |

How to read this:

- **The move tokens add nothing.** Models with and without them score within
  noise of each other.
- **The "all games" models score higher partly because of the leak.** Whether
  a game was analysed depends on opponents, not on the player. The
  analysed-only figure (0.84) is the honest one.
- **Simple baselines to beat:** mean eval loss alone gives a player AUC of
  0.75 (flagged players lose 1.81% win chance per move vs 2.30% for honest
  players), and median think time alone gives 0.63. The model beats both.

## Known limitations

- **False positives are too high for real use.** 9% of honest test players
  are flagged by the shipped model, and 14% on average across splits. A real anti-cheat system needs a much
  lower false-positive rate and should treat this score as a reason to
  review an account, not as a verdict.
- **Labels are noisy.** "Not flagged" doesn't prove someone is honest, and a
  flagged account's older games may come from before it started cheating.
- **Rapid only.** Blitz and bullet aren't downloaded.
- **No position difficulty.** The model sees how long a move took and how
  good it was, but not how hard the position was. Quick, perfect moves in
  complicated positions are the classic sign of engine use.
- **Mean aggregation.** Averaging over games dilutes players who only cheat
  in some of their games.
- **Outdated docs.** `HDF5_FORMAT_GUIDE.md` and `FIXES_SUMMARY.md` describe
  the old pipeline.

## Next steps (in priority order)

1. **More flagged players.** Tripling the data raised the account AUC from
   0.84 to 0.88, so this is still the cheapest improvement. The latest crawl
   found 1,787 flagged accounts and 1,209 are now used. Run
   `collect_flagged_players.py --max-flagged 900` again to add the rest, then
   `download_dataset.py` (about 6 seconds per player).
2. **Position difficulty features.** Add the eval swing of the position, the
   number of legal moves, or whether the best move was a capture or a check.
   This lets the model weigh accuracy against difficulty and time.
3. **Better account aggregation.** Try the top-k or a high percentile of game
   scores instead of the mean, so the model can catch players who cheat in
   only some of their games.
4. **Choose the operating threshold for a target false-positive rate**, for
   example at most 5% of honest players flagged, and report the recall at
   that point.
5. **Add blitz games** and a time-control feature.
6. **Tests.** Unit tests for the PGN parser (clock times, eval losses, move
   numbers) and a small end-to-end training test.
7. **Tidy the docs.** Update or remove `HDF5_FORMAT_GUIDE.md` and
   `FIXES_SUMMARY.md`.

## Reproducing the current results

```bash
pip install -r requirements.txt
python collect_flagged_players.py   # optional: refresh the player lists
python download_dataset.py          # ~4 hours for all 2,570 players; resumable
python train_tensorflow.py          # production configuration
```

To reproduce the comparison table, run `train_tensorflow.py` with
`--split-seed 42`, `7` and `123`, each with and without `--with-moves` and
`--all-games`.

Older datasets are kept locally, and ignored by git, in
`data/old_bot_dataset/` (BOT accounts as cheaters) and
`data/sample100_no_evals/` (100 players, no engine evals).
