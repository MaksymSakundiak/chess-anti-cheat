# Development Status

_Last updated: 2026-10-14_

## Summary

The detector now works at the level that matters, whole accounts. On test
players it never saw in training, it ranks flagged accounts above honest ones
with an **AUC of 0.84**, and it judges a single game with an AUC of 0.72.
These results are averaged over 3 player-level splits of 768 players, with
about 154 test players per split.

Most of the improvement came from adding engine accuracy as a feature and
from a much larger dataset. Feeding the move tokens to the model added
nothing. An earlier figure of 0.90 came from only 20 test players; it was
noise, and 0.84 is the reliable estimate.

## Current pipeline

| Step                  | Script                       | What it does                                                                                                                                                                                                 |
| --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Build player lists | `collect_flagged_players.py` | Crawls the honest players' opponents and looks them up on Lichess. Accounts marked `tosViolation` become cheaters; unflagged opponents from the same pool become honest players. Removes BOT and closed accounts. |
| 2. Download games     | `download_dataset.py`        | Downloads 50 rated rapid games per player with clock data and engine evals. Alternates cheaters and honest players, saves every 25 players, and resumes after interruptions.                                  |
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
  replaced with 309 human accounts flagged for fair-play violations, and 9
  players on the old honest list turned out to be flagged too.
- **Larger dataset:** 768 players and 42,246 games, up from 100 players and
  5,468 games.

### Evaluation

- Train, validation and test sets are split by player, so no player appears
  in more than one set.
- Each player gets a score: the mean of their game scores.
- Thresholds are chosen on validation players instead of a fixed 0.5.
- Training reports cheater-class metrics, balanced accuracy and the
  always-"honest" baseline, separately for games and for players.
- **Leak found and removed:** flagged players' games are analysed more often
  (71% vs 60%), probably because suspicious opponents request analysis. The
  share of analysed games alone separates the classes (AUC 0.63), so the
  production model trains and scores on analysed games only.

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

Dataset: 307 flagged and 461 honest players, 42,246 rated rapid games, 64%
of them analysed. The table shows the mean ± standard deviation over 3
player-level splits (seeds 42, 7 and 123).

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

The production model (seed 42) scores a test-player AUC of 0.84. At its
validation-chosen threshold it flags 36 of 54 test cheaters (67%) and 12 of
99 honest players (12%).

## Known limitations

- **False positives are too high for real use.** About 1 in 8 honest players
  is flagged at the current threshold. A real anti-cheat system needs a much
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

1. **More flagged players.** The crawl found 897 flagged accounts, but only
   300 were used. Run `collect_flagged_players.py --max-flagged 900` (and
   delete `data/opponents_cache.json` to crawl further), then run
   `download_dataset.py`. More cheaters is the cheapest way to improve the
   model.
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
python download_dataset.py          # ~80 minutes; resumable
python train_tensorflow.py          # production configuration
```

To reproduce the comparison table, run `train_tensorflow.py` with
`--split-seed 42`, `7` and `123`, each with and without `--with-moves` and
`--all-games`.

Older datasets are kept locally, and ignored by git, in
`data/old_bot_dataset/` (BOT accounts as cheaters) and
`data/sample100_no_evals/` (100 players, no engine evals).
