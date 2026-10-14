#!/usr/bin/env python3
"""
Download games for every player in cheater_players.json and honest_players.json

Players are fetched in small chunks, alternating cheaters and honest players so a
partial download stays balanced. Each chunk is appended to data/*.h5 as soon as
it's fetched, and finished players are recorded in data/download_progress.json,
so an interrupted download resumes where it stopped.

Usage:
    python download_dataset.py                       # everything, 50 games each
    python download_dataset.py --max-players 100     # first 100 per class
    python download_dataset.py --fresh               # ignore progress, start over
"""

import argparse
import json
import os
import random
import time

from fetch_games import LichessHDF5Fetcher

PROGRESS_FILE = 'data/download_progress.json'


def load_progress() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(done: set):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(sorted(done), f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--games-per-player', type=int, default=50)
    parser.add_argument('--max-players', type=int, default=None,
                        help='Maximum players per class (default: all)')
    parser.add_argument('--chunk-size', type=int, default=25,
                        help='Players fetched between saves (default 25)')
    parser.add_argument('--fresh', action='store_true',
                        help='Forget download progress (does not delete data/*.h5)')
    args = parser.parse_args()

    fetcher = LichessHDF5Fetcher()
    if args.fresh and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    done = load_progress()

    # Seeded shuffle, so --max-players picks the same players on every run
    rng = random.Random(42)
    queues = {}
    for label, players in [(1, fetcher.get_cheater_players()), (0, fetcher.get_honest_players())]:
        players = list(players)
        rng.shuffle(players)
        queues[label] = players[:args.max_players]

    # Alternate cheaters and honest players, skipping finished ones
    todo = []
    for pair in zip(queues[1], queues[0]):
        todo.extend(zip((1, 0), pair))
    longer = queues[1] if len(queues[1]) > len(queues[0]) else queues[0]
    longer_label = 1 if longer is queues[1] else 0
    todo.extend((longer_label, p) for p in longer[min(len(queues[1]), len(queues[0])):])
    todo = [(label, p) for label, p in todo if p.lower() not in done]

    total = len(queues[0]) + len(queues[1])
    print(f"📥 {len(todo)} players to download ({total - len(todo)} of {total} already done), "
          f"{args.games_per_player} games each")
    started = time.time()

    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        for label in (1, 0):
            usernames = [p for l, p in chunk if l == label]
            if not usernames:
                continue
            df = fetcher.fetch_multiple_players(usernames, games_per_player=args.games_per_player)
            if not df.empty:
                df['label'] = label
                fetcher.append_to_hdf5(df, label=label, backup=False)

        # Players that came back empty (closed since, or rate limited) are
        # marked done too; run with --fresh to retry them
        done.update(p.lower() for _, p in chunk)
        save_progress(done)
        finished = start + len(chunk)
        elapsed = time.time() - started
        remaining = elapsed / finished * (len(todo) - finished)
        print(f"\n💾 Progress: {finished}/{len(todo)} players this run, "
              f"~{remaining / 60:.0f} min remaining\n")

    print("✅ Download complete")


if __name__ == '__main__':
    main()
