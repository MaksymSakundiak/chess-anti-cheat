#!/usr/bin/env python3
"""
Collect Lichess accounts flagged for Terms of Service (fair play) violations

Crawls the opponents of the players in honest_players.json, looks them up in
bulk via the Lichess users API, and keeps open human accounts marked with
tosViolation as cheater examples. Non-flagged opponents from the same pool
are added as honest examples, so both classes come from similar players.

Also cleans the existing lists: BOT accounts and closed accounts are removed,
and players in the honest list who have since been flagged move to the
cheater list.

Usage:
    python collect_flagged_players.py                # crawl and update lists
    python collect_flagged_players.py --dry-run      # report only
"""

import argparse
import json
import os
import random
import re
import time
from typing import Dict, Iterable, List

import requests

from fetch_games import LichessHDF5Fetcher

USERS_API_BATCH = 300  # Max usernames per POST /api/users request
OPPONENTS_CACHE = 'data/opponents_cache.json'
PGN_PLAYERS_RE = re.compile(r'\[White "([^"]+)"\]\s*\[Black "([^"]+)"\]')


def request_with_retry(fetcher: LichessHDF5Fetcher, method: str, url: str, retries: int = 5, **kwargs):
    """Send a request, backing off 60s, 120s, 240s... whenever Lichess rate limits us"""
    for attempt in range(retries + 1):
        response = fetcher.session.request(method, url, **kwargs)
        if response.status_code != 429 or attempt == retries:
            return response
        wait = 60 * 2 ** attempt
        print(f"⚠️ Rate limited, waiting {wait} seconds...")
        time.sleep(wait)


def lookup_users(fetcher: LichessHDF5Fetcher, usernames: Iterable[str]) -> Dict[str, Dict]:
    """Bulk-fetch public account data, keyed by lowercase username.
    Closed accounts come back with 'disabled'; unknown names are omitted."""
    usernames = list(usernames)
    users = {}
    for start in range(0, len(usernames), USERS_API_BATCH):
        batch = usernames[start:start + USERS_API_BATCH]
        response = request_with_retry(
            fetcher, 'POST', 'https://lichess.org/api/users',
            data=','.join(batch),
            headers={'Content-Type': 'text/plain'}
        )
        response.raise_for_status()
        for user in response.json():
            users[user['id']] = user
        if len(usernames) > USERS_API_BATCH:
            print(f"   Looked up {min(start + USERS_API_BATCH, len(usernames))}/{len(usernames)}")
        time.sleep(5)
    return users


def rapid_games(user: Dict) -> int:
    return user.get('perfs', {}).get('rapid', {}).get('games', 0)


def is_usable(user: Dict, min_rapid_games: int) -> bool:
    """Open, human account with enough rapid games to fetch"""
    return (not user.get('disabled')
            and user.get('title') != 'BOT'
            and rapid_games(user) >= min_rapid_games)


def collect_opponents(fetcher: LichessHDF5Fetcher, seeds: List[str], games_per_seed: int) -> List[str]:
    """Usernames of everyone the seed players met in recent rated rapid games.

    Progress is saved to OPPONENTS_CACHE after every seed, so an interrupted
    crawl resumes where it stopped. Delete the cache file to crawl from scratch.
    """
    cache = {'seeds_done': [], 'opponents': {}}
    if os.path.exists(OPPONENTS_CACHE):
        with open(OPPONENTS_CACHE) as f:
            cache = json.load(f)
        print(f"   Resuming: {len(cache['seeds_done'])} seeds already crawled")
    os.makedirs(os.path.dirname(OPPONENTS_CACHE), exist_ok=True)

    seen = {seed.lower() for seed in seeds}
    done = set(cache['seeds_done'])
    opponents = cache['opponents']
    for i, seed in enumerate(seeds, 1):
        if seed in done:
            continue
        for attempt in range(2):
            try:
                response = request_with_retry(
                    fetcher, 'GET', f'https://lichess.org/api/games/user/{seed}',
                    params={'max': games_per_seed, 'rated': 'true', 'perfType': 'rapid', 'moves': 'false'},
                    headers={'Accept': 'application/x-chess-pgn'}
                )
                break
            except requests.exceptions.RequestException as e:
                print(f"   {seed}: connection error ({e.__class__.__name__}), "
                      f"{'retrying in 10s' if attempt == 0 else 'skipped'}")
                response = None
                time.sleep(10)
        if response is None:
            continue
        if response.status_code != 200:
            print(f"   {seed}: HTTP {response.status_code}, skipped")
        else:
            for white, black in PGN_PLAYERS_RE.findall(response.text):
                opponent = black if white.lower() == seed.lower() else white
                if opponent.lower() not in seen:
                    opponents.setdefault(opponent.lower(), opponent)
            print(f"   [{i}/{len(seeds)}] {seed}: {len(opponents)} unique opponents so far")

        cache['seeds_done'].append(seed)
        with open(OPPONENTS_CACHE, 'w') as f:
            json.dump(cache, f)
        time.sleep(1)
    return list(opponents.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--games-per-seed', type=int, default=200,
                        help='Recent rapid games to scan per seed player (default 200)')
    parser.add_argument('--min-rapid-games', type=int, default=20,
                        help='Minimum rated rapid games an account needs (default 20)')
    parser.add_argument('--max-flagged', type=int, default=300,
                        help='Maximum flagged accounts to add (default 300)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing the player lists')
    args = parser.parse_args()

    fetcher = LichessHDF5Fetcher()
    cheaters = fetcher.get_cheater_players()
    honest = fetcher.get_honest_players()

    # 1. Audit the existing lists
    print(f"🔍 Checking {len(cheaters)} cheater and {len(honest)} honest players...")
    current = lookup_users(fetcher, cheaters + honest)
    status = {}
    for name in cheaters + honest:
        user = current.get(name.lower())
        if user is None or user.get('disabled'):
            status[name] = 'closed'
        elif user.get('title') == 'BOT':
            status[name] = 'bot'
        elif user.get('tosViolation'):
            status[name] = 'flagged'
        else:
            status[name] = 'ok'

    remove_cheaters = [n for n in cheaters if status[n] in ('closed', 'bot')]
    remove_honest = [n for n in honest if status[n] != 'ok']
    honest_now_flagged = [n for n in honest if status[n] == 'flagged']
    unflagged_cheaters = [n for n in cheaters if status[n] == 'ok']
    print(f"   Cheater list: {sum(status[n] == 'bot' for n in cheaters)} bots, "
          f"{sum(status[n] == 'closed' for n in cheaters)} closed, "
          f"{len(unflagged_cheaters)} open accounts without a flag")
    print(f"   Honest list: {len(honest_now_flagged)} now flagged, "
          f"{sum(status[n] == 'bot' for n in honest)} bots, "
          f"{sum(status[n] == 'closed' for n in honest)} closed")
    if unflagged_cheaters:
        print(f"   ⚠️ Kept but unverified cheaters: {unflagged_cheaters}")

    # 2. Crawl the honest players' opponents and look them up
    seeds = [n for n in honest if status[n] == 'ok']
    print(f"\n🕸️ Collecting opponents of {len(seeds)} honest players...")
    opponents = collect_opponents(fetcher, seeds, args.games_per_seed)
    known = {n.lower() for n in cheaters + honest}
    opponents = [o for o in opponents if o.lower() not in known]
    print(f"\n🔍 Looking up {len(opponents)} opponents...")
    users = lookup_users(fetcher, opponents)

    usable = [u for u in users.values() if is_usable(u, args.min_rapid_games)]
    flagged = [u['username'] for u in usable if u.get('tosViolation')]
    clean = [u['username'] for u in usable if not u.get('tosViolation')]
    print(f"   {len(users)} accounts found, {len(usable)} usable "
          f"(open, human, ≥{args.min_rapid_games} rapid games)")
    print(f"   🚩 {len(flagged)} flagged for ToS violations")

    random.seed(42)
    random.shuffle(flagged)
    new_cheaters = flagged[:args.max_flagged]

    # Match honest examples to the same opponent pool, one per new cheater
    random.shuffle(clean)
    new_honest = clean[:len(new_cheaters)]

    final_cheaters = len(cheaters) - len(remove_cheaters) + len(honest_now_flagged) + len(new_cheaters)
    final_honest = len(honest) - len(remove_honest) + len(new_honest)
    print(f"\n📋 Plan:")
    print(f"   Cheater list: -{len(remove_cheaters)} bots/closed, "
          f"+{len(honest_now_flagged)} from honest list, +{len(new_cheaters)} flagged opponents "
          f"-> {final_cheaters}")
    print(f"   Honest list: -{len(remove_honest)} flagged/bots/closed, "
          f"+{len(new_honest)} unflagged opponents -> {final_honest}")

    if args.dry_run:
        print("\n(dry run, lists not changed)")
        return

    fetcher.remove_cheater_players(remove_cheaters)
    fetcher.add_cheater_players(honest_now_flagged + new_cheaters)
    fetcher.remove_honest_players(remove_honest)
    fetcher.add_honest_players(new_honest)
    print("\n✅ Player lists updated. Next: delete data/*.h5, run fetch_games.py (option 5), "
          "then train_tensorflow.py")


if __name__ == '__main__':
    main()
