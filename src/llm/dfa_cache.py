"""Versioned, transactional storage for grammar transitions, never model scores."""
import json
from pathlib import Path
import sqlite3


def pack_state(state):
    configs, word, needs_space = state
    return [sorted((name, list(prefix)) for name, prefix in configs), word, needs_space]


def unpack_state(state):
    configs, word, needs_space = state
    return frozenset((name, tuple(prefix)) for name, prefix in configs), word, needs_space


class TransitionCache:
    def __init__(self, directory, fingerprint):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/f'{fingerprint}.sqlite3'
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS transitions (state TEXT PRIMARY KEY, choices TEXT NOT NULL)')
        self.hits = 0
        self.misses = 0

    def get(self, state):
        key = json.dumps(pack_state(state), separators=(',', ':'))
        row = self.connection.execute('SELECT choices FROM transitions WHERE state=?', (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        # Preserve token ordering as well as values, so warm/cold sampling agrees.
        return {int(token): unpack_state(following) for token, following in json.loads(row[0])}

    def put(self, state, choices):
        key = json.dumps(pack_state(state), separators=(',', ':'))
        value = json.dumps([(token, pack_state(following)) for token, following in choices.items()], separators=(',', ':'))
        with self.connection:
            self.connection.execute('INSERT OR IGNORE INTO transitions VALUES (?, ?)', (key, value))

    def stats(self):
        count = self.connection.execute('SELECT COUNT(*) FROM transitions').fetchone()[0]
        return dict(path=str(self.path), disk_hits=self.hits, computed_states=self.misses, stored_states=count)

    def close(self):
        self.connection.close()
