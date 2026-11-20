"""SQLite store for anonymous translation feedback.

Usage:
    from feedback import FeedbackStore

    store = FeedbackStore("feedback.db")
    store.add("eng_Latn", "swh_Latn", "Hello.", "Habari.", rating="up")
"""
import sqlite3
import threading
from datetime import datetime, timezone


class FeedbackStore:
    def __init__(self, path="feedback.db"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    rating TEXT CHECK (rating IN ('up','down')),
                    correction TEXT
                )
                """
            )
            self._conn.commit()

    def add(self, source_lang, target_lang, source_text, model_output, rating=None, correction=None):
        if rating not in (None, "up", "down"):
            raise ValueError("rating must be 'up', 'down', or None")
        if correction is not None:
            correction = correction.strip()
            if not correction:
                correction = None
        if rating is None and correction is None:
            raise ValueError("Provide a rating, a correction, or both")
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO feedback (created_at, source_lang, target_lang, source_text, model_output, rating, correction) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (created_at, source_lang, target_lang, source_text, model_output, rating, correction),
            )
            self._conn.commit()
            return cur.lastrowid

    def count(self):
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def close(self):
        self._conn.close()
