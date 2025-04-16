import sqlite3
from typing import Optional

SETTINGS_DB = "settings.db"

class SettingsService:
    """
    Service for getting and setting application configuration (e.g. library_root).
    Uses a simple key-value table in settings.db.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or SETTINGS_DB
        self._ensure_table()

    def _ensure_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

    def get(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value)
            )
            conn.commit()

    def all(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT key, value FROM settings")
            return {row[0]: row[1] for row in cur.fetchall()}
