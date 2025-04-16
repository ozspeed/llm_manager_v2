"""
search_history.py

Module to manage recent Hugging Face search queries and trending/cached model queries for the LLM Model Manager.
Uses SQLite for persistence, storing recent searches and optionally trending search statistics.
"""
"""
search_history.py

Manages recent Hugging Face search queries for the LLM Model Manager.
- Persists recent searches (query, model_type) in SQLite with FIFO max limit.
- Supports trending search as a special persistent tag.
- get_popular_models() returns trending first, then recent user searches.
- All searches are persistent; results can be cached in memory.
- Settings for max history and trending model_type are loaded from config.
"""
import sqlite3
import os
import threading
from typing import List, Dict, Optional

from models import config

_DB_PATH = os.environ.get('SEARCH_HISTORY_DB', 'search_history.db')
_DB_LOCK = threading.Lock()

# --- Settings Integration ---
def get_max_history_limit() -> int:
    """Get max number of recent searches to keep (FIFO). Defaults to 20 if not set."""
    try:
        return int(config.get_setting('frameworks.huggingface.max_search_history') or 20)
    except Exception:
        return 20

def get_trending_model_type() -> str:
    """Get model_type for trending search. Defaults to 'gguf' if not set."""
    return config.get_setting('frameworks.huggingface.trending_model_type') or 'gguf'

# --- DB Setup ---
def _init_db():
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS recent_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                model_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

_init_db()

# --- Core Functions ---
def add_search_query(query: str, model_type: Optional[str] = None, is_persistent: bool = True) -> None:
    """
    Add a search query to history. Always persistent (is_persistent for compatibility).
    Enforces FIFO max limit (oldest removed if over limit).
    """
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO recent_searches (query, model_type) VALUES (?, ?)
        ''', (query, model_type))
        # Enforce FIFO max limit
        max_limit = get_max_history_limit()
        c.execute('''
            DELETE FROM recent_searches WHERE id NOT IN (
                SELECT id FROM recent_searches ORDER BY timestamp DESC, id DESC LIMIT ?
            )
        ''', (max_limit,))
        conn.commit()
        conn.close()

def get_recent_searches(limit: Optional[int] = None) -> List[Dict]:
    """
    Get the most recent N search queries (default: max history limit).
    Returns list of dicts with query, model_type, timestamp.
    """
    if limit is None:
        limit = get_max_history_limit()
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT query, model_type, timestamp FROM recent_searches
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
    results = []
    for row in rows:
        query, model_type, timestamp = row
        results.append({
            'query': query,
            'model_type': model_type,
            'timestamp': timestamp
        })
    return results

def get_popular_models() -> List[Dict]:
    """
    Returns the trending search as the first tag, then most recent user searches (deduped, trending always first).
    """
    trending = {
        'query': 'trending',
        'model_type': get_trending_model_type(),
        'timestamp': None
    }
    seen = set()
    results = [trending]
    seen.add(('trending', trending['model_type']))
    for s in get_recent_searches():
        key = (s['query'], s['model_type'])
        if key not in seen:
            results.append(s)
            seen.add(key)
    return results

def clear_search_history():
    """
    Clear all search history (admin/testing).
    """
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM recent_searches')
        conn.commit()
        conn.close()
