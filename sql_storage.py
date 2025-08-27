import os
import sqlite3
from typing import List, Tuple, Optional

# SQLite database path (can be overridden via environment variable)
DB_PATH = os.getenv("DB_PATH", "data.db")


def connect() -> sqlite3.Connection:
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Use WAL mode for concurrent reads/writes and better durability
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db() -> None:
    """Initialize database tables if they don't already exist."""
    conn = connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
            content TEXT NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_user(username: str, password_hash: str) -> None:
    """Insert or update a user's password hash."""
    conn = connect()
    conn.execute(
        "INSERT OR REPLACE INTO users(username, password_hash) VALUES(?, ?)",
        [username, password_hash],
    )
    conn.commit()
    conn.close()


def get_user_hash(username: str) -> Optional[str]:
    """Retrieve the stored password hash for a given username."""
    conn = connect()
    row = conn.execute("SELECT password_hash FROM users WHERE username=?", [username]).fetchone()
    conn.close()
    return row[0] if row else None


def add_message(username: str, role: str, content: str) -> None:
    """Record a message (user or assistant) in the database."""
    conn = connect()
    conn.execute(
        "INSERT INTO messages(username, role, content) VALUES(?,?,?)",
        [username, role, content],
    )
    conn.commit()
    conn.close()


def load_history(username: str) -> List[Tuple[str, str]]:
    """Load a user's conversation history from the database."""
    conn = connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE username=? ORDER BY id ASC",
        [username],
    ).fetchall()
    conn.close()
    return rows