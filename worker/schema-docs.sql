CREATE TABLE IF NOT EXISTS uploads (
  name TEXT PRIMARY KEY,
  chunks INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
