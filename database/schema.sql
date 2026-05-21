CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(80) NOT NULL UNIQUE,
    name VARCHAR(160) NOT NULL,
    embedding BLOB NOT NULL,
    image_path TEXT NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint VARCHAR(40) NOT NULL,
    image_path TEXT,
    processed_path TEXT,
    face_count INTEGER DEFAULT 0,
    result TEXT NOT NULL,
    best_score FLOAT,
    created_at DATETIME NOT NULL
);

