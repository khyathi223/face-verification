from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = UPLOAD_DIR / "processed"
CROPS_DIR = UPLOAD_DIR / "crops"
DATABASE_DIR = BASE_DIR / "database"
FRONTEND_DIR = BASE_DIR / "frontend"

DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'face_verification.db'}"

MAX_IMAGE_SIZE = 640
MATCH_THRESHOLD = 0.42
DETECTION_CONFIDENCE_THRESHOLD = 0.5

for folder in (UPLOAD_DIR, PROCESSED_DIR, CROPS_DIR, DATABASE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

