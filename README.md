# AI Face Verification Web Application

Full-stack face detection, counting, registration, and verification app built with FastAPI, OpenCV, SQLite, and InsightFace ArcFace embeddings.

## Features

- Upload JPG, JPEG, or PNG images.
- Resize large images while preserving aspect ratio.
- Detect only human faces and ignore non-face objects.
- Count faces and draw bounding boxes.
- Register users with name, user ID, and face image.
- Verify uploaded or webcam faces against registered embeddings.
- Save processed images, face crops, and detection logs.
- Download processed images from the dashboard.

## Project Structure

```text
project/
├── backend/
├── frontend/
├── models/
├── uploads/
├── database/
├── requirements.txt
└── main.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open the app at:

```text
http://127.0.0.1:8000
```

## API Endpoints

- `POST /register` with `name`, `user_id`, and `image`
- `POST /verify` with `image`
- `POST /detect` with `image`
- `GET /users`
- `GET /health`

## Model Notes

The backend uses InsightFace `buffalo_l`, which provides RetinaFace-style detection and ArcFace embeddings. On first run, InsightFace may download model weights. If InsightFace is unavailable, the app starts with an OpenCV fallback detector so local development still works, but production verification should use the InsightFace mode shown by `/health`.

## Database

SQLite is used by default at `database/face_verification.db`. The schema is also documented in `database/schema.sql`.

## Deployment Notes

For local use, run with Uvicorn. For deployment, package the FastAPI backend in Docker or deploy it to Render. The static frontend is served by FastAPI, so Vercel is optional only if you later split the frontend into a separate React app.

