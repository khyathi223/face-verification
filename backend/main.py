from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .config import FRONTEND_DIR, MATCH_THRESHOLD, UPLOAD_DIR
from .database import DetectionLog, User, get_db, init_db
from .face_service import DetectedFace, face_service
from .schemas import FaceBox, ProcessResponse, UserOut

app = FastAPI(
    title="AI Face Verification API",
    description="Face detection, counting, registration, and verification using FastAPI and ArcFace embeddings.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": face_service.mode, "match_threshold": MATCH_THRESHOLD}


@app.get("/users", response_model=list[UserOut])
def users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at.desc()).all()


@app.post("/register")
async def register(
    name: str = Form(...),
    user_id: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _validate_image_type(image)
    existing = db.query(User).filter(User.user_id == user_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="A user with this User ID already exists.")

    content = await image.read()
    raw_path = _save_upload(content, image.filename)
    frame = _decode(content)
    faces = face_service.detect_faces(frame)
    if not faces:
        _log(db, "register", str(raw_path), None, 0, "No human face detected.", None)
        raise HTTPException(status_code=400, detail="No human face detected.")
    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Please register with exactly one face in the image.")

    processed = face_service.draw_faces(frame, faces)
    user = User(
        user_id=user_id.strip(),
        name=name.strip(),
        embedding=faces[0].embedding.astype("float32").tobytes(),
        image_path=str(raw_path),
    )
    db.add(user)
    _log(db, "register", str(raw_path), str(processed), 1, "Registered successfully.", None)
    db.commit()
    return {
        "message": "Registered successfully.",
        "user_id": user.user_id,
        "name": user.name,
        "processed_image_url": _url(processed),
    }


@app.post("/detect", response_model=ProcessResponse)
async def detect(image: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_image_type(image)
    content = await image.read()
    raw_path = _save_upload(content, image.filename)
    frame = _decode(content)
    faces = face_service.detect_faces(frame)
    processed = face_service.draw_faces(frame, faces)
    message = f"Number of faces detected: {len(faces)}" if faces else "No human face detected."
    _log(db, "detect", str(raw_path), str(processed), len(faces), message, None)
    db.commit()
    return _response(message, faces, processed)


@app.post("/verify", response_model=ProcessResponse)
async def verify(image: UploadFile = File(...), db: Session = Depends(get_db)):
    _validate_image_type(image)
    content = await image.read()
    raw_path = _save_upload(content, image.filename)
    frame = _decode(content)
    faces = face_service.detect_faces(frame)
    if not faces:
        processed = face_service.draw_faces(frame, faces)
        message = "No human face detected."
        _log(db, "verify", str(raw_path), str(processed), 0, message, None)
        db.commit()
        return _response(message, faces, processed)

    candidates = [
        (user.user_id, user.name, np.frombuffer(user.embedding, dtype="float32"))
        for user in db.query(User).all()
    ]
    best_score = None
    for face in faces:
        match = face_service.compare(face.embedding, candidates)
        if match:
            face.matched_user_id, face.matched_name, face.score = match
            face.status = "Verified User"
        else:
            face.status = "Unknown Person"
            face.score = None
        if face.score is not None:
            best_score = face.score if best_score is None else max(best_score, face.score)

    verified_count = sum(1 for face in faces if face.status == "Verified User")
    if verified_count:
        message = f"Verified User. Number of faces detected: {len(faces)}"
    else:
        message = f"Unknown Person. Number of faces detected: {len(faces)}"

    processed = face_service.draw_faces(frame, faces)
    _log(db, "verify", str(raw_path), str(processed), len(faces), message, best_score)
    db.commit()
    return _response(message, faces, processed)


@app.post("/webcam/verify", response_model=ProcessResponse)
async def webcam_verify(frame: UploadFile = File(...), db: Session = Depends(get_db)):
    return await verify(frame, db)


def _validate_image_type(upload: UploadFile) -> None:
    allowed = {"image/jpeg", "image/jpg", "image/png"}
    if upload.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG images are supported.")


def _save_upload(content: bytes, filename: str | None) -> Path:
    suffix = Path(filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def _decode(content: bytes):
    try:
        return face_service.decode_image(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _url(path: Path | str | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    return "/" + path.relative_to(path.parents[1]).as_posix()


def _face_box(face: DetectedFace) -> FaceBox:
    x1, y1, x2, y2 = face.bbox
    return FaceBox(
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        confidence=round(face.confidence, 4),
        crop_url=_url(face.crop_path),
        status=face.status,
        matched_user_id=face.matched_user_id,
        matched_name=face.matched_name,
        score=round(face.score, 4) if face.score is not None else None,
    )


def _response(message: str, faces: list[DetectedFace], processed: Path) -> ProcessResponse:
    return ProcessResponse(
        message=message,
        face_count=len(faces),
        faces=[_face_box(face) for face in faces],
        processed_image_url=_url(processed),
        download_url=_url(processed),
    )


def _log(
    db: Session,
    endpoint: str,
    image_path: str | None,
    processed_path: str | None,
    face_count: int,
    result: str,
    best_score: float | None,
) -> None:
    db.add(
        DetectionLog(
            endpoint=endpoint,
            image_path=image_path,
            processed_path=processed_path,
            face_count=face_count,
            result=result,
            best_score=best_score,
        )
    )

