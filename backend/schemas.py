from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserOut(BaseModel):
    user_id: str
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float
    crop_url: Optional[str] = None
    status: Optional[str] = None
    matched_user_id: Optional[str] = None
    matched_name: Optional[str] = None
    score: Optional[float] = None


class ProcessResponse(BaseModel):
    message: str
    face_count: int
    faces: list[FaceBox]
    processed_image_url: Optional[str] = None
    download_url: Optional[str] = None

