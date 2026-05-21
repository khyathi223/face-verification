from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATABASE_URL


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    endpoint: Mapped[str] = mapped_column(String(40), nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=True)
    processed_path: Mapped[str] = mapped_column(Text, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[str] = mapped_column(Text, nullable=False)
    best_score: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

