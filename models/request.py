import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class RequestRecord(Base):
    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda:str(uuid.uuid4))
    prompt_hash: Mapped[str] = mapped_column(String(64),nullable=False,index=True)
    keywords: Mapped[str] = mapped_column(Text, nullable=False, default="[]") #json list

    task_type: Mapped[str] = mapped_column(String(32), nullable=False,index=True)
    routing_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    actual_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_tokens_set: Mapped[int] = mapped_column(Integer, nullable=False)

    model_used: Mapped[str] = mapped_column(String(64),nullable=False)
    provider_used: Mapped[str] = mapped_column(String(64),nullable=False, index=True)
    
    was_fallback: Mapped[bool] = mapped_column(Boolean,nullable=False, default=False)
    fallback_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    was_upgraded: Mapped[bool] = mapped_column(Boolean,nullable=False,default=False)

    response_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    was_misclassified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)



def prompt_hash(prompt:str)-> str:
    return hashlib.sha256(prompt.encode()).hexdigest()