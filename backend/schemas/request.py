from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Purpose = Literal["힐링", "맛집", "문화관광", "쇼핑", "자연/액티비티", "종합"]
Companion = Literal["혼자", "커플", "친구", "가족", "기타"]
MAX_TRIP_DAYS = 7


class RecommendRequest(BaseModel):
    start_date: date
    end_date: date
    companion: Companion
    purpose: Purpose
    lodging_ids: list[str] = Field(min_length=1)
    intensity: int = Field(ge=1, le=5)

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @field_validator("lodging_ids", mode="before")
    @classmethod
    def coerce_lodging_ids(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(item) for item in value]
        return value

    @model_validator(mode="after")
    def validate_trip(self) -> "RecommendRequest":
        if self.end_date < self.start_date:
            raise ValueError("종료일은 시작일 이후여야 합니다.")
        days = (self.end_date - self.start_date).days + 1
        if days > MAX_TRIP_DAYS:
            raise ValueError(f"여행 기간은 최대 {MAX_TRIP_DAYS}일입니다.")
        if len(self.lodging_ids) not in (1, days):
            raise ValueError("숙소는 1곳(전 기간 동일)이거나 일차 수와 같아야 합니다.")
        return self
