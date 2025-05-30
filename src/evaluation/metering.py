from typing import Dict

from pydantic import BaseModel, Field


class MeteringItem(BaseModel):
    requests_cnt: int = 0
    messages_sent_cnt: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


class Metering(BaseModel):
    dataset_compose: Dict[str, MeteringItem] = Field(default_factory=dict)
    stage1: Dict[str, MeteringItem] = Field(default_factory=dict)
    stage2: Dict[str, MeteringItem] = Field(default_factory=dict)
    stage3: Dict[str, MeteringItem] = Field(default_factory=dict)
    stage4: Dict[str, MeteringItem] = Field(default_factory=dict)
