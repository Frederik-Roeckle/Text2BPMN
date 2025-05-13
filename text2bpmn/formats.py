from pydantic import BaseModel
from typing import Literal

class EvaluatorResult(BaseModel):
    next_node: Literal["xml_Agent", "end"]
    reason: str