from pydantic import BaseModel
from typing import Literal

class EvaluatorResult(BaseModel):
    next_node: Literal["extract", "create_xml", "end"]
    reason: str

