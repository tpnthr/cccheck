from typing import Optional, List

from pydantic import BaseModel

class TranscribeRequest(BaseModel):
    fileUrl: str  # required
    language: Optional[str] = None
    model: Optional[str] = "current_model"  # default your current model name here
    prompt: Optional[str] = None
    responseFormat: Optional[str] = "json"  # "json" or "text"
    temperature: Optional[float] = 0.0
    timestamp_granularity: Optional[str] = "segment"  # "segment" or "word"
    need_alignment: Optional[bool] = True  # presumably already present
    label_speakers: Optional[bool] = False  # for stereo endpoint, presumably


class BatchRequest(BaseModel):
    inputs: List[str]
    language: Optional[str] = None
    need_alignment: Optional[bool] = None
    return_srt: Optional[bool] = False
    return_vtt: Optional[bool] = False
    label_speakers: Optional[bool] = False