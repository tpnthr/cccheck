import json
import os
import threading
from pathlib import Path

import whisperx
import torch

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

USE_CPU = os.getenv("USE_CPU") == "1"
DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Force an explicit public model (override via env)
ALIGN_MODEL_NAME = os.getenv("ALIGN_MODEL_NAME", "facebook/wav2vec2-large-xlsr-53").strip()

MODEL_LOCK = threading.Lock()
_current_model_name = None
_current_model_instance = None

# Explicit align model to avoid private/removed repos
ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(
    language_code="pl",
    device=DEVICE,
    model_name=ALIGN_MODEL_NAME,
)

def get_asr_model(model_name: str):
    global _current_model_name, _current_model_instance
    with MODEL_LOCK:
        if _current_model_instance is None or _current_model_name != model_name:
            _current_model_instance = whisperx.load_model(
                model_name,
                device=DEVICE,
                compute_type="float16" if DEVICE == "cuda" else "float32",
            )
            _current_model_name = model_name
        return _current_model_instance
