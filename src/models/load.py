import whisperx
from optuna.terminator.improvement.emmr import torch
from pathlib import Path
import json
import threading

# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Model globals
_model_lock = threading.Lock()
_current_model_name = None
_current_model_instance = None

# Preload align model (unchanged)
ALIGN_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(language_code="pl", device=DEVICE, model_name=ALIGN_MODEL_NAME)

def get_asr_model(model_name: str):
    """
    Returns the ASR model instance. Loads model if different from current.
    Thread-safe.
    """
    global _current_model_name, _current_model_instance

    with _model_lock:
        if _current_model_instance is None or _current_model_name != model_name:
            # Load and cache new model instance
            _current_model_instance = whisperx.load_model(
                model_name,
                device=DEVICE,
                device_id=None,
                compute_type="float16" if DEVICE == "cuda" else "float32"
            )
            _current_model_name = model_name
    return _current_model_instance