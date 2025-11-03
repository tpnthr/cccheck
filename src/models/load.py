import json
import threading
from pathlib import Path

import whisperx
from optuna.terminator.improvement.emmr import torch

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Model globals
_model_lock = threading.Lock()
_current_model_name = None
_current_model_instance = None

MAX_PAUSE = 1.5  # Max Pause of the fraze
# Preload align model
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


_align_models = {}
_align_metas = {}
_align_lock = threading.Lock()

ALIGN_MODEL_MAP = {
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    # add more language:model entries as needed
}


def get_align_model(language_code: str):
    """
    Return the alignment model and meta for the given language.
    Loads and caches the model only if not already present.
    If language_code is in ALIGN_MODEL_MAP, use your mapping.
    Otherwise, let whisperx decide the alignment model.
    """
    # Use language_code as cache key (can enhance with more params if needed)
    with _align_lock:
        if language_code in _align_models:
            return _align_models[language_code], _align_metas[language_code]

        if language_code in ALIGN_MODEL_MAP:
            align_model, align_meta = whisperx.load_align_model(
                language_code=language_code,
                device=DEVICE,
                model_name=ALIGN_MODEL_MAP[language_code]
            )
        else:
            align_model, align_meta = whisperx.load_align_model(
                language_code=language_code,
                device=DEVICE
            )
        _align_models[language_code] = align_model
        _align_metas[language_code] = align_meta
        return align_model, align_meta
