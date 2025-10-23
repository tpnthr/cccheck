import json
import pathlib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
APP_NAME = config.get("app_name", "speech2text")
VERSION = config.get("version", "0.0.1")
ALLOW_SHUTDOWN = config.get("allow_shutdown", False)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

DATA_INPUT_DIR = pathlib.Path("data/input")
DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR = pathlib.Path("data/output")
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_TEMP_DIR = pathlib.Path("data/temp")
DATA_TEMP_DIR.mkdir(parents=True, exist_ok=True)


HF_TOKEN = "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"  # for gated alignment if needed
