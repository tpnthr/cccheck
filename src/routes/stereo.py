import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from models.transcribe import transcribe_channel
from schemas.transcribe import TranscribeRequest
from utils.file import save_transcription_text, DATA_TEMP_DIR, prepare_audio_input
from utils.format import group_words, render_stereo_dialogue_lines
from utils.sound import split_stereo
from loguru import logger

router = APIRouter()

@router.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.fileUrl)
    tmp_files = []
    try:
        left_path, right_path = split_stereo(audio_file)
        left_words = transcribe_channel(
            left_path,
            language=req.language,
            model=req.model,
            prompt=req.prompt,
            temperature=req.temperature,
            timestamp_granularity=req.timestamp_granularity,
            needs_alignment=True
        )
        right_words = transcribe_channel(
            right_path,
            language=req.language,
            model=req.model,
            prompt=req.prompt,
            temperature=req.temperature,
            timestamp_granularity=req.timestamp_granularity,
            needs_alignment=True
        )

        if req.label_speakers:
            for w in left_words:
                w["speaker"] = "client"
            for w in right_words:
                w["speaker"] = "agent"
        else:
            for w in left_words:
                w["speaker"] = "speaker1"
            for w in right_words:
                w["speaker"] = "speaker2"

        all_words = left_words + right_words
        grouped_dialogue = group_words(all_words)
        dialog_lines = render_stereo_dialogue_lines(grouped_dialogue)
        dialog_text = "\n".join(dialog_lines)
        output_path = save_transcription_text(dialog_text, audio_file)

        if req.responseFormat == "text":
            return dialog_text
        else:
            return {
                "success": True,
                "json": grouped_dialogue,
                "dialog": dialog_text,
                "transcript_file": output_path
            }

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        for f in tmp_files:
            try:
                pathlib.Path(f).unlink()
            except FileNotFoundError:
                logger.warning(f"File not found during cleanup: {f}")