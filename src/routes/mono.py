from fastapi import APIRouter, HTTPException
from models.transcribe import transcribe_channel
from schemas.transcribe import TranscribeRequest
from utils.file import prepare_audio_input, save_transcription_text
from loguru import logger
from utils.format import render_mono_dialogue_lines

router = APIRouter()

@router.post("/transcribe")
def transcribe(req: TranscribeRequest):
    audio_file = prepare_audio_input(req.fileUrl)
    try:
        words = transcribe_channel(
            audio_file,
            language=req.language.lower() if req.language else None,
            model=req.model,
            prompt=req.prompt,
            temperature=req.temperature,
            timestamp_granularity=req.timestamp_granularity,
            needs_alignment=True
        )
        transcript_text = " ".join([w["word"] for w in words])
        dialog_lines = render_mono_dialogue_lines(words)
        dialog_text = "\n".join(dialog_lines)
        output_path = save_transcription_text(dialog_text, audio_file)

        if req.responseFormat == "text":
            return dialog_text
        else:
            return {
                "success": True,
                "json": words,
                "dialog": dialog_text,
                "transcription": transcript_text,
                "transcript_file": output_path
            }
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))