from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import whisperx
from fastapi import HTTPException

from models.load import get_asr_model, ALIGN_MODEL, ALIGN_META, DEVICE, get_align_model
from utils.logger import logger


# def transcribe_channel(path: str) -> List[Dict]:
#     result = ASR_MODEL.transcribe(path)
#     audio_np = whisperx.load_audio(path)  # load audio as np.ndarray for alignment
#     logger.info("Type: {}, Shape: {}", type(audio_np), audio_np.shape if hasattr(audio_np, "shape") else None)
#     aligned = whisperx.align(result["segments"], ALIGN_MODEL, ALIGN_META, audio_np, device=DEVICE)
#     return aligned["word_segments"]

def transcribe_channel(
        path: str,
        language: Optional[str] = "en",
        model: Optional[str] = "large-v3",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        timestamp_granularity: str = "segment",
        needs_alignment: bool = True
) -> list:
    # Ensure model is up to date
    asr_model = get_asr_model(model)

    # Pass prompt, temperature etc. to transcription call if supported
    result = asr_model.transcribe(
        path,
        language=language,
        # prompt=prompt,
        # temperature=temperature,
        # timestamp_granularity=timestamp_granularity,
    )

    if needs_alignment:
        try:
            align_model, align_meta = get_align_model(language)
            audio_np = whisperx.load_audio(path)
            aligned_result = whisperx.align_result(
                result["segments"], align_model, align_meta, audio_np, device=align_model.device
            )
            # Choose correct granularity for output
            if timestamp_granularity == "word":
                final_result = aligned_result["word_segments"]
            else:
                final_result = aligned_result["segments"]
            return final_result
        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
    # Return raw transcription segments or words per the asr model output format
    if timestamp_granularity == "word":
        return result.get("word_segments", [])
    else:
        return result.get("segments", [])

def parallel_transcribe(paths, needs_alignment=True, language="pl"):
    results = []
    with ThreadPoolExecutor(max_workers=len(paths)) as executor:
        futures = [executor.submit(transcribe_channel, path, needs_alignment, language) for path in paths]
        for future in futures:
            results.append(future.result())
    return results