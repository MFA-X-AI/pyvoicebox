"""V_SAPISYNTH - SAPI speech synthesis (stub).

This wraps Microsoft SAPI which is Windows-specific.
"""

from __future__ import annotations

def v_sapisynth(*args, **kwargs) -> None:
    """Synthesize speech using Microsoft SAPI.

    This is a Windows-specific function that interfaces with Microsoft SAPI.
    For cross-platform text-to-speech, consider using pyttsx3 or gTTS.

    Raises
    ------
    NotImplementedError
        SAPI is Windows-specific.
    """
    raise NotImplementedError(
        "v_sapisynth is Windows/SAPI-specific. "
        "Consider using pyttsx3 or gTTS for cross-platform TTS."
    )
