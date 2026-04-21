"""
stt.py — Real-Time Streaming Speech-to-Text via Deepgram
tts.py — Streaming Text-to-Speech via ElevenLabs

Latency budget:
  STT: ~150ms (Deepgram streaming interim results)
  TTS: first chunk ~100ms (ElevenLabs stream)
"""

# ══════════════════════════════════════════════════════════════════════════════
# STT — Deepgram WebSocket Streaming
# ══════════════════════════════════════════════════════════════════════════════

import asyncio
import base64
import json
import logging
import os
from typing import AsyncGenerator, Callable, Awaitable

import websockets
import httpx

logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Map language codes to Deepgram model + ElevenLabs voice IDs
LANGUAGE_CONFIG = {
    "en": {
        "deepgram_model": "nova-2",
        "deepgram_language": "en-IN",   # Indian English
        "elevenlabs_voice": "21m00Tcm4TlvDq8ikWAM",  # Rachel (neutral)
        "elevenlabs_model": "eleven_turbo_v2",
    },
    "hi": {
        "deepgram_model": "nova-2",
        "deepgram_language": "hi",
        "elevenlabs_voice": "pNInz6obpgDQGcFmaJgB",  # Adam (use Hindi-capable voice)
        "elevenlabs_model": "eleven_turbo_v2",
    },
    "ta": {
        "deepgram_model": "nova-2",
        "deepgram_language": "ta",
        "elevenlabs_voice": "pNInz6obpgDQGcFmaJgB",
        "elevenlabs_model": "eleven_turbo_v2",
    },
}


class DeepgramSTTClient:
    """
    Streams raw PCM audio to Deepgram LiveTranscription.
    Fires callback on each transcript event (interim + final).
    
    Deepgram config for low latency:
      - interim_results=true  → partial transcripts every ~100ms
      - endpointing=200       → 200ms silence → finalize utterance
      - vad_events=true       → voice activity detection
      - encoding=linear16     → raw PCM 16kHz mono
    """

    DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

    def __init__(self, language: str = "en"):
        self.language = language
        self._ws = None

    def update_language(self, language: str):
        self.language = language
        # New language takes effect on next stream() call
        # For mid-session switch: reconnect with new language params

    def _build_url(self) -> str:
        cfg = LANGUAGE_CONFIG.get(self.language, LANGUAGE_CONFIG["en"])
        params = "&".join([
            f"model={cfg['deepgram_model']}",
            f"language={cfg['deepgram_language']}",
            "encoding=linear16",
            "sample_rate=16000",
            "channels=1",
            "interim_results=true",
            "endpointing=200",         # 200ms silence → end of utterance
            "vad_events=true",
            "punctuate=true",
            "smart_format=true",
            "utterance_end_ms=1000",   # Force flush after 1s silence
        ])
        return f"{self.DEEPGRAM_URL}?{params}"

    async def stream(
        self,
        audio_queue: asyncio.Queue,
        callback: Callable[[str, bool], Awaitable[None]],
    ):
        """
        Long-running task that:
        1. Opens Deepgram WebSocket
        2. Sends audio chunks from queue
        3. Fires callback on transcripts
        """
        url = self._build_url()
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                self._ws = ws
                logger.info(f"🎙️ Deepgram connected (lang={self.language})")

                # Concurrent: sender + receiver
                await asyncio.gather(
                    self._send_audio(ws, audio_queue),
                    self._receive_transcripts(ws, callback),
                )
        except Exception as e:
            logger.exception(f"Deepgram stream error: {e}")

    async def _send_audio(self, ws, audio_queue: asyncio.Queue):
        """Forward PCM chunks from queue to Deepgram WebSocket."""
        while True:
            chunk = await audio_queue.get()
            if chunk == b"__END__":
                # Send close signal to Deepgram
                await ws.send(json.dumps({"type": "CloseStream"}))
                break
            await ws.send(chunk)

    async def _receive_transcripts(
        self,
        ws,
        callback: Callable[[str, bool], Awaitable[None]],
    ):
        """Receive and dispatch Deepgram transcript events."""
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "Results":
                channel = data.get("channel", {})
                alternatives = channel.get("alternatives", [])
                if not alternatives:
                    continue

                transcript = alternatives[0].get("transcript", "").strip()
                is_final = data.get("is_final", False)

                if transcript:
                    await callback(transcript, is_final)

            elif msg_type == "UtteranceEnd":
                logger.debug("🔇 Utterance end detected by Deepgram VAD")

            elif msg_type == "Error":
                logger.error(f"Deepgram error: {data}")
                break


# ══════════════════════════════════════════════════════════════════════════════
# TTS — ElevenLabs Streaming Synthesis
# ══════════════════════════════════════════════════════════════════════════════

class ElevenLabsTTSClient:
    """
    Streams text → speech using ElevenLabs WebSocket streaming API.
    First audio chunk delivered in ~80-120ms after sending text.
    
    Sends base64-encoded PCM audio chunks back to caller.
    Supports mid-stream interruption (barge-in) via task cancellation.
    """

    STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    def __init__(self, language: str = "en"):
        self.language = language

    def update_language(self, language: str):
        self.language = language

    async def synthesize_stream(self, text: str) -> AsyncGenerator[str, None]:
        """
        Yields base64-encoded PCM chunks as they arrive from ElevenLabs.
        Caller sends each chunk over WebSocket immediately.
        """
        cfg = LANGUAGE_CONFIG.get(self.language, LANGUAGE_CONFIG["en"])
        voice_id = cfg["elevenlabs_voice"]
        model_id = cfg["elevenlabs_model"]

        url = self.STREAM_URL.format(voice_id=voice_id)
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True,
            },
            "output_format": "pcm_16000",   # 16kHz mono PCM for low latency
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield base64.b64encode(chunk).decode("utf-8")
        except asyncio.CancelledError:
            logger.info("🛑 TTS stream cancelled (barge-in)")
            return
        except httpx.HTTPError as e:
            logger.exception(f"ElevenLabs TTS error: {e}")
            return


# ══════════════════════════════════════════════════════════════════════════════
# Fallback: Google Cloud TTS (for Tamil — better coverage than ElevenLabs)
# ══════════════════════════════════════════════════════════════════════════════

class GoogleCloudTTSClient:
    """
    Fallback TTS for languages with poor ElevenLabs coverage (Tamil, etc.)
    Uses Google Cloud Text-to-Speech with WaveNet voices.
    """

    async def synthesize_stream(self, text: str, language: str = "ta") -> AsyncGenerator[str, None]:
        from google.cloud import texttospeech_v1 as tts
        
        client = tts.TextToSpeechAsyncClient()

        voice_map = {
            "ta": tts.VoiceSelectionParams(
                language_code="ta-IN",
                name="ta-IN-Standard-A",
                ssml_gender=tts.SsmlVoiceGender.FEMALE,
            ),
            "hi": tts.VoiceSelectionParams(
                language_code="hi-IN",
                name="hi-IN-Wavenet-D",
                ssml_gender=tts.SsmlVoiceGender.FEMALE,
            ),
        }

        response = await client.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=voice_map.get(language, voice_map["ta"]),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
            ),
        )

        # Google returns full audio — chunk it for streaming simulation
        audio = response.audio_content
        chunk_size = 4096
        for i in range(0, len(audio), chunk_size):
            yield base64.b64encode(audio[i:i + chunk_size]).decode("utf-8")
            await asyncio.sleep(0)  # yield control
