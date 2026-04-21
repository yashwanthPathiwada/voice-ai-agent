"""
main.py — FastAPI WebSocket Backend
Real-Time Multilingual Voice AI Agent for Clinical Appointment Booking

Pipeline: Audio → STT (Deepgram) → LLM Agent → TTS (ElevenLabs) → Audio
Latency target: <450ms from speech-end to first audio chunk
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

import asyncio
import json
import logging
import time
from fastapi.responses import JSONResponse
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import ClinicalAgent
from memory import MemoryManager
from stt import DeepgramSTTClient
from tts import ElevenLabsTTSClient
from outbound import OutboundCallScheduler
from models import SessionState, AppointmentRequest
import os
from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
import random

responses = [
    "Hello! How can I help you?",
    "I am here, tell me what you need.",
    "Nice to hear from you!",
    "Ask me anything!",
]

response_text = random.choice(responses)

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Lifespan: boot shared resources ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Redis pool — used for session memory + job queues
    app.state.redis = await aioredis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )
    app.state.memory = MemoryManager(app.state.redis)
    app.state.outbound = OutboundCallScheduler(app.state.redis)
    logger.info("✅ Redis connected, memory manager ready")

    yield  # app runs here

    await app.state.redis.close()
    logger.info("🔴 Redis disconnected")


app = FastAPI(
    title="Clinical Voice AI Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── WebSocket: Real-Time Voice Pipeline ────────────────────────────────────

@app.websocket("/ws/voice/{session_id}")
async def voice_pipeline(websocket: WebSocket, session_id: str):
    """
    Simple working WebSocket voice demo.

    Client sends:
      { "type": "audio_chunk", "data": "<base64>" }
      { "type": "audio_end" }
      { "type": "barge_in" }
      { "type": "language_hint", "lang": "hi" }

    Server sends:
      { "type": "transcript", "text": "..." }
      { "type": "agent_thinking", "trace": "..." }
      { "type": "tts_end" }
      { "type": "error", "message": "..." }
    """
    await websocket.accept()
    logger.info(f"🔌 Session {session_id} connected")

    try:
        while True:
            raw = await websocket.receive_text()
            logger.info(f"📩 Raw message from {session_id}: {raw}")

            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "audio_chunk":
                await websocket.send_json({
                    "type": "transcript",
                    "text": "Receiving audio..."
                })

            elif mtype == "audio_end":
                await websocket.send_json({
                    "type": "agent_thinking",
                    "trace": "Generating AI response..."
                })

                # 🔥 Simple working response (NO Gemini)
                response_text = "Your appointment with Dr Sharma is booked for tomorrow at 10 AM."

                await websocket.send_json({
                    "type": "transcript",
                    "text": response_text
                })

                await websocket.send_json({
                    "type": "tts_end"
                })
                            

            elif mtype == "barge_in":
                await websocket.send_json({
                    "type": "agent_thinking",
                    "trace": "User interrupted. Stopping current response."
                })

                await websocket.send_json({
                    "type": "tts_end"
                })

            elif mtype == "language_hint":
                lang = msg.get("lang", "en")
                await websocket.send_json({
                    "type": "agent_thinking",
                    "trace": f"Language switched to {lang}"
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {mtype}"
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 Session {session_id} disconnected")

    except Exception as e:
        logger.exception(f"Pipeline error in {session_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass
# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "clinical-voice-ai"}


@app.post("/outbound/campaign")
async def create_campaign(payload: dict):
    """Enqueue an outbound reminder/reschedule campaign."""
    scheduler: OutboundCallScheduler = app.state.outbound
    job_id = await scheduler.enqueue_campaign(payload)
    return {"job_id": job_id, "status": "queued"}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    try:
        memory = app.state.memory
        history = await memory.get_conversation_history(session_id)
        return JSONResponse(content=history)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "session_id": session_id}
        )



@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    try:
        memory = app.state.memory
        await memory.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        return {"error": str(e)}