"""
memory.py — 2-Level Memory System
  Level 1: Session Memory  → Redis (TTL: 30min, extends on activity)
  Level 2: Long-Term Memory → Redis (TTL: 90 days) + optional Vector DB
  
Retrieval → Prompt injection for agent context awareness
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import uuid

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ─── Redis Key Schema ─────────────────────────────────────────────────────────
"""
session:{session_id}          → SessionState JSON         TTL: 1800s (30min)
patient:{patient_id}:prefs    → PatientPreferences JSON   TTL: 7776000s (90 days)
patient:{patient_id}:history  → list of appt summaries    TTL: 7776000s (90 days)
lang:{session_id}             → language string           TTL: 1800s
"""

SESSION_TTL = 1800          # 30 minutes
LONGTERM_TTL = 7_776_000    # 90 days


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """In-session ephemeral state — wiped when session expires."""
    session_id: str
    patient_id: Optional[str] = None
    language: str = "en"
    intent: Optional[str] = None          # book | reschedule | cancel | query
    pending_confirmation: Optional[dict] = None  # action awaiting user yes/no
    conversation_state: str = "greeting"  # greeting | collecting | confirming | done
    history: list[dict] = field(default_factory=list)  # [{role, content}]
    context: dict = field(default_factory=dict)         # scratch pad for agent
    last_active: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PatientPreferences:
    """Persistent long-term patient data."""
    patient_id: str
    name: Optional[str] = None
    phone: Optional[str] = None
    language_preference: str = "en"
    preferred_doctor: Optional[str] = None
    preferred_time_of_day: str = "morning"
    appointment_type: str = "in_person"
    last_seen: Optional[str] = None
    total_appointments: int = 0
    no_show_count: int = 0


# ─── Memory Manager ───────────────────────────────────────────────────────────

class MemoryManager:

    def __init__(self, redis: aioredis.Redis):
        self.r = redis

    # ── Session Memory ────────────────────────────────────────────────────────

    async def get_or_create_session(self, session_id: str) -> SessionState:
        key = f"session:{session_id}"
        raw = await self.r.get(key)
        if raw:
            data = json.loads(raw)
            session = SessionState(**data)
            logger.debug(f"🔄 Loaded session {session_id}")
        else:
            session = SessionState(session_id=session_id)
            await self.save_session_state(session_id, session)
            logger.info(f"🆕 Created session {session_id}")

        # Refresh TTL on access
        await self.r.expire(key, SESSION_TTL)
        return session

    async def save_session_state(self, session_id: str, session: SessionState):
        key = f"session:{session_id}"
        session.last_active = datetime.utcnow().isoformat()
        await self.r.setex(key, SESSION_TTL, json.dumps(asdict(session)))

    async def save_session(self, session: SessionState):
        await self.save_session_state(session.session_id, session)

    async def clear_session(self, session_id: str):
        await self.r.delete(f"session:{session_id}", f"lang:{session_id}")

    async def update_language(self, session_id: str, language: str):
        """Update language in both session and long-term memory."""
        # Session
        session = await self.get_or_create_session(session_id)
        session.language = language
        await self.save_session_state(session_id, session)

        # Long-term (if patient known)
        if session.patient_id:
            await self.update_patient_preference(
                session.patient_id,
                {"language_preference": language}
            )

    async def get_conversation_history(self, session_id: str) -> list[dict]:
        session = await self.get_or_create_session(session_id)
        return session.history

    # ── Long-Term Memory ──────────────────────────────────────────────────────

    async def get_patient_preferences(self, patient_id: str) -> Optional[PatientPreferences]:
        key = f"patient:{patient_id}:prefs"
        raw = await self.r.get(key)
        if not raw:
            return None
        return PatientPreferences(**json.loads(raw))

    async def save_patient_preferences(self, prefs: PatientPreferences):
        key = f"patient:{prefs.patient_id}:prefs"
        await self.r.setex(key, LONGTERM_TTL, json.dumps(asdict(prefs)))

    async def update_patient_preference(self, patient_id: str, updates: dict):
        prefs = await self.get_patient_preferences(patient_id)
        if not prefs:
            prefs = PatientPreferences(patient_id=patient_id)
        for k, v in updates.items():
            if hasattr(prefs, k):
                setattr(prefs, k, v)
        await self.save_patient_preferences(prefs)

    async def record_appointment_event(
        self,
        patient_id: str,
        appointment_id: str,
        event: str,  # booked | cancelled | rescheduled | completed | no_show
        details: dict = None,
    ):
        """Append appointment event to patient's history list."""
        key = f"patient:{patient_id}:history"
        entry = {
            "appointment_id": appointment_id,
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {},
        }
        # Push to list, keep last 50
        await self.r.lpush(key, json.dumps(entry))
        await self.r.ltrim(key, 0, 49)
        await self.r.expire(key, LONGTERM_TTL)

        # Update preference counters
        if event == "no_show":
            prefs = await self.get_patient_preferences(patient_id)
            if prefs:
                prefs.no_show_count += 1
                await self.save_patient_preferences(prefs)
        elif event == "booked":
            await self.update_patient_preference(
                patient_id, {"total_appointments": None}  # increment via HINCRBY in production
            )

    async def get_patient_appointment_history(self, patient_id: str, limit: int = 10) -> list[dict]:
        key = f"patient:{patient_id}:history"
        raw_list = await self.r.lrange(key, 0, limit - 1)
        return [json.loads(r) for r in raw_list]

    # ── Prompt Context Builder ────────────────────────────────────────────────

    async def build_prompt_context(self, session_id: str) -> str:
        """
        Build a context string to inject into the system prompt.
        Pulls from session (current state) + long-term (patient prefs + history).
        """
        session = await self.get_or_create_session(session_id)
        lines = ["## Patient Context"]

        # Session-level info
        if session.patient_id:
            lines.append(f"- Patient ID: {session.patient_id}")
        if session.language != "en":
            lines.append(f"- Current language: {session.language}")
        if session.intent:
            lines.append(f"- Detected intent this session: {session.intent}")
        if session.pending_confirmation:
            lines.append(f"- AWAITING CONFIRMATION for: {json.dumps(session.pending_confirmation)}")
            lines.append("  → Ask for yes/no before proceeding with this action.")

        # Long-term patient preferences
        if session.patient_id:
            prefs = await self.get_patient_preferences(session.patient_id)
            if prefs:
                if prefs.name:
                    lines.append(f"- Patient name: {prefs.name} (use this in responses)")
                if prefs.preferred_doctor:
                    lines.append(f"- Preferred doctor: {prefs.preferred_doctor}")
                if prefs.preferred_time_of_day:
                    lines.append(f"- Prefers {prefs.preferred_time_of_day} slots")
                if prefs.no_show_count > 2:
                    lines.append(f"- ⚠️ High no-show risk patient ({prefs.no_show_count} no-shows). "
                                  "Send SMS reminder immediately after booking.")
                if prefs.language_preference != "en":
                    lines.append(f"- Language preference: {prefs.language_preference}")

            # Recent appointment history (last 3 for context)
            history = await self.get_patient_appointment_history(session.patient_id, limit=3)
            if history:
                lines.append("- Recent appointment events:")
                for h in history:
                    lines.append(f"  • [{h['event'].upper()}] {h['appointment_id']} on {h['timestamp'][:10]}")

        return "\n".join(lines)


# ─── Vector Memory (Optional / for semantic retrieval) ────────────────────────
"""
When to use Vector DB (e.g., Pinecone / Weaviate):
  - Patient has >50 appointments → semantic search over history
  - "What was the last time I saw Dr. Sharma?" → embed query → nearest neighbor
  - Storing doctor notes / patient summaries for RAG

Example (Pinecone):

from pinecone import Pinecone
from openai import AsyncOpenAI

class VectorMemory:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index("clinical-patients")
        self.embed = AsyncOpenAI()
    
    async def store(self, patient_id: str, text: str, metadata: dict):
        embedding = await self.embed.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        self.index.upsert(vectors=[{
            "id": f"{patient_id}_{uuid.uuid4().hex[:8]}",
            "values": embedding.data[0].embedding,
            "metadata": {"patient_id": patient_id, "text": text, **metadata}
        }])
    
    async def retrieve(self, patient_id: str, query: str, top_k: int = 3) -> list[str]:
        embedding = await self.embed.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        results = self.index.query(
            vector=embedding.data[0].embedding,
            filter={"patient_id": {"$eq": patient_id}},
            top_k=top_k,
            include_metadata=True
        )
        return [r.metadata["text"] for r in results.matches]
"""
