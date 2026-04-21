"""
outbound.py — Outbound Call Campaign System
Uses Redis + Celery for durable job queues.
Supports: appointment reminders, rescheduling campaigns, no-show follow-ups.

Architecture:
  FastAPI → Redis Queue → Celery Workers → Twilio/Plivo → Voice AI Agent
                                                ↓
                                          OutboundCallAgent (handles rejection/reschedule)
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import redis.asyncio as aioredis
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")


# ─── Celery App ───────────────────────────────────────────────────────────────

celery_app = Celery(
    "clinical_outbound",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_acks_late=True,           # Re-queue on worker crash
    worker_prefetch_multiplier=1,  # One task at a time per worker (fair distribution)
    task_reject_on_worker_lost=True,
    task_default_retry_delay=300,  # 5 min retry delay
    task_max_retries=3,
)

# ─── Scheduled Beat Tasks ─────────────────────────────────────────────────────

celery_app.conf.beat_schedule = {
    # Run reminder sweep every 5 minutes
    "send-appointment-reminders": {
        "task": "outbound.tasks.send_appointment_reminders",
        "schedule": crontab(minute="*/5"),
    },
    # Check for no-shows every 30 minutes
    "check-no-shows": {
        "task": "outbound.tasks.check_no_shows",
        "schedule": crontab(minute="*/30"),
    },
}


# ─── Campaign Types ───────────────────────────────────────────────────────────

class CampaignType(str, Enum):
    REMINDER_24H = "reminder_24h"
    REMINDER_2H = "reminder_2h"
    RESCHEDULE = "reschedule"
    NO_SHOW_FOLLOWUP = "no_show_followup"
    HEALTH_CHECKUP = "health_checkup"


class CallStatus(str, Enum):
    QUEUED = "queued"
    CALLING = "calling"
    ANSWERED = "answered"
    COMPLETED = "completed"
    REJECTED = "rejected"
    NO_ANSWER = "no_answer"
    FAILED = "failed"
    RESCHEDULED = "rescheduled"


# ─── Outbound Campaign Scheduler ─────────────────────────────────────────────

class OutboundCallScheduler:
    """
    Manages outbound call campaigns.
    Enqueues jobs into Redis for Celery workers to pick up.
    """

    CAMPAIGN_KEY_PREFIX = "campaign:"
    QUEUE_KEY = "outbound:queue"

    def __init__(self, redis: aioredis.Redis):
        self.r = redis

    async def enqueue_campaign(self, payload: dict) -> str:
        """
        Enqueue a batch outbound campaign.
        
        Payload example:
        {
            "type": "reminder_24h",
            "appointments": [
                {"appointment_id": "APT-001", "patient_id": "P123", "phone": "+91..."}
            ],
            "scheduled_at": "2025-01-15T09:00:00",
            "priority": "high"
        }
        """
        job_id = f"campaign_{uuid.uuid4().hex[:12]}"
        campaign = {
            "job_id": job_id,
            "status": "queued",
            "type": payload.get("type"),
            "appointments": payload.get("appointments", []),
            "scheduled_at": payload.get("scheduled_at"),
            "created_at": datetime.utcnow().isoformat(),
            "stats": {"total": 0, "completed": 0, "failed": 0, "rescheduled": 0},
        }

        # Store campaign metadata
        await self.r.setex(
            f"{self.CAMPAIGN_KEY_PREFIX}{job_id}",
            86400,  # 24h TTL
            json.dumps(campaign)
        )

        # Schedule via Celery
        eta = None
        if scheduled_at := payload.get("scheduled_at"):
            eta = datetime.fromisoformat(scheduled_at)

        # Dispatch individual call tasks
        for appt in payload.get("appointments", []):
            call_outbound_task.apply_async(
                kwargs={
                    "campaign_id": job_id,
                    "appointment_id": appt["appointment_id"],
                    "patient_id": appt["patient_id"],
                    "phone": appt["phone"],
                    "campaign_type": payload.get("type"),
                    "language": appt.get("language", "en"),
                },
                eta=eta,
                queue="outbound_calls",
                priority=9 if payload.get("priority") == "high" else 5,
            )

        logger.info(f"📞 Campaign {job_id} enqueued ({len(payload.get('appointments', []))} calls)")
        return job_id

    async def schedule_reminder(self, appointment_id: str, patient_id: str, phone: str,
                                 appointment_time: datetime, language: str = "en"):
        """Auto-schedule 24h and 2h reminders for a new appointment."""
        remind_24h = appointment_time - timedelta(hours=24)
        remind_2h = appointment_time - timedelta(hours=2)

        for campaign_type, eta in [
            (CampaignType.REMINDER_24H, remind_24h),
            (CampaignType.REMINDER_2H, remind_2h),
        ]:
            if eta > datetime.utcnow():
                call_outbound_task.apply_async(
                    kwargs={
                        "campaign_id": f"auto_{appointment_id}",
                        "appointment_id": appointment_id,
                        "patient_id": patient_id,
                        "phone": phone,
                        "campaign_type": campaign_type,
                        "language": language,
                    },
                    eta=eta,
                    queue="outbound_calls",
                )
                logger.info(f"⏰ {campaign_type} reminder scheduled for {appointment_id} at {eta}")

    async def get_campaign_status(self, job_id: str) -> Optional[dict]:
        raw = await self.r.get(f"{self.CAMPAIGN_KEY_PREFIX}{job_id}")
        return json.loads(raw) if raw else None


# ─── Celery Tasks ─────────────────────────────────────────────────────────────

@celery_app.task(
    name="outbound.tasks.call_outbound",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="outbound_calls",
)
def call_outbound_task(
    self,
    campaign_id: str,
    appointment_id: str,
    patient_id: str,
    phone: str,
    campaign_type: str,
    language: str = "en",
):
    """
    Celery task that initiates an outbound call via Twilio.
    On answer: launches the voice AI agent to handle the conversation.
    """
    import asyncio
    try:
        logger.info(f"📞 Initiating outbound call: {patient_id} @ {phone}")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _make_outbound_call(campaign_id, appointment_id, patient_id, phone, campaign_type, language)
        )
        loop.close()
        return result
    except Exception as exc:
        logger.exception(f"Outbound call failed for {appointment_id}: {exc}")
        raise self.retry(exc=exc)


async def _make_outbound_call(
    campaign_id: str,
    appointment_id: str,
    patient_id: str,
    phone: str,
    campaign_type: str,
    language: str,
) -> dict:
    """
    Initiate Twilio outbound call.
    TwiML webhook points back to our WebSocket voice agent.
    """
    from twilio.rest import Client as TwilioClient
    
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # Generate a unique session for this outbound call
    call_session_id = f"out_{appointment_id}_{uuid.uuid4().hex[:8]}"

    # TwiML: Connect to our WebSocket voice AI
    twiml_url = (
        f"https://api.yourclinic.com/twilio/outbound-voice"
        f"?session_id={call_session_id}"
        f"&patient_id={patient_id}"
        f"&appointment_id={appointment_id}"
        f"&campaign_type={campaign_type}"
        f"&language={language}"
    )

    call = client.calls.create(
        to=phone,
        from_=TWILIO_PHONE_NUMBER,
        url=twiml_url,
        method="GET",
        status_callback=f"https://api.yourclinic.com/twilio/status/{call_session_id}",
        timeout=30,
        machine_detection="Enable",          # Detect answering machine
        machine_detection_timeout=3000,
    )

    logger.info(f"📞 Call initiated: SID={call.sid} session={call_session_id}")
    return {"call_sid": call.sid, "session_id": call_session_id, "status": "initiated"}


# ─── Outbound Call Flow Scripts ───────────────────────────────────────────────

OUTBOUND_PROMPTS = {
    CampaignType.REMINDER_24H: {
        "en": (
            "You are Priya from {clinic_name}. You are calling to remind the patient "
            "about their appointment tomorrow with {doctor_name} at {time}. "
            "Confirm attendance. If they cannot come, offer to reschedule. "
            "If no answer in 3 seconds, say 'Hello, can you hear me?' then continue. "
            "Keep the conversation under 2 minutes. Be warm and concise."
        ),
        "hi": (
            "आप {clinic_name} से प्रिया बोल रही हैं। मरीज को कल {doctor_name} के साथ "
            "{time} पर अपॉइंटमेंट की याद दिलाने के लिए कॉल कर रही हैं।"
        ),
        "ta": (
            "நீங்கள் {clinic_name} இலிருந்து பிரியா பேசுகிறீர்கள். "
            "நாளை {doctor_name} அவர்களுடன் {time} மணிக்கு சந்திப்பை நினைவூட்டுகிறீர்கள்."
        ),
    },
    CampaignType.NO_SHOW_FOLLOWUP: {
        "en": (
            "You are Priya from {clinic_name}. The patient missed their appointment today. "
            "Express concern, ask if they are okay, and offer to reschedule. "
            "Be empathetic, not accusatory."
        ),
    },
}


def get_outbound_prompt(campaign_type: str, language: str, context: dict) -> str:
    """Build outbound agent system prompt based on campaign type and language."""
    prompts = OUTBOUND_PROMPTS.get(campaign_type, {})
    template = prompts.get(language, prompts.get("en", ""))
    return template.format(**context)


# ─── Scheduled Tasks ─────────────────────────────────────────────────────────

@celery_app.task(name="outbound.tasks.send_appointment_reminders")
def send_appointment_reminders():
    """
    Sweep DB for appointments in next 24h and 2h windows, dispatch reminders.
    Runs every 5 minutes via celery beat.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_sweep_and_remind())
    loop.close()


async def _sweep_and_remind():
    """Query appointments needing reminders and enqueue calls."""
    # Production: DB query
    # SELECT * FROM appointments
    # WHERE status = 'confirmed'
    # AND start_time BETWEEN NOW() + INTERVAL '23 hours 50 minutes'
    #                    AND NOW() + INTERVAL '24 hours 10 minutes'
    # AND reminder_24h_sent = false
    logger.info("🔍 Sweeping for appointment reminders...")


@celery_app.task(name="outbound.tasks.check_no_shows")
def check_no_shows():
    """Mark appointments as no_show if 30+ minutes past start with no check-in."""
    logger.info("🔍 Checking for no-shows...")
    # Production: UPDATE appointments SET status='no_show'
    # WHERE status='confirmed' AND start_time < NOW() - INTERVAL '30 minutes'
    # AND check_in_time IS NULL
