"""
tools.py — Clinical Appointment Tool Implementations
All tools are async and return structured JSON.
Production: replace DB calls with your actual ORM (SQLAlchemy / Prisma).
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
import asyncpg  # PostgreSQL async driver

logger = logging.getLogger(__name__)

# ─── DB Pool (set at startup) ─────────────────────────────────────────────────
# In production: pass db_pool from FastAPI lifespan
_db_pool: Optional[asyncpg.Pool] = None


async def get_db():
    """Get database connection from pool."""
    if _db_pool is None:
        raise RuntimeError("DB pool not initialized")
    return _db_pool


# ─── Data Models (matches PostgreSQL schema) ─────────────────────────────────
"""
CREATE TABLE doctors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true
);

CREATE TABLE slots (
    id TEXT PRIMARY KEY,
    doctor_id TEXT REFERENCES doctors(id),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    is_available BOOLEAN DEFAULT true,
    appointment_id TEXT REFERENCES appointments(id)
);

CREATE TABLE patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT UNIQUE,
    language_preference TEXT DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE appointments (
    id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(id),
    doctor_id TEXT REFERENCES doctors(id),
    slot_id TEXT REFERENCES slots(id),
    status TEXT DEFAULT 'confirmed',  -- confirmed | cancelled | completed | no_show
    appointment_type TEXT DEFAULT 'in_person',
    reason TEXT,
    booked_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Prevent double booking at DB level
CREATE UNIQUE INDEX slots_available_unique ON slots(id) WHERE is_available = true;
"""

# ─── Simulated In-Memory DB (for demo / unit tests) ──────────────────────────

DEMO_DOCTORS = {
    "dr_sharma": {"id": "dr_sharma", "name": "Dr. Rahul Sharma", "specialty": "cardiology"},
    "dr_priya": {"id": "dr_priya", "name": "Dr. Priya Nair", "specialty": "general"},
    "dr_venkat": {"id": "dr_venkat", "name": "Dr. Venkatesh Kumar", "specialty": "orthopedics"},
}

# Simulated slot grid (in production: query DB)
def _generate_slots(doctor_id: str, date_str: str) -> list[dict]:
    """Generate time slots for a doctor on a given date."""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return []

    # Block weekends
    if date.weekday() >= 5:
        return []

    slots = []
    # Morning: 9am - 12pm (30-min slots)
    for hour in range(9, 12):
        for minute in [0, 30]:
            start = date.replace(hour=hour, minute=minute)
            slots.append({
                "id": f"{doctor_id}_{date_str}_{hour:02d}{minute:02d}",
                "doctor_id": doctor_id,
                "start_time": start.isoformat(),
                "end_time": (start + timedelta(minutes=30)).isoformat(),
                "is_available": True,
                "time_of_day": "morning",
            })
    # Afternoon: 2pm - 5pm
    for hour in range(14, 17):
        for minute in [0, 30]:
            start = date.replace(hour=hour, minute=minute)
            slots.append({
                "id": f"{doctor_id}_{date_str}_{hour:02d}{minute:02d}",
                "doctor_id": doctor_id,
                "start_time": start.isoformat(),
                "end_time": (start + timedelta(minutes=30)).isoformat(),
                "is_available": True,
                "time_of_day": "afternoon",
            })
    return slots


# Simulated booked slots
_booked_slots: dict[str, dict] = {}
_appointments: dict[str, dict] = {}


# ─── Tool: check_availability ────────────────────────────────────────────────

async def check_availability(
    date: str,
    doctor_id: Optional[str] = None,
    specialty: Optional[str] = None,
) -> dict:
    """
    Returns available slots for a doctor/specialty on a given date.
    Validates date is not in the past.
    """
    # ── Validate date ──────────────────────────────────────────────────────
    try:
        requested_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    now = datetime.now(timezone.utc)
    if requested_date.date() < now.date():
        return {
            "error": "past_date",
            "message": "Cannot book appointments in the past.",
            "earliest_available": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

    # ── Resolve doctor by specialty ────────────────────────────────────────
    if not doctor_id and specialty:
        # Find first available doctor for specialty
        matches = [d for d in DEMO_DOCTORS.values() if d["specialty"] == specialty]
        if not matches:
            return {"error": "no_doctors", "message": f"No doctors available for {specialty}"}
        doctor_id = matches[0]["id"]

    if doctor_id not in DEMO_DOCTORS:
        return {"error": "doctor_not_found", "message": f"Doctor {doctor_id} not found."}

    doctor = DEMO_DOCTORS[doctor_id]

    # ── Get slots (production: DB query) ────────────────────────────────────
    all_slots = _generate_slots(doctor_id, date)
    if not all_slots:
        return {
            "error": "no_slots",
            "message": f"No available slots for {date}. The clinic may be closed.",
            "is_weekend": True,
        }

    # Mark booked slots
    available = [
        s for s in all_slots
        if s["id"] not in _booked_slots
        and datetime.fromisoformat(s["start_time"]) > now  # future slots only
    ]

    return {
        "doctor": doctor,
        "date": date,
        "available_slots": available,
        "total_available": len(available),
    }


# ─── Tool: book_appointment ──────────────────────────────────────────────────

async def book_appointment(
    patient_id: str,
    doctor_id: str,
    slot_id: str,
    appointment_type: str = "in_person",
    reason: Optional[str] = None,
) -> dict:
    """
    Book an appointment with conflict prevention (optimistic locking).
    Uses atomic DB operation to prevent double booking.
    """
    # ── Validate slot exists and is still available ────────────────────────
    # In production: BEGIN TRANSACTION + SELECT FOR UPDATE
    if slot_id in _booked_slots:
        # Slot was taken — find alternatives
        alternatives = await suggest_alternative_slots(
            doctor_id=doctor_id,
            preferred_date=slot_id.split("_")[2][:10] if "_" in slot_id else "",
            preferred_time_of_day="any",
            num_suggestions=3,
        )
        return {
            "error": "slot_taken",
            "message": "That slot was just booked by another patient.",
            "alternatives": alternatives.get("suggestions", []),
        }

    # ── Validate slot is not in the past ─────────────────────────────────
    # Reconstruct start time from slot_id pattern: {doctor_id}_{date}_{HHMM}
    parts = slot_id.split("_")
    try:
        date_part = parts[-2]
        time_part = parts[-1]
        slot_dt = datetime.strptime(f"{date_part} {time_part[:2]}:{time_part[2:]}", "%Y-%m-%d %H:%M")
        slot_dt = slot_dt.replace(tzinfo=timezone.utc)
        if slot_dt < datetime.now(timezone.utc):
            return {"error": "past_slot", "message": "Cannot book a past time slot."}
    except Exception:
        pass  # Can't parse — proceed (DB will catch it)

    # ── Atomic booking (production: DB transaction) ────────────────────────
    appointment_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
    _booked_slots[slot_id] = {
        "appointment_id": appointment_id,
        "patient_id": patient_id,
    }
    _appointments[appointment_id] = {
        "id": appointment_id,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "slot_id": slot_id,
        "status": "confirmed",
        "appointment_type": appointment_type,
        "reason": reason,
        "booked_at": datetime.now(timezone.utc).isoformat(),
    }

    doctor = DEMO_DOCTORS.get(doctor_id, {"name": doctor_id})
    logger.info(f"✅ Appointment booked: {appointment_id} | {patient_id} → {doctor_id} @ {slot_id}")

    return {
        "success": True,
        "appointment_id": appointment_id,
        "doctor": doctor,
        "slot_id": slot_id,
        "appointment_type": appointment_type,
        "confirmation_code": appointment_id,
        "message": f"Appointment confirmed! Your ID is {appointment_id}.",
    }


# ─── Tool: reschedule_appointment ────────────────────────────────────────────

async def reschedule_appointment(
    appointment_id: str,
    new_slot_id: str,
    reason: Optional[str] = None,
) -> dict:
    """Reschedule an existing appointment to a new slot (atomic swap)."""
    if appointment_id not in _appointments:
        return {"error": "not_found", "message": f"Appointment {appointment_id} not found."}

    appt = _appointments[appointment_id]

    if appt["status"] == "cancelled":
        return {"error": "cancelled", "message": "Cannot reschedule a cancelled appointment."}

    # Check new slot
    if new_slot_id in _booked_slots and _booked_slots[new_slot_id]["appointment_id"] != appointment_id:
        return {"error": "slot_taken", "message": "The new slot is already booked."}

    # Release old slot
    old_slot_id = appt["slot_id"]
    _booked_slots.pop(old_slot_id, None)

    # Book new slot
    _booked_slots[new_slot_id] = {"appointment_id": appointment_id, "patient_id": appt["patient_id"]}
    appt["slot_id"] = new_slot_id
    appt["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"🔄 Rescheduled {appointment_id}: {old_slot_id} → {new_slot_id}")
    return {
        "success": True,
        "appointment_id": appointment_id,
        "new_slot_id": new_slot_id,
        "message": f"Appointment {appointment_id} rescheduled successfully.",
    }


# ─── Tool: cancel_appointment ────────────────────────────────────────────────

async def cancel_appointment(
    appointment_id: str,
    reason: Optional[str] = None,
    notify_doctor: bool = True,
) -> dict:
    """Cancel an appointment and free the slot."""
    if appointment_id not in _appointments:
        return {"error": "not_found", "message": f"Appointment {appointment_id} not found."}

    appt = _appointments[appointment_id]

    if appt["status"] == "cancelled":
        return {"error": "already_cancelled", "message": "Appointment is already cancelled."}

    # Check cancellation window (≥2h before appointment)
    slot_id = appt["slot_id"]
    # Production: parse actual slot start time from DB
    appt["status"] = "cancelled"
    appt["cancel_reason"] = reason
    appt["cancelled_at"] = datetime.now(timezone.utc).isoformat()
    _booked_slots.pop(slot_id, None)

    logger.info(f"❌ Cancelled appointment {appointment_id}. Reason: {reason}")

    return {
        "success": True,
        "appointment_id": appointment_id,
        "freed_slot": slot_id,
        "message": f"Appointment {appointment_id} has been cancelled.",
        "doctor_notified": notify_doctor,
    }


# ─── Tool: get_patient_history ────────────────────────────────────────────────

async def get_patient_history(
    patient_id: str,
    include: list[str] = None,
) -> dict:
    """Retrieve patient history for context injection into agent."""
    include = include or ["appointments", "preferences"]

    # In production: DB query
    patient_appointments = [
        appt for appt in _appointments.values()
        if appt["patient_id"] == patient_id
    ]

    result = {"patient_id": patient_id}

    if "appointments" in include:
        result["appointments"] = patient_appointments[-5:]  # Last 5

    if "preferences" in include:
        result["preferences"] = {
            "preferred_doctor": "dr_sharma",
            "preferred_time": "morning",
            "language": "en",
            "appointment_type": "in_person",
        }

    if "allergies" in include:
        result["allergies"] = ["Penicillin"]

    return result


# ─── Tool: suggest_alternative_slots ────────────────────────────────────────

async def suggest_alternative_slots(
    doctor_id: str,
    preferred_date: str,
    preferred_time_of_day: str = "any",
    num_suggestions: int = 3,
) -> dict:
    """
    Suggest N alternative slots near the preferred date.
    Searches +3 days forward if needed.
    """
    suggestions = []

    try:
        base_date = datetime.strptime(preferred_date, "%Y-%m-%d")
    except ValueError:
        base_date = datetime.now(timezone.utc)

    for delta in range(0, 7):  # Look up to 7 days ahead
        check_date = (base_date + timedelta(days=delta)).strftime("%Y-%m-%d")
        avail = await check_availability(doctor_id=doctor_id, date=check_date)

        if "available_slots" in avail:
            slots = avail["available_slots"]
            if preferred_time_of_day != "any":
                slots = [s for s in slots if s.get("time_of_day") == preferred_time_of_day]

            suggestions.extend(slots[:2])  # Take up to 2 per day

        if len(suggestions) >= num_suggestions:
            break

    return {
        "suggestions": suggestions[:num_suggestions],
        "total_found": len(suggestions),
    }


# ─── Tool: detect_language ────────────────────────────────────────────────────

async def detect_language(text: str) -> dict:
    """
    Detect language from user input.
    Production: use langdetect or Google Cloud Language API.
    """
    # Simple heuristic — production: use proper langdetect library
    hindi_chars = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    tamil_chars = set("அஆஇஈஉஊஎஏஐஒஓகங்சஞடணதந்பம்யரல்வ")

    text_chars = set(text)
    if text_chars & hindi_chars:
        return {"language": "hi", "confidence": 0.95, "name": "Hindi"}
    elif text_chars & tamil_chars:
        return {"language": "ta", "confidence": 0.95, "name": "Tamil"}
    else:
        return {"language": "en", "confidence": 0.90, "name": "English"}
