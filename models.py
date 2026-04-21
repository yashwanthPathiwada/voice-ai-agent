from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class SessionState:
    session_id: str
    patient_id: Optional[str] = None
    language: str = "en"
    intent: Optional[str] = None
    pending_confirmation: Optional[dict] = None
    conversation_state: str = "greeting"
    history: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    last_active: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AppointmentRequest:
    
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    date: Optional[str] = None
    slot_id: Optional[str] = None
    appointment_type: str = "in_person"
    reason: Optional[str] = None