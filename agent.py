"""
agent.py — Clinical Appointment Booking Agent
Uses OpenAI function calling with streaming + visible reasoning traces.
Tool calls run in parallel where dependency-free.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from tools import (
    check_availability,
    book_appointment,
    reschedule_appointment,
    cancel_appointment,
    get_patient_history,
    suggest_alternative_slots,
    detect_language,
)
from memory import MemoryManager, SessionState

logger = logging.getLogger(__name__)

# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Priya, a clinical appointment booking assistant.
You speak English, Hindi, and Tamil. Always respond in the same language the patient uses.

## Your Capabilities
- Check doctor availability
- Book, reschedule, and cancel appointments
- Retrieve patient history
- Suggest alternatives when slots are unavailable

## Behavior Rules
1. ALWAYS confirm critical actions (booking, cancellation) before executing.
2. If a slot is unavailable, immediately suggest 3 alternatives.
3. For ambiguous requests ("tomorrow morning"), ask for clarification ONCE.
4. Never hallucinate availability — always call check_availability first.
5. If patient mentions urgency ("chest pain", "emergency"), escalate to emergency line.
6. Keep responses concise for voice — max 2 sentences per turn.
7. Use patient's name when you know it (from history).

## Reasoning Pattern (THINK BEFORE SPEAKING)
Before responding:
1. What is the patient's intent? (book / reschedule / cancel / query)
2. What information is missing? (doctor? date? time?)
3. Which tools do I need? Can any run in parallel?
4. What confirmation is needed?

## Language Rules
- Detect language from first message
- Switch language mid-conversation if patient switches
- Persist language preference in memory

## Response Format for Voice
- No markdown, no bullet points
- Short, natural sentences
- Confirmation questions as yes/no when possible
"""

# ─── Tool JSON Schemas ────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check available appointment slots for a specific doctor and date. Always call this before booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "Doctor's unique identifier or name (e.g., 'dr_sharma', 'Dr. Priya Sharma')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format. Use today's date if 'today', calculate for 'tomorrow', etc."
                    },
                    "specialty": {
                        "type": "string",
                        "description": "Medical specialty if no specific doctor named (e.g., 'cardiology', 'general')",
                        "enum": ["general", "cardiology", "orthopedics", "neurology", "pediatrics", "gynecology"]
                    }
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment after confirming availability. Requires patient confirmation first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Unique patient identifier"
                    },
                    "doctor_id": {
                        "type": "string",
                        "description": "Doctor's unique identifier"
                    },
                    "slot_id": {
                        "type": "string",
                        "description": "Specific slot ID returned by check_availability"
                    },
                    "appointment_type": {
                        "type": "string",
                        "enum": ["in_person", "teleconsult"],
                        "description": "Mode of appointment"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for visit (optional, for doctor's reference)"
                    }
                },
                "required": ["patient_id", "doctor_id", "slot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule an existing appointment to a new slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "Existing appointment ID to reschedule"
                    },
                    "new_slot_id": {
                        "type": "string",
                        "description": "New slot ID from check_availability"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for rescheduling"
                    }
                },
                "required": ["appointment_id", "new_slot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment. Always confirm with patient before calling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "Appointment ID to cancel"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation"
                    },
                    "notify_doctor": {
                        "type": "boolean",
                        "description": "Whether to notify the doctor",
                        "default": True
                    }
                },
                "required": ["appointment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_history",
            "description": "Retrieve patient's appointment history, preferences, and medical record summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient's unique identifier"
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["appointments", "preferences", "allergies", "medications"]
                        },
                        "description": "What to include in the response"
                    }
                },
                "required": ["patient_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_alternative_slots",
            "description": "Suggest alternative appointment slots when requested slot is unavailable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {"type": "string"},
                    "preferred_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "preferred_time_of_day": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "any"]
                    },
                    "num_suggestions": {
                        "type": "integer",
                        "default": 3,
                        "description": "How many alternatives to suggest"
                    }
                },
                "required": ["doctor_id", "preferred_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_language",
            "description": "Detect the language of the patient's input. Call on first message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to detect language from"}
                },
                "required": ["text"]
            }
        }
    }
]

# ─── Tool Dispatcher ──────────────────────────────────────────────────────────

TOOL_MAP = {
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "reschedule_appointment": reschedule_appointment,
    "cancel_appointment": cancel_appointment,
    "get_patient_history": get_patient_history,
    "suggest_alternative_slots": suggest_alternative_slots,
    "detect_language": detect_language,
}


async def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a single tool call asynchronously."""
    func = TOOL_MAP.get(tool_name)
    if not func:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = await func(**args)
        logger.info(f"🔧 Tool {tool_name}({args}) → {result}")
        return result
    except Exception as e:
        logger.exception(f"Tool {tool_name} failed: {e}")
        return {"error": str(e)}


async def execute_tools_parallel(tool_calls: list[dict]) -> list[dict]:
    """
    Execute independent tool calls in parallel.
    Dependency-free tools (e.g., check_availability + get_patient_history)
    run concurrently to minimize latency.
    """
    tasks = [
        execute_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        for tc in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if not isinstance(r, Exception) else {"error": str(r)}
        for r in results
    ]


# ─── Agent Class ──────────────────────────────────────────────────────────────

class ClinicalAgent:
    """
    Stateful agent per session.
    Streams reasoning traces + final response text back to the caller.
    Uses OpenAI function calling with LangChain chat model.
    """

    def __init__(self, memory: MemoryManager, session_id: str):
        self.memory = memory
        self.session_id = session_id

        # GPT-4o with streaming — use claude-3-5-sonnet via Anthropic for production
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            streaming=True,
        )

    async def stream_response(
        self,
        user_input: str,
        session: SessionState,
    ) -> AsyncGenerator[dict, None]:
        """
        Core agentic loop with streaming.
        Yields:
          { type: "thinking", content: "..." }   ← reasoning traces
          { type: "text",     content: "..." }   ← final response for TTS
        """

        # ── Inject session + long-term memory into context ────────────────
        memory_context = await self.memory.build_prompt_context(self.session_id)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n\n" + memory_context),
        ]

        # Add conversation history (last 10 turns)
        for turn in session.history[-10:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_input))

        # ── Agentic Loop ──────────────────────────────────────────────────
        max_iterations = 5

        for iteration in range(max_iterations):
            yield {"type": "thinking", "content": f"[Iteration {iteration+1}] Calling LLM..."}

            # Stream LLM response
            collected_chunks = []
            tool_calls_raw = []
            text_buffer = ""

            async for chunk in self.llm.astream(
                messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            ):
                # Accumulate text
                if chunk.content:
                    text_buffer += chunk.content
                # Accumulate tool call info
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    tool_calls_raw.extend(chunk.tool_calls)
                collected_chunks.append(chunk)

            # Reconstruct full AI message
            ai_message_content = text_buffer
            tool_calls = self._deduplicate_tool_calls(tool_calls_raw)

            yield {
                "type": "thinking",
                "content": f"Text: '{text_buffer[:80]}...' | Tools: {[tc['function']['name'] for tc in tool_calls]}",
            }

            # ── No tool calls → final response ───────────────────────────
            if not tool_calls:
                # Update session history
                session.history.append({"role": "user", "content": user_input})
                session.history.append({"role": "assistant", "content": text_buffer})
                await self.memory.save_session_state(self.session_id, session)

                yield {"type": "text", "content": text_buffer}
                return

            # ── Execute tool calls (parallel where possible) ──────────────
            yield {
                "type": "thinking",
                "content": f"Executing {len(tool_calls)} tool(s) in parallel: " +
                           ", ".join(tc["function"]["name"] for tc in tool_calls),
            }

            tool_results = await execute_tools_parallel(tool_calls)

            # Append AI message + tool results to conversation
            messages.append(AIMessage(
                content=ai_message_content,
                additional_kwargs={"tool_calls": tool_calls},
            ))

            for tc, result in zip(tool_calls, tool_results):
                messages.append(ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tc["id"],
                ))
                yield {
                    "type": "thinking",
                    "content": f"Tool result [{tc['function']['name']}]: {json.dumps(result)[:200]}",
                }

        # Fallback if max iterations hit
        yield {"type": "text", "content": "I'm sorry, I had trouble processing that. Could you please repeat?"}

    def _deduplicate_tool_calls(self, raw: list) -> list[dict]:
        """Deduplicate tool call chunks collected during streaming."""
        seen_ids = set()
        result = []
        for tc in raw:
            tc_id = tc.get("id") or tc.get("index", "")
            if tc_id not in seen_ids:
                seen_ids.add(tc_id)
                result.append(tc)
        return result
