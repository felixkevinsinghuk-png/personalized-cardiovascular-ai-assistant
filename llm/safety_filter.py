"""
llm/safety_filter.py
Scans LLM responses for blocked clinical advice phrases and appends the
mandatory research disclaimer to every response.

This module implements two safety mechanisms:

1. PHRASE BLOCKING: If the LLM response contains any of the blocked phrases,
   the entire response is replaced with a safe fallback message. This prevents
   the tool from being mistaken for a clinical diagnostic system.

2. DISCLAIMER APPENDING: The mandatory research disclaimer is appended to every
   response — including the safe fallback — regardless of whether the phrase
   filter was triggered.

The blocked phrase list is intentionally broad to err on the side of safety.
"""

from llm.prompt_builder import MANDATORY_DISCLAIMER


# ---------------------------------------------------------------------------
# Blocked clinical advice phrases (case-insensitive matching)
# ---------------------------------------------------------------------------
BLOCKED_PHRASES = [
    "you are diagnosed",
    "you should take",
    "you need to take",
    "prescribe",
    "prescription",
    "clinical advice",
    "see a doctor immediately",
    "medical emergency",
    "call an ambulance",
    "go to the hospital",
    "i diagnose",
    "you are at risk of dying",
    "start medication",
    "stop medication",
]

# Safe fallback message shown when a response is blocked
SAFE_FALLBACK = (
    "The AI system detected that the generated response may contain content "
    "that could be misinterpreted as clinical advice. For safety reasons, this "
    "response has been withheld.\n\n"
    "Please note: This tool is a research prototype only. It does not provide "
    "medical diagnoses or treatment recommendations. The numerical scores shown "
    "reflect AI model outputs from retinal image analysis and should not be "
    "interpreted as clinical findings."
)


def apply_safety_filter(raw_response: str) -> tuple[str, bool]:
    """
    Apply the safety filter to an LLM-generated response.

    Steps:
    1. Check for any blocked clinical advice phrases (case-insensitive)
    2. If found, replace the response with SAFE_FALLBACK and set was_filtered=True
    3. Append MANDATORY_DISCLAIMER to whichever response text results from step 1/2

    Args:
        raw_response: The raw text string returned by the LLM.

    Returns:
        safe_response (str):  The filtered response with disclaimer appended.
        was_filtered  (bool): True if the phrase filter was triggered.
    """
    response_lower = raw_response.lower()

    # Check for any blocked phrase
    triggered = any(phrase in response_lower for phrase in BLOCKED_PHRASES)

    if triggered:
        response_text = SAFE_FALLBACK
    else:
        response_text = raw_response.strip()

    # Always append the mandatory disclaimer
    safe_response = response_text + MANDATORY_DISCLAIMER

    return safe_response, triggered
