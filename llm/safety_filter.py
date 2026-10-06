# Two safety mechanisms applied to every LLM response:
#
# 1. Phrase blocking — if the response contains any of the blocked clinical phrases,
#    it's replaced entirely with a safe fallback message.
# 2. Disclaimer — the mandatory research disclaimer is always appended, even to the fallback.

from llm.prompt_builder import MANDATORY_DISCLAIMER


# Phrases that suggest the model has strayed into actual clinical advice
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
    Check the LLM response for blocked phrases and ensure the disclaimer is present.
    Returns (filtered_response, was_filtered).
    """
    triggered = any(phrase in raw_response.lower() for phrase in BLOCKED_PHRASES)
    response_text = SAFE_FALLBACK if triggered else raw_response.strip()

    # The prompt instructs Mistral-7B to end with the disclaimer, so it usually does.
    # Only append it if it's genuinely missing — otherwise we get it twice.
    if "RESEARCH DISCLAIMER" not in response_text:
        response_text = response_text + MANDATORY_DISCLAIMER

    return response_text, triggered
