# Sends the patient's free-text clinical history to BioMistral and extracts a single
# float (0.0–1.0) representing additional cardiovascular risk from their history.
# This score is then used as a third input to the fusion formula alongside DR and CVD scores.

import logging
from llm.biomistral import query_biomistral

logger = logging.getLogger(__name__)

# The prompt instructs the model to return only a number — no words, no explanation.
# Scoring rules are explicit to reduce hallucination and keep outputs consistent.
SYSTEM_PROMPT = """You are a clinical risk scoring assistant.
Read the patient description below and return ONLY a single decimal number between 0.0 and 1.0 representing their cardiovascular risk based on the clinical history mentioned.
Do not return any words, explanation, or punctuation.
Return only the number.

Scoring rules:
- Stent, bypass, or heart attack mentioned = 0.90
- Coronary artery disease or blocked arteries mentioned = 0.90
- Diabetes over 10 years = add 0.20 to base
- Diabetes 5 to 10 years = add 0.10 to base
- High blood pressure or hypertension = add 0.15
- Current smoker = add 0.10
- Family history of heart disease = add 0.08
- High cholesterol = add 0.08
- No significant history = return 0.10
- Base score is 0.10. Add the above modifiers if mentioned.
- Always cap the final number at 1.0.

User text:
{user_text}

Return only the number."""


def extract_clinical_context_score(clinical_history_text: str) -> float:
    """
    Query BioMistral with the patient's typed history and parse out a risk score float.
    Returns 0.10 (low baseline) if the input is empty or the model fails to parse.
    """
    if not clinical_history_text or not clinical_history_text.strip():
        return 0.10

    prompt = SYSTEM_PROMPT.format(user_text=clinical_history_text.strip())

    try:
        raw = query_biomistral(prompt)
        score = float(raw.strip().replace('"', '').replace("'", ""))
        return max(0.0, min(1.0, score))
    except ValueError:
        logger.error(f"Could not parse float from BioMistral response: {raw!r}")
        return 0.10
    except Exception as e:
        logger.error(f"extract_clinical_context_score failed: {e}")
        return 0.10
