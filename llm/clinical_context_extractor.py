"""
llm/clinical_context_extractor.py
Sends free-text clinical history to Mistral-7B and extracts a single CCS float.
"""

import logging
from config import Config
from llm.biomistral import query_biomistral

logger = logging.getLogger(__name__)

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
    Query Mistral to extract a clinical context score from the user's history.
    """
    if not clinical_history_text or not clinical_history_text.strip():
        return 0.10
        
    prompt = SYSTEM_PROMPT.format(user_text=clinical_history_text.strip())
    
    try:
        # Call existing Ollama function
        raw_response = query_biomistral(prompt)
        
        # Clean up the response (strip whitespace, newlines, etc.)
        cleaned = raw_response.strip().replace('"', '').replace("'", "")
        
        score = float(cleaned)
        # Cap between 0.0 and 1.0
        return max(0.0, min(1.0, score))
        
    except ValueError:
        logger.error(f"Failed to parse float from Mistral response: {raw_response}")
        return 0.10
    except Exception as e:
        logger.error(f"Error in extract_clinical_context_score: {e}")
        return 0.10
