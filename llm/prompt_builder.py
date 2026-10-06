# Builds the prompts sent to BioMistral for the initial report and the chatbot.
#
# The prompts are structured to:
#   - Ground the LLM with actual numeric scores to reduce hallucination
#   - Constrain responses to a research/educational context
#   - Ensure the mandatory disclaimer is always included
#
# MANDATORY_DISCLAIMER is defined here (not in safety_filter.py) so it can be
# imported without creating a circular dependency.


# Mandatory disclaimer text appended to every LLM response by the safety filter.
# Defined here so it can be imported by safety_filter.py without circular imports.
MANDATORY_DISCLAIMER = (
    "\n\n---\n"
    "RESEARCH DISCLAIMER: This analysis is produced by an automated AI system "
    "for academic research purposes only. It does NOT constitute medical advice, "
    "diagnosis, or treatment. Always consult a qualified healthcare professional "
    "for any health concerns. This tool has not been validated for clinical use."
)


def build_prompt(
    dr_grade: int,
    dr_label: str,
    cvd_score: float,
    fused_score: float,
    risk_level: str,
    clinical_history: str = None,
    ccs: float = None,
    retinal_caption: str = None,
) -> str:
    """Build the structured report prompt from the pipeline outputs."""
    
    # Format the history section
    if clinical_history and ccs is not None:
        history_text = f"Patient Clinical History (as provided by patient):\n{clinical_history}\n\nClinical Context Score extracted from history: {ccs:.4f}"
    else:
        history_text = "Patient Clinical History: Not provided by the patient.\nClinical Context Score: N/A"
    prompt = f"""You are a medical AI assistant in a non-clinical research context. \
Your task is to provide an educational explanation of retinal-based cardiovascular \
risk assessment results. You must not provide clinical advice or personal medical \
recommendations.

--- ANALYSIS RESULTS ---
Diabetic Retinopathy (DR) Grade: {dr_grade}/4 — {dr_label}
Cardiovascular Disease (CVD) Risk Score: {cvd_score:.4f} (range 0.0 to 1.0)
Fused Risk Score: {fused_score:.4f} (range 0.0 to 1.0)
Overall Risk Level: {risk_level}

--- RETINAL FEATURE DESCRIPTION ---
{retinal_caption if retinal_caption else "Not available."}

{history_text}

Important: If the patient has mentioned a stent, bypass, heart attack, or coronary artery disease, \
acknowledge in your report that these conditions are outside the scope of retinal imaging and that the \
clinical history score has been incorporated to reflect this.

--- YOUR TASK ---
Write a clear, structured cardiovascular risk explanation for a medical researcher \
or student. Your response must include the following sections exactly:

1. **Retinal Findings Summary**: What the retinal image shows or does not show.

2. **Clinical Context Considered**: Summary of what was extracted from the patient's typed history \
and how it influenced the final risk score. If no history was provided, state this.

3. **Risk Score Breakdown**: Explain the retinal DR contribution, retinal CVD contribution, \
and clinical history contribution leading to the final fused score and risk level.

4. **System Limitations**: Explicit statement of what retinal imaging cannot detect. \
Especially important if stent or coronary artery disease was mentioned in the history.

Keep your response factual, evidence-based, and educational. Do not address the user \
as a patient. Do not provide treatment advice or suggest seeing a specific doctor. \
End your response with the following disclaimer exactly as written:

{MANDATORY_DISCLAIMER}"""

    return prompt


def build_chat_prompt(
    user_message: str,
    dr_grade: int,
    dr_label: str,
    cvd_score: float,
    fused_score: float,
    risk_level: str,
) -> str:
    """Build a contextualised chatbot prompt that includes the original prediction results."""
    prompt = f"""You are a medical AI research assistant helping explain retinal analysis results.
You have access to the following analysis results from this session:

ANALYSIS RESULTS:
- Diabetic Retinopathy (DR) Grade: {dr_grade}/4 — {dr_label}
- CVD Risk Score: {cvd_score:.4f} (scale 0.0 to 1.0, lower is better)
- Fused Risk Score: {fused_score:.4f} (scale 0.0 to 1.0, lower is better)
- Overall Risk Level: {risk_level}

SCORING LOGIC:
- Fused score below 0.20 = Low Risk
- Fused score 0.20 to 0.35 = Borderline Risk
- Fused score 0.35 to 0.65 = Moderate Risk
- Fused score above 0.65 = High Risk
- If clinical history is provided, fusion weights are: DR (30%), CVD (45%), History (25%)
- If clinical history is empty, fusion weights are: DR (40%), CVD (60%)

The user asks: "{user_message}"

CRITICAL INSTRUCTIONS:
1. If the user's question is completely unrelated to the analysis, retinal images, or cardiovascular risk (e.g., asking for recipes, coding help, or general knowledge), you MUST refuse to answer. Reply EXACTLY with this sentence and nothing else: "This tool is not designed for these types of questions. It is exclusively for biological and medical analysis of retinal images."
2. If the question IS related, answer clearly and specifically using the numbers above. Explain the reasoning behind the risk classification in plain language. Be educational and factual. Do not give personal medical advice or suggest specific treatments. End with: This is a research tool only and does not constitute medical advice."""

    return prompt
