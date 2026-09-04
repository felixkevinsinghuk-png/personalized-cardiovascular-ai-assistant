"""
llm/biomistral.py
Sends a prompt to BioMistral-7B running locally via Ollama.

Ollama exposes a local REST API on port 11434.
BioMistral must be pulled first with: ollama pull biomistral

API endpoint: POST http://localhost:11434/api/generate
Request body: { "model": "biomistral", "prompt": "...", "stream": false }
Response:     { "response": "generated text ...", ... }

The function returns the raw generated text string.
Safety filtering and disclaimer appending are handled in safety_filter.py.
"""

import requests
from config import Config


def query_biomistral(prompt: str) -> str:
    """
    Send a prompt to BioMistral via the local Ollama API and return the response.

    Args:
        prompt: The fully constructed prompt string from prompt_builder.py.

    Returns:
        response_text (str): Raw LLM-generated text.
                             Returns a fallback error message if the request fails.
    """
    payload = {
        "model": Config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,           # Receive the full response at once
        "options": {
            "temperature": 0.3,    # Low temperature for factual, consistent output
            "top_p": 0.9,
            "num_predict": 600,    # Max tokens to generate (~450 words)
        },
    }

    try:
        response = requests.post(
            Config.OLLAMA_URL,
            json=payload,
            timeout=Config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            "[BioMistral Unavailable] Could not connect to Ollama on "
            f"{Config.OLLAMA_URL}. Please ensure Ollama is running: "
            "`ollama serve`"
        )
    except requests.exceptions.Timeout:
        return (
            "[BioMistral Timeout] The model took too long to respond. "
            "Try again or reduce the prompt length."
        )
    except requests.exceptions.HTTPError as e:
        return f"[BioMistral HTTP Error] {e}"
    except Exception as e:
        return f"[BioMistral Error] Unexpected error: {e}"
