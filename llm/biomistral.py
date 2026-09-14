# Sends prompts to BioMistral-7B via Ollama's local REST API (port 11434).
# Pull the model first with:  ollama pull biomistral
# Safety filtering and disclaimer are handled separately in safety_filter.py.

import requests
from config import Config


def query_biomistral(prompt: str) -> str:
    """Send a prompt to BioMistral and return the raw generated text."""
    payload = {
        "model":   Config.OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature": 0.3,   # low temp for factual, consistent output
            "top_p":       0.9,
            "num_predict": 600,   # roughly 450 words
        },
    }

    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=Config.OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            f"[BioMistral Unavailable] Could not connect to Ollama on {Config.OLLAMA_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        return "[BioMistral Timeout] The model took too long to respond. Try again or shorten the prompt."
    except requests.exceptions.HTTPError as e:
        return f"[BioMistral HTTP Error] {e}"
    except Exception as e:
        return f"[BioMistral Error] {e}"
