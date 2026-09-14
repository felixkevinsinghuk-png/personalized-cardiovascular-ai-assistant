# MedGemma-4B via the mlx_lm library (Apple Silicon only).
# First use downloads ~4 GB from Hugging Face and caches locally.
# Accept the model terms at huggingface.co/google/medgemma-4b-it first,
# then run: huggingface-cli login

from config import Config

_model     = None
_tokenizer = None


def _get_model():
    """Load MedGemma once and reuse — mlx_lm.load takes several seconds."""
    global _model, _tokenizer
    if _model is None:
        try:
            from mlx_lm import load
            _model, _tokenizer = load(Config.MEDGEMMA_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"[MedGemma] Failed to load '{Config.MEDGEMMA_MODEL}': {e}\n"
                "Have you accepted the model terms and logged in with huggingface-cli?"
            )
    return _model, _tokenizer


def query_medgemma(prompt: str) -> str:
    """Send a prompt to MedGemma and return the generated text."""
    try:
        from mlx_lm import generate
        model, tokenizer = _get_model()
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=Config.MEDGEMMA_MAX_TOKENS,
            temp=0.3,
            verbose=False,
        )
        return response.strip()
    except RuntimeError as e:
        return f"[MedGemma Unavailable] {e}"
    except Exception as e:
        return f"[MedGemma Error] {e}"
