"""
llm/medgemma.py
Sends a prompt to MedGemma-4B via the mlx_lm Python library.

MedGemma runs entirely locally on Apple Silicon using the MLX framework.
It requires:
    pip install mlx mlx-lm
    A Hugging Face account with the MedGemma model terms accepted at:
    https://huggingface.co/google/medgemma-4b-it

On first use, the model downloads from Hugging Face (~4 GB) and caches locally.
Subsequent calls load from the local cache with no internet required.

Usage note: mlx_lm.load() takes several seconds to load the model weights.
In production, the model should be loaded once at app startup (lazy singleton
pattern) rather than reloaded on every request — see _get_model() below.
"""

from config import Config

# Module-level singleton — model and tokenizer loaded once and reused
_model = None
_tokenizer = None


def _get_model():
    """
    Load MedGemma model and tokenizer (singleton — only loads once per process).

    Returns:
        (model, tokenizer) tuple from mlx_lm.
    """
    global _model, _tokenizer
    if _model is None:
        try:
            from mlx_lm import load  # noqa: import inside function to defer loading
            _model, _tokenizer = load(Config.MEDGEMMA_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"[MedGemma] Failed to load model '{Config.MEDGEMMA_MODEL}': {e}\n"
                "Ensure you have accepted the model terms at huggingface.co/google/medgemma-4b-it "
                "and run `huggingface-cli login` before first use."
            )
    return _model, _tokenizer


def query_medgemma(prompt: str) -> str:
    """
    Generate a response from MedGemma-4B using the mlx_lm framework.

    Args:
        prompt: The fully constructed prompt string from prompt_builder.py.

    Returns:
        response_text (str): Generated text from MedGemma.
                             Returns a fallback error message if generation fails.
    """
    try:
        from mlx_lm import generate  # noqa: import inside function to defer loading
        model, tokenizer = _get_model()

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=Config.MEDGEMMA_MAX_TOKENS,
            temp=0.3,     # Low temperature for factual, consistent output
            verbose=False,
        )
        return response.strip()

    except RuntimeError as e:
        return f"[MedGemma Unavailable] {e}"
    except Exception as e:
        return f"[MedGemma Error] Unexpected error during generation: {e}"
