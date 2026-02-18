"""
Run tokenizer analysis with zip2zip LZW compression tokenizer support.

This script registers a Zip2ZipTokenizerWrapper under the class key "zip2zip"
and then delegates to the standard run_tokenizer_analysis.main(), so all the
usual CLI flags work unchanged.

python scripts/run_zip2zip_analysis.py --tokenizer-config configs/zip2zip_tokenizers.json --language-config configs/core_lang_config.json --measurement-config configs/text_measurement_config_lines.json --verbose --run-grouped-analysis  --per-language-plots --no-global-lines --update-results-md --dataset flores_core

"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Root of tokenizer-intrinsic-evals (parent of this scripts/ dir)
EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_ROOT))

# zip2zip_tokenizer source tree (sibling library)
ZIP2ZIP_SRC = EVAL_ROOT.parent / "zip2zip_tokenizer" / "src"
if ZIP2ZIP_SRC.exists():
    sys.path.insert(0, str(ZIP2ZIP_SRC))

# Make run_tokenizer_analysis importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Zip2Zip wrapper
# ---------------------------------------------------------------------------
from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper, register_tokenizer_class

logger = logging.getLogger(__name__)


class Zip2ZipTokenizerWrapper(TokenizerWrapper):
    """
    TokenizerWrapper that applies LZW compression on top of a base HF tokenizer.

    Each call to encode() starts with a fresh codebook, so compression reflects
    within-text repetition only (consistent with per-sentence evaluation).

    Config keys
    -----------
    base_tokenizer   : str  – HuggingFace model name or local path for the base tokenizer
    max_codebook_size: int  – Maximum number of compound tokens (default 4096)
    max_subtokens    : int  – Maximum base-tokens per compound token (default 8)
    """

    def __init__(
        self,
        name: str,
        base_tokenizer_name: str,
        max_codebook_size: int,
        max_subtokens: int,
    ) -> None:
        self._name = name

        from transformers import AutoTokenizer
        from zip2zip_tokenizer import Zip2ZipTokenizer

        base_tok = AutoTokenizer.from_pretrained(base_tokenizer_name)
        # Exclude special/added tokens from being merged
        disabled_ids = set(base_tok.get_added_vocab().values())

        self._tokenizer = Zip2ZipTokenizer(
            base_tok,
            max_codebook_size=max_codebook_size,
            max_subtokens=max_subtokens,
            disabled_ids=disabled_ids,
        )
        # Report the theoretical maximum vocabulary (base + full codebook)
        self._vocab_size = self._tokenizer.initial_vocab_size + max_codebook_size
        self._base_vocab: Dict[str, int] = base_tok.get_vocab()

    # ------------------------------------------------------------------
    # TokenizerWrapper interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return self._name

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Optional[Dict[str, int]]:
        # Return the static base vocabulary; the dynamic codebook entries
        # are not included here (they are text-dependent).
        return self._base_vocab

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str) -> List[int]:
        """Encode *text* and return the LZW-compressed token ID sequence."""
        result = self._tokenizer.batch_encode_plus([text])
        return result["input_ids"][0]

    def can_pretokenize(self) -> bool:
        return False

    def pretokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Zip2ZipTokenizerWrapper does not support pretokenization")

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any]) -> "Zip2ZipTokenizerWrapper":
        if "base_tokenizer" not in config:
            raise ValueError(
                f"Zip2Zip tokenizer config for '{name}' must include 'base_tokenizer'"
            )
        return cls(
            name=name,
            base_tokenizer_name=config["base_tokenizer"],
            max_codebook_size=config.get("max_codebook_size", 4096),
            max_subtokens=config.get("max_subtokens", 8),
        )


# ---------------------------------------------------------------------------
# Register the class, then hand off to the standard analysis entry-point
# ---------------------------------------------------------------------------
register_tokenizer_class("zip2zip", Zip2ZipTokenizerWrapper)
logger.info("Registered 'zip2zip' tokenizer class.")

from run_tokenizer_analysis import main  # noqa: E402 (must come after registration)

if __name__ == "__main__":
    main()
