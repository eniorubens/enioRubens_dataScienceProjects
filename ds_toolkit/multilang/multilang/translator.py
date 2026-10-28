from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


class LangMap:
    """Dictionary-based translator with persistent JSON cache."""

    def __init__(
        self,
        source: str = "pt",
        target: str = "en",
        cache_dir: str | None = None,
        offline: bool = False,
    ) -> None:
        self.source = source
        self.target = target
        self.offline = offline
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        labels: dict[str, str],
        force: bool = False,
    ) -> dict[str, str]:
        """Translate values of *labels* and return a new dict with same keys."""
        if self.source == self.target:
            return dict(labels)

        missing = [text for text in labels.values() if force or text not in self._cache]
        if missing:
            translations = self._translate_batch(missing)
            for text, translated in zip(missing, translations):
                self._cache[text] = translated
            self._save_cache()

        return {key: self._cache[text] for key, text in labels.items()}

    def switch(self, target: str) -> None:
        """Change target language and reload the corresponding cache."""
        self.target = target
        self._cache = self._load_cache()

    def show_cache(self) -> pd.DataFrame:
        """Return current cache as a DataFrame."""
        if not self._cache:
            return pd.DataFrame(columns=["source", "translation"])
        return pd.DataFrame(
            list(self._cache.items()),
            columns=["source", "translation"],
        )

    def warm_up(self, texts: list[str]) -> None:
        """Pre-translate a list of texts, populating the cache in one batch."""
        if self.source == self.target:
            return
        missing = [t for t in texts if t not in self._cache]
        if not missing:
            return
        translations = self._translate_batch(missing)
        for text, translated in zip(missing, translations):
            self._cache[text] = translated
        self._save_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self._cache_dir / f"{self.source}_{self.target}.json"

    def _load_cache(self) -> dict[str, str]:
        path = self._cache_path()
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with self._cache_path().open("w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _translate_batch(self, texts: list[str]) -> list[str]:
        if self.offline:
            raise RuntimeError(
                f"offline=True but {len(texts)} text(s) not found in cache: {texts}"
            )
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise ImportError(
                "deep_translator is required for online translation. "
                "Install it with:  pip install deep_translator"
            ) from exc
        translator = GoogleTranslator(source=self.source, target=self.target)
        return [translator.translate(t) for t in texts]
