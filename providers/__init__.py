"""Provider registry. To add a provider, write a connector module and add one line here;
to remove one, delete its line (and module). Nothing else in the app names a provider.
"""

from __future__ import annotations

import os

from providers.base import Cancelled, Provider, ProviderError
from providers.gemini import GeminiProvider
from providers.groq import GroqProvider
from providers.ollama import OllamaProvider

PROVIDERS = {
    GeminiProvider.name: GeminiProvider,
    GroqProvider.name: GroqProvider,
    OllamaProvider.name: OllamaProvider,
}


def create(name: str) -> Provider:
    """Build a provider from the environment: <ENV_KEY> for the key, <NAME>_MODEL to override the model."""
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
    cls = PROVIDERS[name]
    api_key = os.environ.get(cls.env_key) if cls.env_key else None
    if cls.env_key and not api_key:
        raise ProviderError(f"{cls.env_key} is not set")
    return cls(api_key=api_key, model=os.environ.get(f"{name.upper()}_MODEL") or cls.default_model)


__all__ = ["PROVIDERS", "Cancelled", "Provider", "ProviderError", "create"]
