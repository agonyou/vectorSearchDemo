#!/usr/bin/env python3
"""Central configuration for load.py and the rag*.py scripts (backlog #3).

Single source of truth for Couchbase + embedding settings. Reads the process environment
(populate it first with python-dotenv's load_dotenv()), validates per-provider requirements,
and resolves the effective embedding provider / model / dimension with a documented precedence:

    provider:   --embedding-provider  >  SENTENCE_TRANSFORMER_MODEL set → local, else openai
    model:      --embedding-model      >  SENTENCE_TRANSFORMER_MODEL / OPENAI_EMBEDDING_MODEL  >  built-in default
    dimensions: --dimensions           >  VECTOR_DIMENSIONS                                    >  derived from the model

Built on pydantic (already an installed dependency) rather than pydantic-settings so no new
runtime package is required.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# Output dimensions for known embedding models, used to derive the expected dimension when
# neither --dimensions nor VECTOR_DIMENSIONS is set. Keyed by bare name and "org/name" form.
KNOWN_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def known_dimension(model: str) -> Optional[int]:
    """Return the known output dimension for a model name, or None if unknown."""
    return KNOWN_DIMENSIONS.get(model) or KNOWN_DIMENSIONS.get(model.split("/")[-1])


def _as_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Settings(BaseModel):
    """Resolved configuration. Build with Settings.load(...); read the effective values via the
    provider / embedding_model / declared_dimensions properties."""

    # Couchbase
    couchbase_connstr: str = "couchbase://127.0.0.1"
    couchbase_username: str = "Administrator"
    couchbase_password: str = "password"

    # Embedding provider inputs (from .env)
    sentence_transformer_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = DEFAULT_OPENAI_MODEL
    openai_chat_model: str = "gpt-4o-mini"
    vector_dimensions: Optional[int] = None
    hf_token: Optional[str] = None

    # Per-invocation CLI overrides (highest precedence)
    provider_override: Optional[str] = None
    model_override: Optional[str] = None
    dimensions_override: Optional[int] = None

    @classmethod
    def load(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> "Settings":
        """Build Settings from the environment (after .env is loaded), applying CLI overrides."""
        return cls(
            couchbase_connstr=os.getenv("COUCHBASE_CONNSTR", "couchbase://127.0.0.1"),
            couchbase_username=os.getenv("COUCHBASE_USERNAME", "Administrator"),
            couchbase_password=os.getenv("COUCHBASE_PASSWORD", "password"),
            sentence_transformer_model=os.getenv("SENTENCE_TRANSFORMER_MODEL") or None,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            vector_dimensions=_as_int(os.getenv("VECTOR_DIMENSIONS")),
            hf_token=os.getenv("HF_TOKEN") or None,
            provider_override=provider,
            model_override=model,
            dimensions_override=dimensions,
        )

    @property
    def provider(self) -> str:
        """Effective embedding provider: CLI override, else inferred from .env."""
        if self.provider_override:
            return self.provider_override
        return "local" if self.sentence_transformer_model else "openai"

    @property
    def embedding_model(self) -> str:
        """Effective embedding model name for the resolved provider."""
        if self.model_override:
            return self.model_override
        if self.provider == "local":
            return self.sentence_transformer_model or DEFAULT_LOCAL_MODEL
        return self.openai_embedding_model or DEFAULT_OPENAI_MODEL

    @property
    def declared_dimensions(self) -> Optional[int]:
        """Expected embedding dimension: --dimensions, else VECTOR_DIMENSIONS, else derived from
        the model. None means "unknown" (the runtime preflight then has nothing to compare against)."""
        if self.dimensions_override is not None:
            return self.dimensions_override
        if self.vector_dimensions is not None:
            return self.vector_dimensions
        return known_dimension(self.embedding_model)

    def validate_for_embedding(self) -> None:
        """Fail fast on missing per-provider requirements for computing embeddings."""
        if self.provider == "openai" and not self.openai_api_key:
            raise SystemExit(
                "OPENAI_API_KEY is required for the OpenAI embedding provider. "
                "Set it in .env, or use --embedding-provider local."
            )

    def validate_for_rag(self) -> None:
        """RAG scripts also need OpenAI for the chat completion, regardless of embedding provider."""
        self.validate_for_embedding()
        if not self.openai_api_key:
            raise SystemExit("OPENAI_API_KEY is required for RAG (LLM generation).")
