#!/usr/bin/env python3
"""
Interactive or CLI-driven RAG demo using Couchbase Hyperscale Vector Queries.

Rules:
- --bucket, --scope, --collection are REQUIRED
- If ANY user-input CLI args are provided, ALL must be provided
- Otherwise, user inputs are collected interactively
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

load_dotenv()

# Embedding model is cached on disk; force HF offline so sentence-transformers doesn't
# revalidate the cache over the network on every run (the HEAD storm → HTTP 429 backoffs).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOG = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        LOG.error("Required environment variable %s is not set", name)
        sys.exit(1)
    return val


# ------------------------------------------------------------
# OpenAI Client (OpenAI-compatible)
# ------------------------------------------------------------

_openai_client = None


def get_openai_client() -> OpenAI:
    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=get_env("OPENAI_API_KEY", required=True),
            base_url=get_env(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ),
        )

    return _openai_client


# ------------------------------------------------------------
# Dataset / Query Configuration
# ------------------------------------------------------------

@dataclass
class RAGConfig:
    bucket: str
    scope: str
    collection: str

    embedding_field: str = "embedding"
    content_field: str = "contents"

    limit: int = 5


def load_config_from_args(args) -> RAGConfig:
    return RAGConfig(
        bucket=args.bucket,
        scope=args.scope,
        collection=args.collection,
    )


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

_st_model = None

# CLI overrides for the embedding provider ("local"/"openai") and model name. None → infer from
# .env. Set from --embedding-provider / --embedding-model in main().
_provider_override = None
_model_override = None


def resolve_provider() -> str:
    """Effective embedding provider: the --embedding-provider override if given, else inferred
    from .env (SENTENCE_TRANSFORMER_MODEL set → 'local', otherwise 'openai')."""
    if _provider_override:
        return _provider_override
    return "local" if get_env("SENTENCE_TRANSFORMER_MODEL") else "openai"


def st_model_name() -> str:
    """Local model name — --embedding-model if given, else SENTENCE_TRANSFORMER_MODEL, else the
    documented 384-dim default."""
    return _model_override or get_env("SENTENCE_TRANSFORMER_MODEL") or "all-MiniLM-L6-v2"


def openai_embedding_model() -> str:
    """OpenAI embedding model — --embedding-model if given, else OPENAI_EMBEDDING_MODEL, else default."""
    return _model_override or get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def compute_embedding(text: str) -> List[float]:
    """
    Compute an embedding using the resolved provider:
    - "local"  → SentenceTransformer (falls back to OpenAI only on error)
    - "openai" → OpenAI-compatible API
    """
    global _st_model

    if resolve_provider() == "local":
        name = st_model_name()
        try:
            if _st_model is None:
                LOG.info("Loading SentenceTransformer: %s", name)
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(name)

            return _st_model.encode(text).tolist()
        except Exception as e:
            LOG.warning("SentenceTransformer failed, falling back to OpenAI: %s", e)

    client = get_openai_client()
    model = openai_embedding_model()

    resp = client.embeddings.create(
        model=model,
        input=text,
    )

    return resp.data[0].embedding


def preflight_dimensions(actual_dim: int, collection: str) -> None:
    """Fail fast when the embedding dimension disagrees with VECTOR_DIMENSIONS (.env).
    A mismatch (e.g. a 1536-dim query against a 384-dim collection) otherwise returns an
    empty result with no error — this turns that silent miss into an actionable message."""
    declared = get_env("VECTOR_DIMENSIONS")
    if not declared:
        return
    try:
        declared_dim = int(declared)
    except ValueError:
        LOG.warning("VECTOR_DIMENSIONS=%r is not an integer; skipping dimension preflight", declared)
        return
    if actual_dim != declared_dim:
        LOG.error(
            "Embedding dimension mismatch: provider '%s' produced %d-dim vectors but "
            "VECTOR_DIMENSIONS=%d (querying collection '%s'). The provider, the collection's "
            "vector index, and VECTOR_DIMENSIONS must all agree (local/MiniLM=384, OpenAI=1536).",
            resolve_provider(), actual_dim, declared_dim, collection,
        )
        sys.exit(1)


# ------------------------------------------------------------
# LLM Generation
# ------------------------------------------------------------

def generate_with_llm(prompt: str, context: str) -> str:
    client = get_openai_client()
    model = get_env("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the provided context. "
                "If the context is insufficient, say so plainly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nPrompt:\n{prompt}",
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------
# Couchbase
# ------------------------------------------------------------

def get_cluster():
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions
    from couchbase.auth import PasswordAuthenticator

    return Cluster(
        get_env("COUCHBASE_CONNSTR", "couchbase://127.0.0.1"),
        ClusterOptions(
            PasswordAuthenticator(
                get_env("COUCHBASE_USERNAME", "Administrator"),
                get_env("COUCHBASE_PASSWORD", "password"),
            )
        ),
    )


def run_hyperscale_query(
    cluster,
    cfg: RAGConfig,
    query_embedding: List[float],
) -> List[str]:
    params = {
        "vector": query_embedding,
        "limit": cfg.limit,
    }

    statement = f"""
    SELECT RAW 'Title: ' || m.Title || ': ' || m.{cfg.content_field}
    FROM `{cfg.bucket}`.`{cfg.scope}`.`{cfg.collection}` m
    ORDER BY APPROX_VECTOR_DISTANCE(m.{cfg.embedding_field}, $vector, "COSINE", 3)
    LIMIT $limit
    """

    print("\n=== Query ===\n")
    print(statement, {**params, "vector": "<removed>"})

    return [row for row in cluster.query(statement, **params)]


# ------------------------------------------------------------
# CLI / Input Handling
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperscale Vector RAG Demo")

    # Required dataset args
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--collection", required=True)

    # Optional user-input args
    parser.add_argument("--prompt")

    # Embedding provider override (see #1). Omit to keep the .env-driven default.
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai"],
        default=None,
        help="Force the embedding provider. Omit to infer from .env "
             "(SENTENCE_TRANSFORMER_MODEL set → local, else openai). Must match the "
             "collection's index dimensions (local/MiniLM=384, openai=1536).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model name for the chosen provider (local sentence-transformers "
             "model or OpenAI embedding model). Its output dimension must match the collection's index.",
    )

    return parser.parse_args()


def resolve_inputs(args):
    if args.prompt:
        return args.prompt

    prompt = input("\nEnter your prompt:\n> ").strip()

    if not prompt:
        print("Prompt required")
        sys.exit(1)

    return prompt


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== Couchbase Hyperscale Vector RAG Demo ===\n")

    args = parse_args()

    global _provider_override, _model_override
    _provider_override = args.embedding_provider
    _model_override = args.embedding_model

    prompt = resolve_inputs(args)

    cfg = load_config_from_args(args)
    cluster = get_cluster()

    LOG.info("Computing embedding")
    embedding = compute_embedding(prompt)
    preflight_dimensions(len(embedding), cfg.collection)

    LOG.info("Running hyperscale vector query")
    chunks = run_hyperscale_query(
        cluster=cluster,
        cfg=cfg,
        query_embedding=embedding
    )

    if not chunks:
        print("\nNo matching context found.")
        return

    context = "\n\n---\n\n".join(chunks)

    print("\n=== Context to Augment with ===\n")
    print(context)

    answer = generate_with_llm(prompt, context)

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()