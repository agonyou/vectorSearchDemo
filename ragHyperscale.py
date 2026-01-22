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
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

load_dotenv()

LOG = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        LOG.error("Required environment variable %s is not set", name)
        sys.exit(1)
    return val


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

_openai_client = None
_st_model = None


def compute_embedding(text: str) -> List[float]:
    """
    Compute an embedding using:
    1) SentenceTransformer (if configured)
    2) OpenAI fallback
    """
    global _st_model, _openai_client

    st_model_name = get_env("SENTENCE_TRANSFORMER_MODEL")

    if st_model_name:
        try:
            if _st_model is None:
                LOG.info("Loading SentenceTransformer: %s", st_model_name)
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(st_model_name)

            return _st_model.encode(text).tolist()
        except Exception as e:
            LOG.warning("SentenceTransformer failed, falling back to OpenAI: %s", e)

    api_key = get_env("OPENAI_API_KEY", required=True)

    if _openai_client is None:
        _openai_client = OpenAI(api_key=api_key)

    model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    resp = _openai_client.embeddings.create(
        model=model,
        input=text,
    )

    return resp.data[0].embedding


# ------------------------------------------------------------
# LLM Generation
# ------------------------------------------------------------

def generate_with_llm(prompt: str, context: str) -> str:
    api_key = get_env("OPENAI_API_KEY", required=True)
    client = OpenAI(api_key=api_key)

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
    from couchbase.cluster import Cluster, ClusterOptions
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

    return parser.parse_args()


def resolve_inputs(args):
    any_cli_inputs = any([args.prompt])

    if any_cli_inputs:
        if not args.prompt:
            LOG.error("When using CLI mode, --prompt is required")
            sys.exit(1)
        return args.prompt

    # Interactive user inputs
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
    prompt = resolve_inputs(args)

    cfg = load_config_from_args(args)
    cluster = get_cluster()

    LOG.info("Computing embedding")
    embedding = compute_embedding(prompt)

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
