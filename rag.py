#!/usr/bin/env python3
"""
Interactive RAG demo using Couchbase Composite Vector Queries.

Vector embeddings:
- Uses OpenAI embeddings if OPENAI_API_KEY is set
- Falls back to local sentence-transformers if not

LLM generation:
- Always uses OpenAI
- Fails fast with a helpful message if OPENAI_API_KEY is missing
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load env vars early
load_dotenv()

LOG = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------

def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        LOG.error("Required environment variable %s is not set", name)
        sys.exit(1)
    return val


# ------------------------------------------------------------
# Embedding helpers (OpenAI OR local)
# ------------------------------------------------------------

_openai_client = None
_st_model = None

def compute_embedding(text: str) -> List[float]:
    """
    Compute a vector embedding for the given text.

    Order:
    1. SentenceTransformer (only if SENTENCE_TRANSFORMER_MODEL is set)
    2. OpenAI embeddings fallback

    Caches models so repeated calls are fast.
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
            LOG.warning(
                "SentenceTransformer failed, falling back to OpenAI: %s", e
            )

    openai_api_key = get_env("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "No embedding provider available: "
            "set SENTENCE_TRANSFORMER_MODEL or OPENAI_API_KEY"
        )

    if _openai_client is None:
        _openai_client = OpenAI(api_key=openai_api_key)

    model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    resp = _openai_client.embeddings.create(
        model=model,
        input=text,
    )
    return resp.data[0].embedding


# ------------------------------------------------------------
# OpenAI LLM (required)
# ------------------------------------------------------------

def generate_with_llm(prompt: str, context: str) -> str:
    """
    Generate a response using retrieved context.
    Uses OpenAI Chat Completions (openai >= 1.0).
    """
    openai_api_key = get_env("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for generation")

    client = OpenAI(api_key=openai_api_key)

    model = get_env("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    system_prompt = (
        "You are an assistant answering questions using only the provided context. "
        "If the context does not contain enough information, say so clearly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Context:
{context}

Prompt:
{prompt}""",
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------
# Couchbase helpers
# ------------------------------------------------------------

def get_collection():
    from couchbase.cluster import Cluster, ClusterOptions
    from couchbase.auth import PasswordAuthenticator

    conn_str = get_env("COUCHBASE_CONNSTR", "couchbase://127.0.0.1")
    username = get_env("COUCHBASE_USERNAME", "Administrator")
    password = get_env("COUCHBASE_PASSWORD", "password")

    bucket = get_env("COUCHBASE_BUCKET", "default")
    scope = get_env("COUCHBASE_SCOPE", "_default")
    collection = get_env("COUCHBASE_COLLECTION", "_default")

    cluster = Cluster(
        conn_str,
        ClusterOptions(PasswordAuthenticator(username, password)),
    )

    coll = cluster.bucket(bucket).scope(scope).collection(collection)

    return {
        "cluster": cluster,
        "bucket": bucket,
        "scope": scope,
        "collection": collection,
        "collection_obj": coll,
    }


# ------------------------------------------------------------
# Composite vector query
# ------------------------------------------------------------

def run_composite_query(
    cluster,
    bucket: str,
    scope: str,
    collection: str,
    query_embedding: List[float],
    sender: str | None,
    receiver: str | None,
    limit: int = 5,
) -> List[str]:
    """
    Run a composite vector query:
    - Structured filters (sender / receiver)
    - k-NN over embedding
    """
    where_clauses = []
    params = {
        "vector": query_embedding,
        "limit": limit,
    }

    # sender is a scalar object, not an array
    if sender:
        where_clauses.append(" ANY s IN e.sender SATISFIES s == $sender END ")
        params["sender"] = sender

    # receivers is an object with arrays (to/cc/bcc)
    if receiver:
        where_clauses.append(
            """
            (
              ANY r IN e.receivers.to SATISFIES r == $receiver END
              OR ANY r IN e.receivers.cc SATISFIES r == $receiver END
              OR ANY r IN e.receivers.bcc SATISFIES r == $receiver END
            )
            """
        )
        params["receiver"] = receiver

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    statement = f"""
    SELECT RAW e.contents
    FROM `{bucket}`.`{scope}`.`{collection}` e
    {where_sql}
    ORDER BY APPROX_VECTOR_DISTANCE(e.embedding, $vector, "COSINE")
    LIMIT $limit
    """

    rows = cluster.query(statement,**params)

    return [row for row in rows]


# ------------------------------------------------------------
# Interactive CLI
# ------------------------------------------------------------

def main():
    print("\n=== Couchbase Composite Vector RAG Demo ===\n")

    sender = input('Filter by sender (exact match): ').strip() or None
    receiver = input("Filter by receiver field (exact match): ").strip() or None

    prompt = input("\nEnter your prompt:\n> ").strip()
    if not prompt:
        print("Prompt required")
        return

    LOG.info("Computing embedding for prompt")
    embedding = compute_embedding(prompt)

    cb = get_collection()

    LOG.info("Running composite vector search")
    context_chunks = run_composite_query(
        cluster=cb["cluster"],
        bucket=cb["bucket"],
        scope=cb["scope"],
        collection=cb["collection"],
        query_embedding=embedding,
        sender=sender,
        receiver=receiver,
        limit=5,
    )

    if not context_chunks:
        print("\nNo matching context found.")
        return

    context = "\n\n---\n\n".join(context_chunks)

    response = generate_with_llm(prompt, context)

    print("\n=== Answer ===\n")
    print(response)


if __name__ == "__main__":
    main()
