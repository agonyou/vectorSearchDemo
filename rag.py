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

_ST_MODEL = None

def compute_embedding(text: str) -> List[float]:
    """
    Compute an embedding for vector search.

    Preference order:
    1. OpenAI embeddings (if OPENAI_API_KEY is set)
    2. Local sentence-transformers
    """
    openai_key = get_env("OPENAI_API_KEY")

    if openai_key:
        try:
            import openai

            openai.api_key = openai_key
            model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

            resp = openai.Embedding.create(
                model=model,
                input=text,
            )
            return resp["data"][0]["embedding"]
        except Exception as e:
            LOG.warning("OpenAI embedding failed, falling back to local: %s", e)

    # ---- Local fallback ----
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        model_name = get_env("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        LOG.info("Loading local sentence transformer: %s (CPU)", model_name)
        _ST_MODEL = SentenceTransformer(model_name, device="cpu")

    return _ST_MODEL.encode(text).tolist()


# ------------------------------------------------------------
# OpenAI LLM (required)
# ------------------------------------------------------------

def generate_with_llm(prompt: str, context_chunks: List[str]) -> str:
    """
    Generate text using OpenAI Chat Completions.
    This ALWAYS requires OpenAI.
    """
    import openai

    openai.api_key = get_env("OPENAI_API_KEY", required=True)

    system_msg = (
        "You are an assistant helping write text based on prior emails. "
        "Use the provided email excerpts as context. "
        "Do not invent details not supported by the context."
    )

    context_text = "\n\n---\n\n".join(context_chunks)

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": f"Context emails:\n{context_text}\n\nPrompt:\n{prompt}",
        },
    ]

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )

    return resp["choices"][0]["message"]["content"]


# ------------------------------------------------------------
# Couchbase helpers
# ------------------------------------------------------------

def get_collection():
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions
    from couchbase.auth import PasswordAuthenticator

    conn_str = get_env("COUCHBASE_CONNSTR", required=True)
    username = get_env("COUCHBASE_USERNAME", required=True)
    password = get_env("COUCHBASE_PASSWORD", required=True)
    bucket_name = get_env("COUCHBASE_BUCKET", required=True)
    scope_name = get_env("COUCHBASE_SCOPE")
    coll_name = get_env("COUCHBASE_COLLECTION")

    cluster = Cluster(
        conn_str,
        ClusterOptions(PasswordAuthenticator(username, password)),
    )

    bucket = cluster.bucket(bucket_name)

    if scope_name and coll_name:
        return bucket.scope(scope_name).collection(coll_name)

    return bucket.default_collection()


# ------------------------------------------------------------
# Composite vector query
# ------------------------------------------------------------

def run_composite_query(
    collection,
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

    if sender:
        where_clauses.append(
            "ANY s IN e.sender SATISFIES s == $sender END"
        )
        params["sender"] = f"%{sender}%"

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
        params["receiver"] = f"%{receiver}%"

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    statement = f"""
    SELECT RAW e.contents
    FROM `{collection.bucket_name}` e
    {where_sql}
    ORDER BY APPROX_VECTOR_DISTANCE(e.embedding, $vector, "DOT")
    LIMIT $limit
    """

    LOG.info("Executing composite vector query")
    rows = collection.query(statement, params)

    return [row for row in rows]


# ------------------------------------------------------------
# Interactive CLI
# ------------------------------------------------------------

def main() -> None:
    print("\n=== Couchbase Composite Vector RAG Demo ===\n")

    # LLM requires OpenAI — fail fast if missing
    get_env("OPENAI_API_KEY", required=True)

    collection = get_collection()

    sender = input("Filter by sender (optional, substring match): ").strip() or None
    receiver = input("Filter by receiver (optional, substring match): ").strip() or None
    prompt = input("\nEnter your prompt:\n> ").strip()

    if not prompt:
        print("Prompt is required.")
        return

    LOG.info("Computing embedding for prompt")
    query_embedding = compute_embedding(prompt)

    LOG.info("Running composite vector search")
    context_chunks = run_composite_query(
        collection=collection,
        query_embedding=query_embedding,
        sender=sender,
        receiver=receiver,
        limit=5,
    )

    if not context_chunks:
        print("\nNo matching context found.")
        return

    LOG.info("Generating response with LLM")
    result = generate_with_llm(prompt, context_chunks)

    print("\n=== Generated Result ===\n")
    print(result)
    print("\n========================\n")


if __name__ == "__main__":
    main()
