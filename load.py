#!/usr/bin/env python3
"""
Simple loader to compute embeddings for email-like JSON rows
and upsert them into Couchbase.

Key behaviors:
- Uses `timestamp` from each JSON row as the Couchbase document key
- Skips documents that already exist
- Supports loading only N *new* documents via --limit
- Computes embeddings only for documents that will actually be written

Usage:
  python load.py --data data.json
  python load.py --data data.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env early
load_dotenv()

LOG = logging.getLogger("loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def log_safe_doc(doc_id: str, doc: dict, *, contents_max=200, embedding_max=8) -> None:
    """
    Log a human-readable, truncated version of a document for debugging.

    - contents: truncated to `contents_max` characters
    - embedding: truncated to first `embedding_max` floats
    """
    safe = dict(doc)

    if "contents" in safe and isinstance(safe["contents"], str):
        if len(safe["contents"]) > contents_max:
            safe["contents"] = safe["contents"][:contents_max] + "…"

    if "embedding" in safe and isinstance(safe["embedding"], list):
        safe["embedding"] = safe["embedding"][:embedding_max]
        safe["embedding_truncated"] = True
        safe["embedding_dim"] = len(doc["embedding"])

    LOG.info(
        "doc_id=%s\n%s",
        doc_id,
        json.dumps(safe, indent=2, ensure_ascii=False),
    )


def get_env(name: str, default: str | None = None) -> str | None:
    """
    Small helper so env access is consistent and centralized.
    """
    return os.getenv(name, default)

def compute_embedding(text: str) -> List[float]:
    """
    Compute a vector embedding for the given text.

    Order of preference:
    1. Local sentence-transformers (if SENTENCE_TRANSFORMER_MODEL is set)
    2. OpenAI embeddings fallback (if OPENAI_API_KEY is set)

    Provider details are intentionally hidden so swapping models is trivial.
    """

    st_model_name = get_env("SENTENCE_TRANSFORMER_MODEL")

    if st_model_name:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(st_model_name)
            return model.encode(text).tolist()
        except Exception as e:
            LOG.warning(
                "SentenceTransformer embedding failed, falling back to OpenAI: %s",
                e,
            )

    openai_api_key = get_env("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "No embedding provider available: "
            "set SENTENCE_TRANSFORMER_MODEL or OPENAI_API_KEY"
        )

    import openai

    openai.api_key = openai_api_key
    model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    resp = openai.Embedding.create(
        model=model,
        input=text,
    )
    return resp["data"][0]["embedding"]


def get_collection():
    """
    Create and return a Couchbase collection.

    Uses:
    - COUCHBASE_CONNSTR
    - COUCHBASE_USERNAME
    - COUCHBASE_PASSWORD
    - COUCHBASE_BUCKET
    - Optional COUCHBASE_SCOPE / COUCHBASE_COLLECTION

    Keeping this in one place avoids connection logic being
    scattered across the script.
    """
    from couchbase.cluster import Cluster, ClusterOptions
    from couchbase.auth import PasswordAuthenticator

    conn_str = get_env("COUCHBASE_CONNSTR", "couchbase://127.0.0.1")
    username = get_env("COUCHBASE_USERNAME", "Administrator")
    password = get_env("COUCHBASE_PASSWORD", "password")
    bucket_name = get_env("COUCHBASE_BUCKET", "default")
    scope_name = get_env("COUCHBASE_SCOPE")
    coll_name = get_env("COUCHBASE_COLLECTION")

    cluster = Cluster(
        conn_str,
        ClusterOptions(PasswordAuthenticator(username, password)),
    )

    bucket = cluster.bucket(bucket_name)

    # Prefer named scope/collection when provided
    if scope_name and coll_name:
        return bucket.scope(scope_name).collection(coll_name)

    return bucket.default_collection()


def normalize_row(row: dict) -> dict:
    """
    Normalize incoming JSON rows so downstream code does not care
    about field casing or alternate names.

    This allows mixed or messy source data without spreading
    conditionals everywhere.
    """
    return {
        "sender": row.get("Sender") or row.get("sender") or row.get("From") or row.get("from"),
        "receivers": row.get("Receivers") or row.get("receivers") or row.get("To") or row.get("to"),
        "contents": row.get("Contents") or row.get("contents") or row.get("body") or row.get("text") or "",
        # timestamp is required because it becomes the Couchbase document key
        "timestamp": row.get("timestamp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json", help="Path to JSON array file")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Couchbase")
    parser.add_argument(
        "--limit",
        type=int,
        help="Load only N new (previously unseen) documents",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        LOG.error("Data file not found: %s", args.data)
        return

    # Load the entire JSON file into memory
    with open(args.data, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        LOG.error("Expected JSON array of rows")
        return

    collection = get_collection()

    loaded = 0
    skipped = 0

    # Iterate in file order; "first N" is deterministic
    for i, row in enumerate(raw):
        nr = normalize_row(row)

        # timestamp is mandatory since it becomes the document ID
        if nr["timestamp"] is None:
            LOG.warning("Row %d missing timestamp, skipping", i)
            continue

        doc_id = str(nr["timestamp"])

        # Skip documents that already exist to keep loads idempotent
        if collection.exists(doc_id).exists:
            skipped += 1
            continue

        contents = nr["contents"] or ""

        # Only compute embeddings for documents we will actually write
        try:
            embedding = compute_embedding(contents)
        except Exception:
            LOG.exception("Embedding failed for row %d, skipping", i)
            continue

        doc = {
            "sender": nr["sender"],
            "receivers": nr["receivers"],
            "contents": contents,
            "timestamp": nr["timestamp"],
            "embedding": embedding,
        }

        if args.dry_run:
            # LOG.info("DRY upsert %s (contents_len=%d)", doc_id, len(contents))
            LOG.info(log_safe_doc(doc_id, doc))
        else:
            collection.upsert(doc_id, doc)
            LOG.info("Upserted %s", doc_id)

        loaded += 1

        # Stop once we've loaded N new documents
        if args.limit and loaded >= args.limit:
            break

    LOG.info(
        "Done. Loaded=%d, skipped(existing)=%d, limit=%s",
        loaded,
        skipped,
        args.limit,
    )


if __name__ == "__main__":
    main()
