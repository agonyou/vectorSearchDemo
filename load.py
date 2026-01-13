#!/usr/bin/env python3
"""Simple loader to compute embeddings for email `Contents` and upsert into Couchbase.

Supports OpenAI (if `OPENAI_API_KEY` set) or local `sentence-transformers` as a fallback.

Usage:
  python load.py --data data.json

Set connection/config in a `.env` file (see `.env` template).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from typing import List

from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    if v is None:
        LOG.debug("env %s not set", name)
    return v


def compute_embedding(text: str) -> List[float]:
    """Compute an embedding using OpenAI (if configured) or sentence-transformers fallback."""
    openai_api_key = get_env("OPENAI_API_KEY")
    if openai_api_key:
        try:
            import openai

            openai.api_key = openai_api_key
            model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            resp = openai.Embedding.create(model=model, input=text)
            return resp["data"][0]["embedding"]
        except Exception as e:
            LOG.warning("OpenAI embedding failed, falling back to local: %s", e)

    # Fallback to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        model_name = get_env("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        emb = model.encode(text)
        return emb.tolist()
    except Exception as e:
        LOG.error("No embedding provider available: %s", e)
        raise


def upsert_documents(docs: list[dict], dry_run: bool = False) -> None:
    """Upsert documents into Couchbase with an `embedding` field."""
    # Lazy import couchbase to avoid hard dependency at module import
    try:
        from couchbase.cluster import Cluster, ClusterOptions
        from couchbase.auth import PasswordAuthenticator
    except Exception as e:
        LOG.error("Couchbase SDK not available: %s", e)
        raise

    conn_str = get_env("COUCHBASE_CONNSTR", "couchbase://127.0.0.1")
    username = get_env("COUCHBASE_USERNAME", "Administrator")
    password = get_env("COUCHBASE_PASSWORD", "password")
    bucket_name = get_env("COUCHBASE_BUCKET", "default")
    scope_name = get_env("COUCHBASE_SCOPE")
    coll_name = get_env("COUCHBASE_COLLECTION")

    cluster = Cluster(conn_str, ClusterOptions(PasswordAuthenticator(username, password)))
    bucket = cluster.bucket(bucket_name)

    if scope_name and coll_name:
        collection = bucket.scope(scope_name).collection(coll_name)
    else:
        collection = bucket.default_collection()

    LOG.info("Upserting %d documents (dry_run=%s)", len(docs), dry_run)
    for doc in docs:
        doc_id = doc.get("id") or str(uuid.uuid4())
        if dry_run:
            LOG.info("DRY UPsert %s -> sender=%s contents_len=%d", doc_id, doc.get("sender"), len(doc.get("contents", "")))
            continue
        try:
            collection.upsert(doc_id, doc)
        except Exception as e:
            LOG.error("Failed to upsert %s: %s", doc_id, e)


def normalize_row(row: dict) -> dict:
    # Accept different key casings
    sender = row.get("Sender") or row.get("sender") or row.get("From") or row.get("from")
    receivers = row.get("Receivers") or row.get("receivers") or row.get("To") or row.get("to")
    contents = row.get("Contents") or row.get("contents") or row.get("body") or row.get("text") or ""
    return {"sender": sender, "receivers": receivers, "contents": contents}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json", help="Path to input JSON file (array of rows)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Couchbase; just show actions")
    args = parser.parse_args()

    path = args.data
    if not os.path.exists(path):
        LOG.error("Data file not found: %s", path)
        return

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        LOG.error("Expected JSON array of rows in %s", path)
        return

    prepared = []
    for i, row in enumerate(raw):
        nr = normalize_row(row)
        contents = nr["contents"] or ""
        try:
            emb = compute_embedding(contents)
        except Exception:
            LOG.exception("Embedding failed for row %d, skipping", i)
            continue

        doc = {
            "id": row.get("id") or f"email::{i}",
            "sender": nr["sender"],
            "receivers": nr["receivers"],
            "contents": contents,
            "embedding": emb,
        }
        prepared.append(doc)

    upsert_documents(prepared, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
