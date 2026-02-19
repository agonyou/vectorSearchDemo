#!/usr/bin/env python3
"""
Generic CSV / JSON loader that:
- Builds embeddings from selected fields
- Upserts into a specified Couchbase bucket/scope/collection
- Skips existing documents (idempotent)
- Computes embeddings only when needed

Bucket, scope, and collection MUST be provided via CLI.
All other Couchbase and embedding config comes from .env.

# Minimal required flags (JSON)
python load.py --data data.json --id-field id --text-fields contents --bucket mybucket --scope myschema --collection mydocs

# Multiple text fields concatenated
python load.py --data emails.json --id-field timestamp --text-fields subject body --bucket datenight --scope email --collection messages

# CSV with copied metadata fields
python load.py --data tickets.csv --id-field ticket_id --text-fields title description --copy-fields status priority created_at --bucket support --scope ops --collection tickets

# Dry run (no writes)
python load.py --data data.json --id-field id --text-fields text --bucket test --scope sandbox --collection dryrun --dry-run

# Limit number of new documents loaded
python load.py --data large.csv --id-field record_id --text-fields summary notes --bucket demo --scope sample --collection limited --limit 25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Dict, Iterable, List

from dotenv import load_dotenv

# Load .env early (but NOT bucket/scope/collection)
load_dotenv()

LOG = logging.getLogger("loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def truncate(text: str, max_chars: int = 8000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def log_safe_doc(doc_id: str, doc: dict) -> None:
    safe = dict(doc)
    safe["embedding"] = safe["embedding"][:8]
    safe["embedding_dim"] = len(doc["embedding"])
    safe["contents"] = truncate(safe["contents"], 200) + "…"
    LOG.info("doc_id=%s\n%s", doc_id, json.dumps(safe, indent=2))

# to handle data sets (like Yelp) where latitude and longitude are separate root fields
def maybe_add_location(doc: dict, row: dict) -> None:
    """
    If row contains lat/lon or latitude/longitude at root,
    add a derived 'location' geopoint to doc.
    """
    lat = row.get("lat") or row.get("latitude")
    lon = row.get("lon") or row.get("longitude")

    try:
        if "location" not in doc and lat is not None and lon is not None:
            doc["location"] = {
                "lat": float(lat),
                "lon": float(lon),
            }
    except (TypeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def compute_embedding(text: str) -> List[float]:
    """Prefer local sentence-transformers, fall back to OpenAI."""
    st_model = get_env("SENTENCE_TRANSFORMER_MODEL")
    if st_model:
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(st_model).encode(text).tolist()
        except Exception as e:
            LOG.warning("SentenceTransformer failed, falling back: %s", e)

    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No embedding provider configured")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    return client.embeddings.create(
        model=model,
        input=text,
    ).data[0].embedding


# ---------------------------------------------------------------------------
# Couchbase
# ---------------------------------------------------------------------------

def get_collection(bucket: str, scope: str, collection: str):
    """
    Resolve and validate bucket / scope / collection using the management API.
    Terminates with a clear error message if anything does not exist.
    """
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions
    from couchbase.auth import PasswordAuthenticator
    from couchbase.exceptions import CouchbaseException

    cluster = Cluster(
        get_env("COUCHBASE_CONNSTR", "couchbase://127.0.0.1"),
        ClusterOptions(
            PasswordAuthenticator(
                get_env("COUCHBASE_USERNAME", "Administrator"),
                get_env("COUCHBASE_PASSWORD", "password"),
            )
        ),
    )

    try:
        cb_bucket = cluster.bucket(bucket)
        scopes = cb_bucket.collections().get_all_scopes()
    except CouchbaseException as e:
        LOG.error("Failed to access bucket '%s': %s", bucket, e)
        raise SystemExit(1)

    scope_names = {s.name: s for s in scopes}

    if scope not in scope_names:
        LOG.error(
            "Couchbase scope does not exist: '%s' (bucket: '%s')",
            scope,
            bucket,
        )
        raise SystemExit(1)

    collection_names = {c.name for c in scope_names[scope].collections}

    if collection not in collection_names:
        LOG.error(
            "Couchbase collection does not exist: '%s' (bucket: '%s', scope: '%s')",
            collection,
            bucket,
            scope,
        )
        raise SystemExit(1)

    # Safe: existence confirmed via management API
    return cb_bucket.scope(scope).collection(collection)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(path: str) -> Iterable[Dict]:
    """
    Load rows from:
    - JSON array
    - newline-delimited JSON (JSONL / NDJSON)
    - CSV
    """
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            first_char = fh.read(1)
            fh.seek(0)

            # JSON array
            if first_char == "[":
                data = json.load(fh)
                if not isinstance(data, list):
                    raise ValueError("JSON array expected")
                return data

            # JSON Lines / NDJSON
            return [
                json.loads(line)
                for line in fh
                if line.strip()
            ]

    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return [
                json.loads(line)
                for line in fh
                if line.strip()
            ]

    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    raise ValueError("Unsupported file type (use .csv, .json, or .jsonl)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    # Data + mapping
    parser.add_argument("--data", required=True, help="CSV or JSON file")
    parser.add_argument(
        "--id-field",
        nargs="+",
        required=True,
        help="One or more fields that form the document ID (joined with '::')",
    )
    parser.add_argument(
        "--text-fields",
        nargs="+",
        required=True,
        help="Fields used to build embedding text",
    )
    parser.add_argument(
        "--copy-fields",
        nargs="*",
        default=[],
        help="Fields copied verbatim into the document",
    )

    # Couchbase target (REQUIRED)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--collection", required=True)

    # Control flags
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    rows = load_rows(args.data)
    collection = get_collection(args.bucket, args.scope, args.collection)

    loaded = skipped = 0

    for i, row in enumerate(rows):
        # Build deterministic document ID
        try:
            doc_id = "::".join(str(row[f]) for f in args.id_field)
        except KeyError as e:
            LOG.warning("Row %d missing id-field %s, skipping", i, e)
            continue

        # Skip existing docs → idempotent
        if collection.exists(doc_id).exists:
            skipped += 1
            continue

        contents = truncate(
            " ".join(str(row.get(f, "") or "") for f in args.text_fields)
        )

        try:
            embedding = compute_embedding(contents)
        except Exception:
            LOG.exception("Embedding failed for row %d", i)
            continue

        doc = {f: row.get(f) for f in args.copy_fields}
        doc["contents"] = contents
        doc["embedding"] = embedding
        maybe_add_location(doc, row)

        if args.dry_run:
            log_safe_doc(doc_id, doc)
        else:
            collection.upsert(doc_id, doc)
            LOG.info("Upserted %s", doc_id)

        loaded += 1
        if args.limit and loaded >= args.limit:
            break

    LOG.info(
        "Done. Loaded=%d skipped(existing)=%d limit=%s",
        loaded,
        skipped,
        args.limit,
    )


if __name__ == "__main__":
    main()