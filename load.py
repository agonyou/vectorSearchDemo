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
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Dict, Iterable, List

from dotenv import load_dotenv

# Load .env early (but NOT bucket/scope/collection)
load_dotenv()

# The embedding model is already cached on disk. Force HuggingFace fully offline so
# sentence-transformers does NOT revalidate the cache over the network on every load
# (that per-embed HEAD storm is what triggers HTTP 429 rate-limiting + long backoffs).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOG = logging.getLogger("loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def truncate(text: str, max_chars: int = 8000) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _chunked(seq, size):
    """Yield successive `size`-length lists from a sequence/iterator."""
    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


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

_ST_MODEL = None       # cache the SentenceTransformer across calls — load it ONCE, not per document
_OPENAI_CLIENT = None  # cache the OpenAI client too

# CLI overrides for the embedding provider ("local"/"openai") and model name. None → infer from
# .env. Set from --embedding-provider / --embedding-model in main().
_PROVIDER_OVERRIDE = None
_MODEL_OVERRIDE = None


def resolve_provider() -> str:
    """Effective embedding provider: the --embedding-provider override if given, else inferred
    from .env (SENTENCE_TRANSFORMER_MODEL set → 'local', otherwise 'openai')."""
    if _PROVIDER_OVERRIDE:
        return _PROVIDER_OVERRIDE
    return "local" if get_env("SENTENCE_TRANSFORMER_MODEL") else "openai"


def st_model_name() -> str:
    """Local model name — --embedding-model if given, else SENTENCE_TRANSFORMER_MODEL, else the
    documented 384-dim default."""
    return _MODEL_OVERRIDE or get_env("SENTENCE_TRANSFORMER_MODEL") or "all-MiniLM-L6-v2"


def openai_embedding_model() -> str:
    """OpenAI embedding model — --embedding-model if given, else OPENAI_EMBEDDING_MODEL, else default."""
    return _MODEL_OVERRIDE or get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _get_st_model(name: str):
    """Lazily construct and cache the embedding model. Constructing SentenceTransformer is what
    hits the HF hub, so doing it once (not per row) is the main fix for the ingest stall."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(name)
    return _ST_MODEL


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI(
            api_key=get_env("OPENAI_API_KEY"),
            base_url=get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return _OPENAI_CLIENT


def using_sentence_transformers() -> bool:
    return resolve_provider() == "local"


def compute_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed a LIST of texts in one shot. Uses the resolved provider (local sentence-transformers
    falls back to OpenAI only on error). Batching is the main speedup: one encode()/one API call
    per batch instead of one per document."""
    if resolve_provider() == "local":
        try:
            arr = _get_st_model(st_model_name()).encode(
                texts,
                batch_size=min(64, len(texts)) or 1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in arr]
        except Exception as e:
            LOG.warning("SentenceTransformer batch failed, falling back to OpenAI: %s", e)

    if not get_env("OPENAI_API_KEY"):
        raise RuntimeError("No embedding provider configured")

    client = _get_openai_client()
    model = openai_embedding_model()
    resp = client.embeddings.create(model=model, input=texts)
    # Return in input order (OpenAI echoes an `index` on each item).
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def compute_embedding(text: str) -> List[float]:
    """Single-text convenience wrapper (kept for compatibility)."""
    return compute_embeddings([text])[0]


def preflight_dimensions(actual_dim: int, collection: str) -> None:
    """Fail fast when the embedding dimension disagrees with VECTOR_DIMENSIONS (.env).
    Loading 384-dim vectors into a collection whose index expects 1536 (or vice-versa)
    otherwise fails silently at query time with an empty result."""
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
            "Embedding dimension mismatch: provider '%s' produces %d-dim vectors but "
            "VECTOR_DIMENSIONS=%d (loading into collection '%s'). The provider, the collection's "
            "vector index, and VECTOR_DIMENSIONS must all agree (local/MiniLM=384, OpenAI=1536).",
            resolve_provider(), actual_dim, declared_dim, collection,
        )
        raise SystemExit(1)


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

    # Embedding provider override (see #1). Omit to keep the .env-driven default.
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai"],
        default=None,
        help="Force the embedding provider. Omit to infer from .env "
             "(SENTENCE_TRANSFORMER_MODEL set → local, else openai). The dimension it "
             "produces (local/MiniLM=384, openai=1536) must match the target collection's index.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model name for the chosen provider (local sentence-transformers "
             "model or OpenAI embedding model). Its output dimension must match the target index.",
    )

    # Throughput controls
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents embedded + upserted per batch (one encode()/API call per batch)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Batches processed in parallel. Honored for the OpenAI path (network-bound); "
             "the local sentence-transformers path runs batches sequentially (CPU-bound, GIL) "
             "but still gets the batched-encode + multi-upsert speedup.",
    )

    args = parser.parse_args()

    global _PROVIDER_OVERRIDE, _MODEL_OVERRIDE
    _PROVIDER_OVERRIDE = args.embedding_provider
    _MODEL_OVERRIDE = args.embedding_model

    rows = load_rows(args.data)
    collection = get_collection(args.bucket, args.scope, args.collection)

    # ---- Stage 1: build work items (deterministic id + embedding text) -----------------
    work = []  # list of (doc_id, contents, row)
    bad_ids = 0
    for i, row in enumerate(rows):
        try:
            doc_id = "::".join(str(row[f]) for f in args.id_field)
        except KeyError as e:
            LOG.warning("Row %d missing id-field %s, skipping", i, e)
            bad_ids += 1
            continue
        contents = truncate(" ".join(str(row.get(f, "") or "") for f in args.text_fields))
        work.append((doc_id, contents, row))

    # ---- Stage 2: drop already-loaded docs (idempotent) via batched exists_multi --------
    skipped = 0
    pending = []
    for chunk in _chunked(work, 500):
        ids = [w[0] for w in chunk]
        try:
            res = collection.exists_multi(ids)
            results = getattr(res, "results", res)  # {id: ExistsResult}
        except Exception:
            results = {}  # if the bulk check fails, fall through and let upsert overwrite
        for w in chunk:
            er = results.get(w[0]) if isinstance(results, dict) else None
            if er is not None and getattr(er, "exists", False):
                skipped += 1
            else:
                pending.append(w)

    if args.limit:
        pending = pending[: args.limit]

    total = len(pending)
    LOG.info(
        "Planning: %d rows, %d already loaded, %d bad-id → %d to embed (batch=%d, concurrency=%d)",
        len(work), skipped, bad_ids, total, args.batch_size, args.concurrency,
    )

    # ---- Stage 3: embed + upsert in batches --------------------------------------------
    def build_docs(batch):
        texts = [contents for (_id, contents, _row) in batch]
        vectors = compute_embeddings(texts)
        out = {}
        for (doc_id, contents, row), vec in zip(batch, vectors):
            doc = {f: row.get(f) for f in args.copy_fields}
            doc["contents"] = contents
            doc["embedding"] = vec
            maybe_add_location(doc, row)
            out[doc_id] = doc
        return out

    def process_batch(batch):
        try:
            docs = build_docs(batch)
        except Exception:
            LOG.exception("Embedding failed for a batch of %d (skipped)", len(batch))
            return 0
        if args.dry_run:
            first_id = next(iter(docs))
            log_safe_doc(first_id, docs[first_id])
            return len(docs)
        try:
            collection.upsert_multi(docs)
        except Exception:
            LOG.exception("Upsert failed for a batch of %d (skipped)", len(docs))
            return 0
        return len(docs)

    batches = list(_chunked(pending, args.batch_size))
    loaded = 0

    # Warm the provider once before fan-out so lazy singletons don't race across threads.
    if total:
        if using_sentence_transformers():
            _get_st_model(st_model_name())
        elif get_env("OPENAI_API_KEY"):
            _get_openai_client()
        # Preflight: confirm the provider's output dimension matches VECTOR_DIMENSIONS
        # before writing any vectors (a mismatch is otherwise silent at query time).
        preflight_dimensions(len(compute_embeddings(["dimension probe"])[0]), args.collection)

    # Local ST is CPU/GIL-bound → sequential batches. OpenAI is network-bound → parallel.
    workers = 1 if using_sentence_transformers() else max(1, args.concurrency)

    if workers == 1:
        for bi, batch in enumerate(batches, 1):
            loaded += process_batch(batch)
            LOG.info("batch %d/%d  loaded=%d/%d", bi, len(batches), loaded, total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_batch, b): idx for idx, b in enumerate(batches, 1)}
            done = 0
            for fut in as_completed(futures):
                loaded += fut.result()
                done += 1
                LOG.info("batch %d/%d done  loaded=%d/%d", done, len(batches), loaded, total)

    LOG.info(
        "Done. Loaded=%d skipped(existing)=%d bad-id=%d limit=%s",
        loaded, skipped, bad_ids, args.limit,
    )


if __name__ == "__main__":
    main()