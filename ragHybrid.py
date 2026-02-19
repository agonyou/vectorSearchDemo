#!/usr/bin/env python3
"""
Interactive or CLI-driven RAG demo using Couchbase Hybrid Vector Queries.
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

from couchbase import search
from couchbase.search import SearchRequest, MatchAllQuery, SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import (
    SearchRequest,
    GeoDistanceQuery
)
from couchbase.exceptions import DocumentNotFoundException

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

    latitude: float
    longitude: float
    radius_miles: str

    index_name: str

    embedding_field: str = "embedding"
    content_field: str = "contents"
    limit: int = 5


def load_config_from_args(args, inputs) -> RAGConfig:
    return RAGConfig(
        bucket=args.bucket,
        scope=args.scope,
        collection=args.collection,
        latitude=float(inputs["latitude"]),
        longitude=float(inputs["longitude"]),
        radius_miles=inputs["radius"],
        index_name=args.index_name,
        limit=args.limit,
    )


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

_openai_client = None
_st_model = None


def compute_embedding(text: str) -> List[float]:
    global _st_model, _openai_client

    st_model_name = get_env("SENTENCE_TRANSFORMER_MODEL")

    if st_model_name:
        try:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(st_model_name)
            return _st_model.encode(text).tolist()
        except Exception as e:
            LOG.warning("SentenceTransformer failed, falling back: %s", e)

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

    resp = client.chat.completions.create(
        model=model,
        messages=[
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
        ],
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


def run_hybrid_query(cluster, cfg: RAGConfig, query_embedding: List[float]) -> List[str]:
    scope = cluster.bucket(cfg.bucket).scope(cfg.scope)
    collection = scope.collection(cfg.collection) 

    #cfg.latitude, cgf.longitude, cfg.radius_miles

    geo_filter = GeoDistanceQuery(
        location=(cfg.longitude, cfg.latitude),  # (lon, lat)
        distance=cfg.radius_miles,
        field="location"
    )

    vector_query = VectorQuery.create(
        field_name="embedding",
        vector=query_embedding,
        num_candidates=3,
        prefilter=geo_filter
    )

    vector_search = VectorSearch.from_vector_query(vector_query)

    request = SearchRequest.create(vector_search)

    search_result = scope.search(
        cfg.index_name,
        request
    )

    # ---- collect FTS hits ----
    hits = []

    for hit in search_result.rows():
        hits.append({
            "id": hit.id,
            "score": hit.score
        })

    if not hits:
        return []

    # ---- KV fetch (one-by-one, SDK-correct) ----
    enriched = []

    for h in hits:
        doc_id = h["id"]
        try:
            res = collection.get(doc_id)
            doc = res.content_as[dict]
        except DocumentNotFoundException:
            continue

        enriched.append({
            "id": doc_id,
            "score": h["score"],
            "name": doc.get("name"),
            "contents": doc.get("contents")
        })

    return enriched

# ------------------------------------------------------------
# CLI / Input Handling
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Vector RAG Demo")

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--collection", required=True)

    parser.add_argument("--index-name", default="ix-yelp-business-vector")
    parser.add_argument("--limit", type=int, default=5)

    parser.add_argument("--prompt")
    parser.add_argument("--latitude")
    parser.add_argument("--longitude")
    parser.add_argument("--radius-miles")

    return parser.parse_args()


def resolve_inputs(args):
    if args.prompt:
        missing = [x for x in ("latitude", "longitude", "radius_miles") if getattr(args, x) is None]
        if missing:
            LOG.error("When using CLI mode, --latitude, --longitude, and --radius-miles are required")
            sys.exit(1)

        return {
            "prompt": args.prompt,
            "latitude": args.latitude,
            "longitude": args.longitude,
            "radius": args.radius_miles,
        }

    prompt = input("\nEnter your prompt:\n> ").strip()
    latitude = input("\nEnter latitude:\n> ").strip()
    longitude = input("\nEnter longitude:\n> ").strip()
    radius = input("\nEnter radius (e.g. 5mi):\n> ").strip()

    if not prompt or not latitude or not longitude or not radius:
        print("Prompt, latitude, longitude, and radius are required")
        sys.exit(1)

    return {
        "prompt": prompt,
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== Couchbase Hybrid Vector RAG Demo ===\n")

    args = parse_args()
    inputs = resolve_inputs(args)

    cfg = load_config_from_args(args, inputs)
    cluster = get_cluster()

    LOG.info("Computing embedding")
    embedding = compute_embedding(inputs["prompt"])

    LOG.info("Running hybrid vector query")
    chunks = run_hybrid_query(cluster, cfg, embedding)

    if not chunks:
        print("\nNo matching context found.")
        return

    # ---- convert structured chunks to prompt-ready text ----
    formatted_chunks = []
    for c in chunks:
        name = c.get("name", "Unknown")
        contents = c.get("contents", "")
        if contents:
            formatted_chunks.append(
                f"Name: {name}\n\n{contents}"
            )

    if not formatted_chunks:
        print("\nNo usable context found.")
        return

    context = "\n\n---\n\n".join(formatted_chunks)

    print("\n=== Context to Augment with ===\n")
    print(context)

    answer = generate_with_llm(inputs["prompt"], context)

    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
