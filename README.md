# Email Demo — Couchbase Composite Vector Query

This demo illustrates how vector search and Couchbase's Composite Vector Query can be combined to improve relevance and performance for RAG-style workflows by applying standard (non-vector) filters alongside k-NN vector searches.

Dataset
 - The demo uses the Kaggle dataset: https://www.kaggle.com/datasets/anuranroy/hunter-biden-mails
 - The dataset contains fields such as `Sender`, `Receivers`, and `Contents`. `Contents` holds the email text used to compute embedding vectors.

Why composite vector queries?
 - A plain vector search (k-NN over all vectors) finds semantically similar documents but cannot efficiently restrict the search by structured attributes (for example, author or recipient).
 - A composite vector query lets you apply standard filters (e.g., `sender == X`) and then run a k-NN search only within that filtered subset, reducing noise and improving both accuracy and performance.

Example use case
 - Task: "Write an email in the style of person X." Without composite filtering, a k-NN search would consider all emails (including those not written by X). With a composite vector query, you can constrain results to `sender == X`, so the semantic search only examines emails authored by X.
    - Beyond emails, other use cases where Composite Vector Search might apply:
    - *Legal eDiscovery*: Restrict by case_id, jurisdiction, or date_range then run vector search to find semantically relevant clauses or communications.
    - *Customer Support Triage*: Filter by product, customer_id, or priority and retrieve semantically similar past tickets/solutions.
    - *Medical Records Retrieval*: Limit by patient_id, visit_date, or specialty and perform semantic retrieval over clinical notes.
- A task that is better suited by needing to examine an entire body of text might be better served by a Couchbase Hyperscale Vector index. For instance:
    - A collection of car owner's manuals: What are the most commons causes of muffler failure in vehicles?
    - Coding copilot: examine all code repositories with a similar purpose to help generate a function
    - Research/writing: examine all books for a related topic to generate a paragraph

What this repo contains
 - `data.json`: sample or preprocessed data used by the demo (email rows and metadata).
 - (Future) code snippets demonstrating how to index embeddings in Couchbase and run composite vector queries.

# Getting started

The data must first be loaded into Couchbase. The script `load.py` will load a given number of emails into Couchbase, giving them embeddings with the specified model (configuration in `.env`).

Alternatively, the embeddings can be automatically generated on the fly with Capella AI Services with a [Process and Vectorize Unstructed Data workflow](https://docs.couchbase.com/ai/build/vectorization-service/vectorize-structured-data-capella.html). This approach will greatly simplify your AI application development, and separates the data processing from your application code.
