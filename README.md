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

# Step 0: Prerequesites

Make sure you've got Python running. You'll probably want to create a virtual environment first, like this:

```bash
python3 -m venv venv
```

Then go into that venv with this command:

```bash
source venv/bin/activate
```

Then install requirments:

```bash
python -m pip install -r requirements.txt
```

At this point, you may want to go ahead and create an `.env` file, using the settings you need for your environment. Check out `.env.sample` for an example.

Now you're ready to start loading data.

# Step 1: Loading the data

The data must first be loaded into Couchbase. The script `load.py` will load a given number of emails into Couchbase, giving them embeddings with the specified model (configuration in `.env`).

data.json excerpt:
```javascript
[
    {
        "sender": ["\"Bob Smith\" <bobsmith@gmail.com>"],
        "receivers": {
            "to": ["\"Mike Muldoon\" <mmuldoon@gmail.com>"],
            "cc": ["\"Steve Hunter\" <hunter@gmail.com>"],
            "bcc": ["\"Ron Kochendorfer\" <kokain9@gmail.com>"]
        },
        "timestamp": "2009-12-18 01:48:29.77933",
        "contents": "body of email"
    },
    // ... etc ...
]
```

You can load this file with this command:

```bash
python load.py --data data.json
```

When stored in Couchbase as a document with an embedding, the document would look like this:

```javascript
[
    key: 
    {
        "sender": ["\"Bob Smith\" <bobsmith@gmail.com>"],
        "receivers": {
            "to": ["\"Mike Muldoon\" <mmuldoon@gmail.com>"],
            "cc": ["\"Steve Hunter\" <hunter@gmail.com>"],
            "bcc": ["\"Ron Kochendorfer\" <kokain9@gmail.com>"]
        },
        "timestamp": "2009-12-18 01:48:29.77933",
        "contents": "body of email",
        "embedding": [0.0123, -0.8471, ... etc ...]
    },
    // ... etc ...
]
```

Alternatively, the embeddings can be automatically generated on the fly with Capella AI Services with a [Process and Vectorize Unstructed Data workflow](https://docs.couchbase.com/ai/build/vectorization-service/vectorize-structured-data-capella.html). This approach will greatly simplify your AI application development, and separates the data processing from your application code.

When embedding with AI Services, that "embedding" field would be automatically created/updated whenever the document itself is created/updated. Furthermore, AI Services can use either an external Open AI type of model, or a private model hosted in Capella itself. (A private model could also be used in `load.py`).

`load.py` loads data into an "email" bucket, in the _default scope and _default collection.

# Step 2: Create the index

Once the data is loaded, create a [Composite Vector Index](https://docs.couchbase.com/cloud/vector-index/composite-vector-index.html).

```SQL
CREATE INDEX `idx_comp_vector_email`
ON `email`(`embedding` VECTOR,`sender`,`receivers`)
WITH {  "dimension":384, "similarity":"DOT", "description":"IVF,SQ8" }
```

Important notes:

* `DOT` similarity is used because it's good for comparing text content.
* Make sure the number of dimensions matches your .env setting.
* In this case, `sender` and `receivers` are an object and array respectively. This may not be optimal for production index/query.

Test the index with a query like:

```SQL
WITH anEmail AS (
    SELECT RAW embedding
    from `email`.`_default`.`_default` e
    WHERE e.timestamp = '2009-12-18 02:12:30.7799'
)
SELECT e.sender, e.receivers, e.content
FROM `email`.`_default`.`_default` e
ORDER BY APPROX_VECTOR_DISTANCE(e.embedding,anEmail.embedding[0],"DOT")
LIMIT 5;
```

The `WITH` clause here spares us from having to copy/paste a long vector into a sample query. Pick any timestamp from the data that has been loaded. The result of this query will almost certainly be the email with that timestamp, because it's the most semanticall similar. Not a very useful query, but it helps us to verify the index is working.

# Step 3: Perform a RAG operation

The `rag.py` program is an interactive command line program that allows you to specify a sender and/or receiver(s) email addresses, and enter a prompt. The program will gather the relevant information from the database, using a query similar to this form:

```SQL
/* TODO replace with correct syntax */
SELECT RAW e.content
FROM `email`.`_default`.`_default` e
WHERE e.sender = 'whatever@sender.com'
AND e.receivers CONTAINS('whoever@receiver.com')
ORDER BY APPROX_VECTOR_DISTANCE(e.embedding,anEmail.embedding[0],"DOT")
LIMIT 5;
```

This content will then will sent, along with the prompt, to an LLM, and the result displayed on the command line. Try similar prompts with different senders/receivers to see how the result differents.

