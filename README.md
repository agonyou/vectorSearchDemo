# Email Demo — Couchbase Vector Query Options in action

This demo illustrates the strengths of each vector search option in Couchase.

Datasets

- TODO

# Three Types of Vector Search indexes

## Hyperscale

This type of index is best for large data sets where you don't plan to do any filtering of data.

Example use cases:

- *Knowledge base*: A collection of *car owner's manuals: What are the most commons causes of muffler failure in vehicles?
- *Coding copilot*: examine all code repositories with a similar purpose to help generate a function
- *Research/writing*: examine all books for a related topic to generate a paragraph

## Composite

A hyperscale vector search (k-NN over all vectors) finds semantically similar documents but cannot efficiently restrict the search by structured attributes (for example, author or recipient of an email). A composite vector query lets you apply standard filters (e.g., `sender == X`) and then run a k-NN search only within that filtered subset, reducing noise and improving both accuracy and performance.

Example use cases
- *Email content* What organizations do emails from jim.smith@gmail.com mention the most?
- *Legal eDiscovery*: Restrict by case_id then run vector search to find semantically relevant clauses or communications.
- *Customer Support Triage*: Filter by product and retrieve semantically similar past tickets/solutions.
- *Medical Records Retrieval*: Limit by patient_id and perform semantic retrieval over clinical notes.

## Hybrid (FTS)

A hybrid vector search can use semantic vector search together with FTS features (for example, geospatial or traditional text).

Example use cases:
- *Business search* What businesses within 5 miles of a certain location might help me with weight loss?
- TODO more

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

The data must first be loaded into Couchbase. The `load.py` script will load data into Couchbase, giving them embeddings with the specified model (configuration in `.env`).

You can load with a command like this:

```bash
python load.py --data data.json --id-field id --text-fields contents --bucket mybucket --scope myschema --collection mydocs --limit 5
```

The `--limit N` parameter means that you want to load the next N documents that haven't been loaded yet.

When stored in Couchbase as a document with an embedding, documents will look like this:

```javascript
[
    key: "doc1"
    {
        "emailFrom": ["bobsmith@gmail.com"],
        "caseId": "123456",
        ... etc ...
        "textToUseInVector": "...text goes here...",
        "embedding": [0.0123, -0.8471, ... etc ...]
    },
    // ... etc ...
]
```

Alternatively, the embeddings can be automatically generated on the fly with Capella AI Services with a [Process and Vectorize Unstructed Data workflow](https://docs.couchbase.com/ai/build/vectorization-service/vectorize-structured-data-capella.html). This approach will greatly simplify your AI application development, and separates the data processing from your application code.

When embedding with AI Services, that "embedding" field would be automatically created/updated whenever the document itself is created/updated. Furthermore, AI Services can use either an external Open AI type of model, or a private model hosted in Capella itself. (A private model could also be used in `load.py`).

Here are three examples of loading data for the three use cases:

**Composite** - load emails from the [Customer Care Emails dataset](https://www.kaggle.com/datasets/rtweera/customer-care-emails): dataset.csv

```bash
python load.py --data data/dataset.csv --text-fields subject message_body --bucket vectorSearchDemo --scope _default --collection emails --copy-fields subject sender receiver message_body --limit 5 --id-field sender timestamp
```

> NOTE: Two fields being vectorized, combined into one "content" field. I also have both of them in `--copy-fields` so they can stay separate. Your needs will vary by use case.

This will vectorize the subject+message body together, and save the sender and receiver for filtering. It also combines sender/timestamp combination as a unique ID for each document.

**Hyperscale** - load movie plots from the [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots): wiki_movie_plots_deduped.csv

```bash
python load.py --data data/wiki_movie_plots_deduped.csv --text-fields Plot --bucket vectorSearchDemo --scope _default --collection movies --copy-fields Title ReleaseYear Director --limit 5 --id-field Title ReleaseYear
```

> NOTE: For simplicity, I removed the spaces from CSV header row (i.e. Release Year became ReleaseYear). Since this is a single field being vectorized, I did not include it in `--copy-fields`. Again, your need will vary by use case.

**Hybrid** - load businesses from the [Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset): yelp_academic_dataset_business.json

```bash
python load.py --data data/yelp_academic_dataset_business.json --text-fields categories --bucket vectorSearchDemo --scope _default --collection yelp --copy-fields latitude longitude name --limit 5 --id-field business_id
```

# Step 2: Create the indexes

Once the data is loaded, create index(es).

## [Composite Vector Index](https://docs.couchbase.com/cloud/vector-index/composite-vector-index.html).

```SQL
CREATE INDEX `idx_comp_vector_email`
ON `emails`(`embedding` VECTOR,`sender`,`receivers`)
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
    from `vectorSearchDemo`.`_default`.`emails` x
    USE KEYS ["Aetheros Support <support@aetheros.com>::2023-10-26T03:42:15Z"]
)
SELECT e.sender, e.receiver, e.content
FROM `vectorSearchDemo`.`_default`.`emails` e
ORDER BY APPROX_VECTOR_DISTANCE(e.embedding,anEmail[0],"DOT")
LIMIT 5;
```

The `WITH` clause here spares us from having to copy/paste a long vector into a sample query. Pick any timestamp from the data that has been loaded. The result of this query will almost certainly be the email with that timestamp, because it's the most semanticall similar. Not a very useful query, but it helps us to verify the index is working.

## [Hyperscale Vector Index](https://docs.couchbase.com/cloud/vector-index/hyperscale-vector-index.html)

```SQL
CREATE VECTOR INDEX `idx_hyperscale_plot`
ON `vectorSearchDemo`.`_default`.`movies`(`embedding` VECTOR)
WITH {
  "dimension": 384,
  "similarity": "COSINE",
  "description": "IVF,SQ8"
};
```

Important notes:
* `COSINE` similarity here is good for text comparison.
* `IVF` with no number allows Capella to choose an appropriate number of centroids
* `dimension` needs to be correct for the model you're using

Test the index with a query like:

```SQL
WITH aMovie AS (
    SELECT RAW m.embedding
    FROM `vectorSearchDemo`.`_default`.`movies` AS m
    WHERE m.Title = "Alice in Wonderland"
    LIMIT 1
)
SELECT m.Title, m.ReleaseYear, approx_distance
FROM `vectorSearchDemo`.`_default`.`movies` AS m
LET approx_distance = APPROX_VECTOR_DISTANCE(
    m.embedding, aMovie[0], "COSINE", 3
)
ORDER BY approx_distance
LIMIT 5;
```

## [Hybrid (FTS) Vector Index]()

TODO

# Step 3: Perform a RAG operation

The `ragXYZ.py` programs are interactive command line programs that allows you to specify a sender and/or receiver(s) email addresses, and enter a prompt. This will be vectorized with the same model as in the load.py scripts. The program will gather the relevant information from the database, using a query corresponding to the index.

## Composite

This is the form of query that will be used for RAG+Composite Query. Note the predicates being used.

```SQL
SELECT RAW e.contents
FROM `email`.`_default`.`_default` e
WHERE e.sender == $sender
AND e.receiver == $receiver
ORDER BY APPROX_VECTOR_DISTANCE(e.embedding, <vector of prompt goes here>, "COSINE")
LIMIT 5
```

## Hyperscale

This is the form of query that will be used for RAG+Hyperscale Query. Note that it's entirely based on knn.

```SQL
SELECT RAW m.contents
FROM `vectorSearchDemo`.`_default`.`movies` AS m
LET approx_distance = APPROX_VECTOR_DISTANCE(
    m.embedding, <embedding>, "COSINE", 3
)
ORDER BY approx_distance
LIMIT 5;
```

## Hybrid (FTS)

TODO

This content will then be sent, along with the prompt, to an LLM (OpenAI gpt-4o-mini), and the result displayed on the command line.

Example executions:

## Composite

To execute a RAG prompt with Composite Vector Query:

```bash
python ragComposite.py --bucket vectorSearchDemo --scope _default --collection emails --prompt "Who is mentioned most?" --sender jim@example.com
```

If you don't use `--prompt` then the program will run interactively, asking you from a prompt and filters.

Sample execution:

```bash
== Couchbase Composite Vector RAG Demo ===

Filter by sender (exact match): "Wayne D. Kimmel" <wayne@etfventurefunds.com>
Filter by receiver field (exact match): 

Enter your prompt:
> What organizations does Wayne most discuss?

=== Answer ===

Wayne most discusses ETF Venture Funds and the National Venture Capital Association (NVCA). He also mentions Fisker, indicating an interest in investing in the company.
```

## Hyperscale

TODO

## Hybrid (FTS)

TODO

# UI Experience

If you prefer a more "out of the box" or UI experience, you can also run a web application wrapper:

```bash
python app.py
```

This will give you options to run the same scripts from a single web app.

TODO: screenshots?

# What's next?

* Composite: Try similar prompts with different senders/receivers to see how the result differs.
* Hybrid (FTS): Try a radius search from a few blocks away (with the same prompt) and see how the result differs.
* Hyperscale: TODO (maybe adjust tuneables like nprobes to see how the result differs?)