# Vector Search Indexes - Couchbase Vector Query Options in action

This demo shows three different ways to run vector search in Couchbase—**Hyperscale**, **Composite**, and **Hybrid (FTS)**, using real datasets and the same end-to-end flow. You’ll load data, generate embeddings, create the appropriate index, and run a RAG-style query to see how each approach affects relevance, filtering, and query flexibility. Each section uses a different dataset to highlight when one vector search option is a better fit than the others.

In this demo, you will:
- Load real-world datasets and generate embeddings
- Create Hyperscale, Composite, and Hybrid vector indexes
- Run RAG-style queries against each index type
- Compare how filtering, relevance, and query expressiveness differ

## Demo map

Each vector search option uses a different dataset and script to highlight when that approach is the best fit:

| Vector option | Dataset | Collection | RAG script |
|--------------|--------|------------|------------|
| Hyperscale | Wikipedia movie plots | `movies` | `ragHyperscale.py` |
| Composite | Customer care emails | `emails` | `ragComposite.py` |
| Hybrid (FTS) | Yelp businesses | `yelp` | `ragHybrid.py` |

(Use the CLI if you want to see the exact queries and tweak parameters; use the UI if you want a faster, more guided way to explore the same workflows).

## Success looks like this

After running this demo, you should be able to see and explain:

- **Hyperscale**: Broad semantic similarity across a large dataset, with no structured filtering.
- **Composite**: More precise results by combining semantic similarity with structured filters like sender, case ID, or product.
- **Hybrid (FTS)**: The ability to mix semantic search with search engine features such as keywords, text, and geospatial constraints.

If you can clearly describe *why* a given query uses one index type over the others, the demo is working as intended.

# Three Types of Vector Search indexes

## Hyperscale

This type of index is best for large data sets where you don't plan to do any filtering of data.

Example use cases:

- **Knowledge base**: A collection of car owner's manuals: What are the most common causes of muffler failure in vehicles?
- **Coding copilot**: examine all code repositories with a similar purpose to help generate a function
- **Research/writing**: examine all books for a related topic to generate a paragraph

## Composite

A hyperscale vector search (k-NN over all vectors) finds semantically similar documents but cannot efficiently restrict the search by structured attributes (for example, author or recipient of an email). A composite vector query lets you apply standard filters (e.g., `sender == X`) and then run a k-NN search only within that filtered subset, reducing noise and improving both accuracy and performance.

Example use cases
- **Email content** What organizations do emails from `jim.smith@gmail.com` mention the most?
- **Legal eDiscovery**: Restrict by `case_id` then run vector search to find semantically relevant clauses or communications.
- **Customer Support Triage**: Filter by product and retrieve semantically similar past tickets/solutions.
- **Medical Records Retrieval**: Limit by `patient_id` and perform semantic retrieval over clinical notes.

## Hybrid (FTS)

A hybrid vector search can use semantic vector search together with FTS features (for example, geospatial or traditional text).

Example use cases:
- **Business search**: What businesses within 5 miles of a certain location might help me with weight loss? (geospatial + vector)
- **Job search**: Find software jobs mentioning `C#` and `cloud`, ranked by semantic similarity to `backend API development`, posted in the last 30 days. (keyword + vector + range)
- **Content moderation**: Locate social posts mentioning a specific event or location, ranked by semantic similarity to harassment or threats. (keyword + vector)
- **News analysis**: List articles mentioning `interest rates` or `inflation`, ranked by similarity to recession risk narratives. (keyword + vector)

# Step 0: Prerequisites

Make sure you've got Python running. You'll probably want to create a virtual environment first, like this:

```bash
# Linux
python3 -m venv venv

# Windows
python -m venv venv
```

Then go into that venv with this command:

```bash
# Linux
source venv/bin/activate

# Windows
# you may need `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first
.\venv\Scripts\Activate.ps1
```

Then install requirements:

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

The `WITH` clause here spares us from having to copy/paste a long vector into a sample query. Pick any document key from the data that has been loaded. The result of this query will almost certainly be the email with that timestamp, because it's the most semantically similar. Not a very useful query, but it helps us to verify the index is working.

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

## [Hybrid (FTS) Vector Index](https://docs.couchbase.com/cloud/vector-search/vector-search.html)

Import "hybridIndex.json" into a Capella Search index. Note that the number of dimensions must match the model you're using (384 for local sentence transformers, 1536 for OpenAI, etc).

Important notes:
* `cosine` is used because it's good for text comparison

# Step 3: Perform a RAG operation

The `ragXYZ.py` programs are interactive command line programs that allows you to specify a sender and/or receiver(s) email addresses, and enter a prompt. This will be vectorized with the same model as in the load.py scripts. The program will gather the relevant information from the database, using a query corresponding to the index.

## Composite

This is the form of query that will be used for RAG+Composite Query. Note the predicates being used.

```SQL
SELECT RAW e.contents
FROM `vectorSearchDemo`.`_default`.`emails` e
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

The Hybrid query (geospatial+vector) is contructed using the Python SDK:

```python
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
```

This query will ultimately return the best matching documents by ID, along with a score. You can embed content fields in the index, but another common pattern that is often used is to perform KV lookups with the resulting IDs.

The content gathered by these queries will then be sent, along with the prompt, to an LLM (OpenAI gpt-4o-mini), and the result displayed on the command line.

## Composite

To execute a RAG prompt with Composite Vector Query:

```bash
python ragComposite.py --bucket vectorSearchDemo --scope _default --collection emails --prompt "Who is mentioned most?" --sender jim@example.com
```

If you don't use `--prompt` then the program will run interactively, asking you for a prompt and filters.

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

To execute a RAG prompt with Hyperscale Query:

```bash
python ragHyperscale.py --bucket vectorSearchDemo --scope _default --collection movies --prompt "What are some movies that involve escapes?"
```

If you don't use `--prompt`, the program will run interactively, asking you for a prompt.

Sample execution:

```bash
=== Couchbase Hyperscale Vector RAG Demo ===

...

=== Query ===

    SELECT RAW 'Title: ' || m.Title || ': ' || m.contents
    FROM `vectorSearchDemo`.`_default`.`movies` m
    ORDER BY APPROX_VECTOR_DISTANCE(m.embedding, $vector, "COSINE", 3)
    LIMIT $limit
     {'vector': '<removed>', 'limit': 5}

=== Context to Augment with ===

Title: For Her Sake: The film is a period drama taking place right before the start of the...
---
Title: The Suburbanite: The film is about a family who move to the suburbs, hoping for a ...
---
Title: The Great Train Robbery: The film opens with two bandits breaking into a railroad telegraph ...
---
Title: The Pasha's Daughter: The film begins with Jack Sparks, a young American, who is traveling...
---
Title: Youth's Endearing Charm: The film is about a court case and embezzlement.

=== Answer ===

The following movies from the provided context involve escapes:

1. **For Her Sake** - The girl helps her lover escape from captivity by giving him a file to free himself from the bars, and they flee on horseback.
2. **The Pasha's Daughter** - Jack Sparks escapes from prison by digging out the bar of his cell window and overpowering a guard before climbing over the wall into the courtyard of the Pasha's palace.

=== finished ===
```

## Hybrid (FTS)

To execute a RAG prompt with Hybrid Query:

```bash
python ragHybrid.py --bucket vectorSearchDemo --scope _default --collection yelp --prompt "Where can I go for various health and wellness services?" --latitude 34.426678 --longitude -119.711196 --radius 5mi
```

If you don't use `--prompt`, `--latitude`, `--longitude`, and `--radius`, the program will run interactively, asking you for all of these data points.

Sample execution:

```bash
=== Couchbase Hybrid Vector RAG Demo ===

...

=== Context to Augment with ===

Name: Abby Rappoport, LAC, CMQ

Doctors, Traditional Chinese Medicine, Naturopathic/Holistic, Acupuncture, Health & Medical, Nutritionists

=== Answer ===

You can go to Abby Rappoport, who offers services in Traditional Chinese Medicine, Naturopathic/Holistic approaches, Acupuncture, and Nutrition.

=== finished ===
```

# UI Experience

If you prefer a more "out of the box" or UI experience, you can also run a web application wrapper:

```bash
python app.py
```

This will give you options to run the same scripts (both data-loading and RAG) from a single web app.

![Loading data](/images/screenshotLoad.png "Loading dataset")

![Loading data](/images/screenshotRag.png "Performing RAG")

# Troubleshooting

Some things to check if you run into issues:

- **No results returned**: Verify data exists in the bucket/scope/collection and that filters (sender, radius, etc.) aren’t too restrictive.
- **Vector index errors**: Make sure the index is online and the `dimension` matches the embedding model.
- **Dimension mismatch**: If you changed embedding models, re-load the data and recreate the index.
- **Hybrid queries not working**: Confirm the FTS index exists and try increasing or removing the geo filter to validate data.
- **Composite filters not applying**: Check that filter fields are present, indexed, and have the expected data types.
- **Poor RAG answers**: Increase `LIMIT`, confirm the same embedding model is used for data and prompts, and inspect the retrieved context.

If in doubt, run a simple non-vector query first to confirm the data looks right.

# What's next?

* Composite: Try similar prompts with different senders/receivers to see how the result differ.
* Hybrid (FTS): Try a radius search from a few blocks away (with the same prompt) and see how the result differ.
* Hyperscale: Adjust tuneables like nprobes to see how the result differs.