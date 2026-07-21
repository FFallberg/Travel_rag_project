Travel RAG Project

An AI-powered travel recommendation project built using travel discussions collected from Reddit.

The first goal is to build a Retrieval-Augmented Generation pipeline that can answer questions such as:

I want sunshine, swimming and access to water, but I want somewhere with a similar feeling to Porto.

The system should retrieve relevant travel experiences, compare destinations and generate recommendations supported by source material.

Current project phase

The current focus is:

1. Collecting travel-related Reddit data
2. Cleaning and structuring the data
3. Creating embeddings
4. Building semantic search
5. Generating travel recommendations with an LLM
6. Evaluating retrieval and recommendation quality

A frontend is not currently part of the MVP.

Planned pipeline

Reddit
  ↓
Raw data
  ↓
Cleaning and preprocessing
  ↓
Travel documents
  ↓
Embeddings
  ↓
Vector search
  ↓
Retrieved context
  ↓
LLM-generated recommendation

Project structure

data/
  raw/          Original collected data
  cleaned/      Cleaned Reddit posts and comments
  processed/    Chunked or structured documents
src/
  ingestion/    Reddit data collection
  processing/   Cleaning, extraction and chunking
  retrieval/    Embeddings and semantic search
  generation/   LLM-based recommendations
tests/          Unit and integration tests
evaluation/     Retrieval and recommendation evaluation

Initial MVP

The first MVP should:

* collect a small dataset from selected travel subreddits
* save raw data without modifying it
* clean posts and comments
* create embeddings
* retrieve relevant passages for a user query
* generate a recommendation using retrieved context
* include links to the supporting Reddit sources

Example query

I am interested in sunshine, water and swimming.
I like Porto and want somewhere with a similar atmosphere.

Development principles

* Start with a small dataset
* Keep raw data separate from processed data
* Preserve source URLs and metadata
* Prefer simple and modular Python code
* Evaluate retrieval before improving the user interface
* Do not rely on the LLM alone for travel facts
* Do not commit API keys or .env files

Status

Early development.

Reddit ingestion

The ingestion script uses Reddit's official OAuth API and stores the API listing
without changing its contents. Collection metadata is stored alongside the raw
response in `data/raw`; later pipeline stages should read the `response` field.

1. Create a Reddit "script" application and copy `.env.example` to `.env`.
2. Add the application's client ID and secret and a descriptive user agent.
3. Install dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.ingestion.reddit --limit 25 --subreddits travel solotravel
```

The limit must be between 1 and 100. Each run creates a timestamped JSON file;
existing raw data is never overwritten.

Travel Stack Exchange ingestion

The Travel Stack Exchange adapter is independent of Reddit and works without an
API key for small anonymous runs:

```bash
python3 -m src.ingestion.stackexchange --limit 10 --sort votes
```

Each run makes one request for questions and one batched request for their
answers. It saves the untouched API responses under `response` in matching
timestamped files:

```text
data/raw/stackexchange_questions_YYYYMMDDTHHMMSSZ.json
data/raw/stackexchange_answers_YYYYMMDDTHHMMSSZ.json
```

The built-in `withbody` filter includes question and answer bodies. IDs, source
links and per-item `content_license` values remain in the raw responses. At most
100 answers are collected per run; use `--answers-limit` to choose a smaller
bound. An optional Stack Apps API key can be configured as
`STACKEXCHANGE_API_KEY` for a higher request quota.

Stack Exchange contributions use versioned CC BY-SA licenses. Any downstream
display or adaptation must preserve attribution, link to the source, identify
the applicable license, and comply with its ShareAlike requirements.

Targeted Stack Exchange ingestion

Use the official advanced-search endpoint to collect destination-focused
threads instead of a generic top-question listing:

```bash
python3 -m src.ingestion.stackexchange \
  --query "Porto swimming beaches" \
  --tags portugal \
  --limit 10 \
  --answers-limit 100
```

Supplying `--query` or `--tags` enables targeted mode and defaults to relevance
sorting. Multiple tags are AND filters, so every returned question must contain
all supplied tags; free text is better for broader concepts such as sunshine,
swimming and atmosphere. The exact API response remains untouched under
`response`, while the non-secret search parameters are recorded separately in
`collection.request`. API keys are never written to raw captures.

Stack Exchange cleaning

Process one matching raw question/answer capture pair with explicit input paths:

```bash
python3 -m src.processing.stackexchange \
  --questions-file data/raw/stackexchange_questions_20260718T172355Z.json \
  --answers-file data/raw/stackexchange_answers_20260718T172355Z.json
```

The processor validates that the captures have the same collection timestamp,
joins answers through `question_id`, converts HTML bodies to readable text, and
writes one attributed document per thread under a timestamped directory in
`data/cleaned`. Raw captures are never modified. Existing cleaned runs are not
overwritten.

Retrieval documents

Convert a cleaned run into embedding-ready JSONL:

```bash
python3 -m src.processing.retrieval_documents \
  --input-dir data/cleaned/stackexchange_threads_20260718T172355Z
```

The output is written to a timestamped
`data/processed/stackexchange_retrieval_*.jsonl` file. Each question and answer
becomes a separate retrieval unit. Answer units include the question title as
context and retain their own source URL, author, score, accepted status,
timestamps and content license. IDs and ordering are deterministic. Existing
processed files are never overwritten.

Local embeddings

Generate normalized multilingual embeddings from a retrieval JSONL file:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.embeddings.local \
  --input-file data/processed/stackexchange_retrieval_20260719T102240Z.jsonl
```

The default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
model supports Swedish and English and produces 384-dimensional vectors. The
first run downloads the model locally. Output consists of a compressed `.npz`
file mapping document IDs to normalized float32 vectors and a matching JSON
manifest containing the model name, dimensions, source checksum and creation
time. Existing embedding artifacts are never overwritten.

Semantic search

Search the local embedding index with a Swedish or English query:

```bash
.venv/bin/python -m src.retrieval.semantic_search \
  --manifest data/processed/stackexchange_embeddings_20260719T110457Z.json \
  --query "sol, bad och närhet till vatten" \
  --top-k 3
```

The search command verifies the retrieval-document checksum and validates the
vector dimensions, document IDs and normalization before searching. It embeds
the query with the exact model recorded in the manifest, computes cosine
similarity, and returns ranked JSON containing scores, text, metadata, licenses
and source links. No LLM is involved at this stage.

Retrieval evaluation

Run the small bilingual retrieval sanity set before adding generation:

```bash
.venv/bin/python -m src.evaluation.retrieval \
  --manifest data/processed/stackexchange_embeddings_20260719T110457Z.json \
  --cases evaluation/stackexchange_queries.json \
  --top-k 3
```

The report includes Hit Rate@k, Mean Reciprocal Rank@k and mean Recall@k, plus
ranked document IDs and scores for every query. The initial evaluation set has
only four manually judged Swedish/English queries covering two source threads;
it verifies the pipeline but is too small to support general quality claims.

For the larger 98-document local index, the expanded evaluation set includes
ten positive queries and four negative queries where no indexed document is
relevant. Use `--min-score` to evaluate an abstention threshold:

```bash
.venv/bin/python -m src.evaluation.retrieval \
  --manifest data/processed/stackexchange_embeddings_20260719T183004Z.json \
  --cases evaluation/stackexchange_queries_expanded.json \
  --top-k 3 \
  --min-score 0.6
```

Negative-query rejection and positive retrieval metrics are reported
separately. Positive cases in the expanded set are judged at thread level, so
any question or answer from the selected source thread counts as topically
relevant. On this small set, `0.6` separates the tested positive and negative
queries, but it is preliminary and must be recalibrated as the corpus and
evaluation set grow.
