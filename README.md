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
