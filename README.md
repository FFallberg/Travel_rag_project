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