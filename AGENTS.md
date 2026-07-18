AGENTS.md

Travel RAG Project

Purpose

The goal of this repository is to build an AI-powered travel recommendation engine.

The initial milestone is not to build a web application.

The first objective is to create a high-quality Retrieval-Augmented Generation (RAG) pipeline using travel-related Reddit discussions.

The system should eventually answer questions such as:

“I like water, sunshine and swimming. I want somewhere similar to Porto.”

The recommendation should be based on retrieved travel experiences rather than only the LLM’s internal knowledge.

⸻

Current Project Phase

The project is currently focused on:

1. Reddit data ingestion
2. Cleaning and preprocessing
3. Embedding generation
4. Vector search
5. RAG
6. Evaluation

Do not build frontend features unless explicitly requested.

Do not spend time on authentication, deployment, styling or UI during this phase.

⸻

Overall Architecture

The pipeline should follow this flow:

Reddit

↓

Raw data

↓

Cleaning

↓

Structured travel documents

↓

Embeddings

↓

Vector database

↓

Retrieval

↓

LLM

↓

Travel recommendations

Each step should remain independent and easy to replace.

⸻

Primary Goals

The repository should eventually support:

* collecting travel discussions
* storing raw source data
* generating embeddings
* semantic retrieval
* destination similarity
* personalized recommendations
* explainable answers
* source attribution

⸻

Data Collection

Prefer official APIs whenever possible.

Always preserve:

* source URL
* subreddit
* post ID
* comment ID (if available)
* collection timestamp
* original text

Never overwrite raw data.

Store raw, cleaned and processed data separately.

Suggested structure:

data/

* raw/
* cleaned/
* processed/

⸻

Retrieval Principles

Retrieval should always happen before generation.

The LLM should answer using retrieved context whenever possible.

The system should provide recommendations together with reasoning supported by retrieved documents.

Avoid hallucinated travel advice.

⸻

Destination Profiles

The project will eventually build destination profiles containing information such as:

* climate
* beaches
* nature
* food
* nightlife
* safety
* budget
* walkability
* family friendliness
* activities
* atmosphere

These profiles should be derived from collected travel data rather than manually maintained whenever practical.

⸻

Coding Principles

Prefer:

* simple code
* readable code
* modular architecture
* pure functions
* type hints
* docstrings

Avoid unnecessary abstraction.

Only introduce new frameworks when there is a clear benefit.

⸻

Repository Structure

Prefer organizing code into modules such as:

src/

* ingestion/
* processing/
* embeddings/
* retrieval/
* generation/
* evaluation/

Each module should have a single responsibility.

⸻

Before Implementing

Before writing code:

1. Inspect the relevant files.
2. Explain the current state.
3. Describe the implementation plan.
4. Identify affected files.
5. Keep changes focused.

⸻

While Implementing

Work on one task at a time.

Avoid changing unrelated files.

Prefer small commits.

Never remove functionality without explanation.

If an implementation decision is uncertain, explain the trade-offs first.

⸻

Git

Do not commit automatically.

Do not push automatically.

After each completed task provide:

* summary of changes
* affected files
* suggested commit message

⸻

Testing

New backend functionality should be testable.

When appropriate, create small unit tests.

Evaluation scripts are preferred over manual testing.

⸻

Definition of Done

A task is complete only if:

* the implementation works
* the code is readable
* errors are handled
* documentation is updated when needed
* no unrelated files were modified
* the Git diff is clean
* a commit message is suggested