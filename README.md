# Grounded Tutoring Module API (FastAPI)

FastAPI service for generating tutoring modules strictly from retrieved evidence.

Pipeline:
1. Evidence retrieval (tools ON): file search from vector stores and optional web search.
2. Module generation (tools OFF): module JSON is generated only from the `evidence_pack`.

Quality rule:
- Every section must have citation IDs (`citations`) referencing evidence IDs, or be marked `unverified=true` with `unverified_reason`.

## Project Layout

```text
tutoring_module_api/
  .env.example
  .gitignore
  app/
    main.py
    models.py
    openai_service.py
    store.py
    config.py
  scripts/
    demo_generate.py
  requirements.txt
  README.md
```

## Install

```bash
cd "/Users/araghavan26/Documents/New project/tutoring_module_api"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## BYOK Setup

1. Create your own OpenAI API key in your OpenAI account.
2. Copy template env file:
   ```bash
   cp .env.example .env
   ```
3. Set your key in `.env`:
   ```bash
   OPENAI_API_KEY=sk-...
   ```

Warning:
- Do not share API keys.
- Do not commit `.env`.
- `.env` is ignored by `.gitignore` in this project.

Note:
- ChatGPT Free/Plus subscriptions are separate from OpenAI API billing and API keys.

## Startup Check (env loaded)

Run this before starting the server:

```bash
cd "/Users/araghavan26/Documents/New project/tutoring_module_api"
python -c "from app.config import ensure_openai_api_key; ensure_openai_api_key(); print('OPENAI_API_KEY loaded')"
```

If the key is missing, startup validation raises:
`Missing OPENAI_API_KEY. Copy .env.example to .env and paste your own OpenAI API key.`

## Quick Test

Run the end-to-end smoke test:

```bash
cd "/Users/araghavan26/Documents/New project/tutoring_module_api"
python scripts/smoke_test.py
```

What it checks:
- Web-enabled generation (`allow_web=true`, no `vector_store_id`)
- At least 3 sections
- Each section has citations or `unverified=true`
- Cited evidence has non-empty snippets
- Web evidence citations include non-empty URLs

Docs-only optional check:
- By default, the smoke test uploads `samples/sample.txt` and runs a docs-only generation (`allow_web=false`).
- The repo already includes `samples/sample.txt`. Replace it with your own text if needed.
- To use a different file:
  ```bash
  python scripts/smoke_test.py --sample-file /absolute/path/your_sample.txt
  ```
- To skip docs-only check:
  ```bash
  python scripts/smoke_test.py --skip-doc-test
  ```

## Run

```bash
cd "/Users/araghavan26/Documents/New project/tutoring_module_api"
uvicorn app.main:app --reload --port 8000
```

The server performs a startup BYOK check and fails fast if `OPENAI_API_KEY` is missing.

## Health Endpoints

- `GET /health` -> `{"status":"OK"}`
- `GET /healthz` -> `{"status":"ok"}`

Neither endpoint returns secrets.

## API Endpoints

### 1) Upload docs

`POST /v1/docs/upload`

Accepts multiple PDF/TXT files, uploads to OpenAI, indexes them in a vector store, and returns `vector_store_id`.

```bash
curl -X POST "http://127.0.0.1:8000/v1/docs/upload" \
  -F "files=@/absolute/path/biology_notes.pdf" \
  -F "files=@/absolute/path/glossary.txt"
```

### 2) Generate module

`POST /v1/modules/generate`

```bash
curl -X POST "http://127.0.0.1:8000/v1/modules/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Cell respiration",
    "audience_level": "High school",
    "learning_objectives": [
      "Define aerobic respiration",
      "Explain ATP production",
      "Compare aerobic and anaerobic pathways"
    ],
    "allow_web": true,
    "vector_store_id": "vs_123"
  }'
```

Response shape:
- `module`: `{module_id,title,sections,glossary,mcqs}`
- `evidence_pack`: `[{evidence_id,source_type,title,url?,doc_name?,location?,snippet,retrieved_at}]`

### 3) Regenerate one section

`POST /v1/modules/{module_id}/regenerate`

```bash
curl -X POST "http://127.0.0.1:8000/v1/modules/<module_id>/regenerate" \
  -H "Content-Type: application/json" \
  -d '{
    "section_index": 1,
    "instructions": "Focus on ATP synthase and keep examples simple.",
    "allow_web": true
  }'
```

## Demo Script

`scripts/demo_generate.py` calls `/v1/modules/generate` and prints:
- section heading
- citation titles and URLs/doc names
- snippets

```bash
cd "/Users/araghavan26/Documents/New project/tutoring_module_api"
python scripts/demo_generate.py \
  --base-url "http://127.0.0.1:8000" \
  --topic "Photosynthesis" \
  --audience-level "Middle school" \
  --objective "Describe the inputs and outputs" \
  --objective "Explain the role of chlorophyll" \
  --allow-web \
  --vector-store-id "vs_123"
```
