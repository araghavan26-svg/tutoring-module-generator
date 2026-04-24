# Grounded Tutoring Module API (FastAPI)

FastAPI service for generating tutoring modules strictly from retrieved evidence.

Pipeline:
1. Evidence retrieval (tools ON): file search from vector stores and optional web search.
2. Module generation (tools OFF): module JSON is generated only from the `evidence_pack`.

Quality rule:
- Every section must have citation IDs (`citations`) referencing evidence IDs, or be marked `unverified=true` with `unverified_reason`.

## Current Status And Roadmap

Current status: MVP is deployed on Render and can be shared by link. Main next goal is improving the user experience for non-technical users.

Next-feature priorities from exhibition feedback:

1. Real user interface
   - Mature the creator UI into a simple non-technical experience so users do not need to use `/docs`.
   - The UI should let users enter a topic, audience level, objectives, upload docs, toggle web search, generate a module, and view/export results.
2. Light mode / dark mode
   - Add a theme toggle.
   - Persist the user's theme choice locally if possible.
3. Image/photo support
   - Add support for modules to include helpful images or photos.
   - Example: if the module is about roasted chicken, the user can see what roasted chicken looks like.
   - Images should be clearly marked as visual aids and should not replace citations.
4. More safeguards
   - Add safer topic handling.
   - Warn users when sources are weak or missing.
   - Prevent uncited claims where possible.
   - Add basic API cost and rate protections.
5. Multi-user collaboration
   - Treat this as a longer-term feature.
   - Multiple people should eventually be able to work on the same module.
   - This likely requires accounts, saved projects, a database, and permissions.

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
  templates/
    base.html
    index.html
    create.html
    module_editor.html
  static/
    style.css
    create.js
    module_editor.js
  start_app.command
  scripts/
    demo_generate.py
  requirements.txt
  README.md
```

## Quick Start

1. Double-click `start_app.command`
2. Wait for the browser to open `http://127.0.0.1:8000`
3. If startup says the API key is missing, complete the BYOK setup below and launch again

`start_app.command` is the main way to launch the app for exhibition use. It will:
- create or reuse `.venv`
- install requirements
- start Uvicorn
- open the app in your default browser

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

## Optional Startup Check

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
- `source_policy` domain controls with `allowed_domains=["kids.britannica.com","www.vocabulary.com"]`
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

## Demo Reset

1. Double-click `reset_demo_state.command`
2. This clears saved modules, version history, and cached evidence so the next demo starts clean

## Creator Web UI

After starting the server, open:
- `GET /` landing page
- `GET /create` module creation form
- `GET /modules/{module_id}` module editor page

Guided first-use flow:
1. Click **Create a Module** (or **Try Sample Module**).
2. Fill in topic and learning goals.
3. Choose source options and generate.

Demo-friendly path:
1. Click **Try Sample Module**
2. Leave **Use web sources** on
3. Leave **Use sample document** checked if it appears
4. Generate for a fast, polished first example

The UI uses minimal HTML + JS and calls API endpoints:
- `POST /v1/docs/upload`
- `POST /v1/modules/generate`
- `POST /v1/modules/{module_id}/sections/{section_id}/regenerate`
- `POST /v1/modules/{module_id}/refresh_sources`
- `GET /v1/modules/{module_id}/export/markdown`

## Health Endpoints

- `GET /health` -> `{"status":"OK"}`
- `GET /healthz` -> `{"status":"ok"}`

Neither endpoint returns secrets.

## Advanced Terminal Launch (Optional)

If you need it for development, you can still launch manually:

```bash
cd "/Users/araghavan26/Documents/New project/tutoring_module_api"
./.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

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
    "source_policy": {
      "allow_web": true,
      "web_recency_days": 30,
      "allowed_domains": ["kids.britannica.com", "www.vocabulary.com"],
      "blocked_domains": ["example.com"]
    },
    "vector_store_id": "vs_123"
  }'
```

Response shape:
- `module`: `{module_id,title,overview,sections,glossary,mcqs,evidence_pack,source_policy}`
- `evidence_pack`: `[{evidence_id,source_type,domain?,title,url?,doc_name?,location?,snippet,retrieved_at}]`

Section shape now includes objective alignment fields:
- `objective_index` (0-based)
- `learning_goal`

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

Cached-evidence edit loop endpoint:

`POST /v1/modules/{module_id}/sections/{section_id}/regenerate`

- Default behavior (`refresh_sources=false`): reuse cached module evidence pack, no retrieval.
- Optional `refresh_sources=true`: refresh evidence pack first using stored `source_policy` + `vector_store_id`, then regenerate that section.

```bash
curl -X POST "http://127.0.0.1:8000/v1/modules/<module_id>/sections/section-1/regenerate" \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "Tighten the explanation for clarity.",
    "refresh_sources": false
  }'
```

### 4) Refresh cached sources

`POST /v1/modules/{module_id}/refresh_sources`

Rebuilds and stores module-level `evidence_pack` and returns kept counts:
- `evidence_count`
- `doc_count`
- `web_count`

```bash
curl -X POST "http://127.0.0.1:8000/v1/modules/<module_id>/refresh_sources" \
  -H "Content-Type: application/json" \
  -d '{
    "source_policy": {
      "allow_web": true,
      "web_recency_days": 30,
      "allowed_domains": ["kids.britannica.com", "www.vocabulary.com"],
      "blocked_domains": ["example.com"]
    }
  }'
```

### 5) Export module

`GET /v1/modules/{module_id}/export/json`

Returns structured export with:
- `title`
- `overview`
- `sections` (including `learning_goal`)
- `glossary`
- `mcqs` (including explanations)
- `footnotes` (citation list)

`GET /v1/modules/{module_id}/export/markdown`

Returns a Markdown document with the same content and citation footnotes.

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
