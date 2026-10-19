# ATLAST MVP release verification

Status: **PASS**
Verified: 2026-10-19 (Europe/Paris)

## Environment

- macOS Darwin 25.6.0, Apple arm64
- Python 3.12.15 in a fresh `uv` virtual environment
- ATLAST wheel: `atlast_rag-0.1.0-py3-none-any.whl`
- Ollama client 0.33.3 with installed `gpt-oss:20b` (13 GB)
- Embeddings: cached `sentence-transformers/all-MiniLM-L6-v2`

## Build and isolated installation

Commands executed:

```bash
UV_CACHE_DIR=/tmp/atlast-uv-cache uv build --wheel \
  --out-dir /tmp/atlast-a6-release/dist
UV_CACHE_DIR=/tmp/atlast-uv-cache uv venv --python 3.12 \
  /tmp/atlast-a6-release/venv
UV_CACHE_DIR=/tmp/atlast-uv-cache uv pip install \
  --python /tmp/atlast-a6-release/venv/bin/python \
  /tmp/atlast-a6-release/dist/atlast_rag-0.1.0-py3-none-any.whl
/tmp/atlast-a6-release/venv/bin/atlast --help
unzip -l /tmp/atlast-a6-release/dist/atlast_rag-0.1.0-py3-none-any.whl
```

Result: wheel build and installation succeeded. The installed CLI exposed
`ingest`, `search`, `ask`, `evaluate`, and `serve`. Wheel inspection confirmed
the backend modules, `index.html`, `app.js`, `styles.css`, evaluation manifest,
and all four evaluation documents.

## Automated and retrieval evaluation

```bash
UV_CACHE_DIR=/tmp/atlast-uv-cache uv run --frozen pytest
HF_HOME=/tmp/atlast-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/atlast-a6-release/venv/bin/atlast evaluate --k 3
```

Results:

- Full suite: 40 passed in 0.82 seconds; one Starlette/AnyIO deprecation warning.
- Packaged evaluation: 4 documents, 6 answerable and 2 unanswerable cases.
- Recall@3: 1.0; MRR: 1.0.
- Both unanswerable queries still returned retrieval candidates.

The perfect metrics describe only the small synthetic benchmark and are not a
general accuracy claim.

## Installed-wheel end-to-end workflow

The installed wheel served the API and UI on loopback:

```bash
ATLAST_UPLOAD_DIR=/tmp/atlast-a6-release/uploads \
HF_HOME=/tmp/atlast-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/atlast-a6-release/venv/bin/atlast \
  --db /tmp/atlast-a6-release/state/qdrant serve \
  --ollama-model gpt-oss:20b --port 8766
```

Representative requests executed from the repository root:

```bash
curl --fail --silent --show-error -F \
  file=@/tmp/atlast-a6-release/release_protocol.pdf \
  http://127.0.0.1:8766/api/documents
curl --fail --silent --show-error -F \
  file=@src/atlast/eval_data/documents/orion_handbook.txt \
  http://127.0.0.1:8766/api/documents
curl --fail --silent --show-error http://127.0.0.1:8766/api/documents
curl --fail --silent --show-error --json \
  '{"question":"What color is the release beacon?","limit":4}' \
  http://127.0.0.1:8766/api/ask
curl --fail --silent --show-error -X PUT -F \
  file=@src/atlast/eval_data/documents/coastal_research.txt \
  http://127.0.0.1:8766/api/documents/8d2014ecf60f6089f585b587e80944987689ccc4f2c59f6635e4ad9342f4df72
curl --fail --silent --show-error -X DELETE \
  http://127.0.0.1:8766/api/documents/f986a023ae157cab69121f25cdfd837cd48117be319d18b8357ad49bf7aec01f
curl --fail --silent --show-error --json \
  '{"query":"quartz release beacon Tuesday emergency review","limit":5}' \
  http://127.0.0.1:8766/api/retrieve
curl --fail --silent --show-error --json \
  '{"query":"Orion checkpoint rollback checklist reviewers","limit":5}' \
  http://127.0.0.1:8766/api/retrieve
curl --fail --silent --show-error --max-time 240 --json \
  '{"question":"Who won the 2036 lunar chess championship?","limit":4}' \
  http://127.0.0.1:8766/api/ask
```

Actual results:

- `/`, `/app.js`, and `/styles.css` returned HTTP 200 from the wheel install.
- A two-page PDF and a TXT document indexed and listed with correct metadata.
- Real `gpt-oss:20b` answer: `quartz`, with citation `S1` resolving to PDF page 1.
- PDF replacement produced a new content-derived document ID and the updated
  coastal passage ranked first with score 0.7835.
- The other TXT document was deleted; exactly one point was removed.
- Queries for the replaced PDF text and deleted Orion text returned only the
  surviving coastal passage. Neither old document ID nor old passage appeared.
- Unsupported question, “Who won the 2036 lunar chess championship?”, returned
  `status: unsupported`, the standard insufficient-evidence answer, and no
  citations or sources.
- After stopping and restarting the installed server against the same database,
  the single replacement document remained listed and searchable (score 0.6408).
  The same `serve` command above was used for the restart.

## README and deployment prerequisites

The README setup commands were executed successfully:

```bash
uv python install 3.12
uv sync --frozen
```

`uv` installed/verified Python 3.12.15 and the frozen dependency sync checked 59
packages. Deployment remains local-only and requires:

- Python 3.10–3.13 and `uv`;
- enough disk/RAM for the embedding model and selected Ollama model;
- a running Ollama service with the configured model already installed;
- network access for first-time Python and model downloads;
- writable local paths for Qdrant data and uploads.

## Known limitations

- The synthetic retrieval evaluation is intentionally small.
- Retrieval has no calibrated similarity threshold, so unrelated questions can
  still return low-scoring candidate passages; the generation layer must abstain.
- Replacement writes new points before deleting old ones. This avoids normal
  data loss, but a process crash between operations can temporarily retain both.
- Document deletion removes Qdrant points, not retained upload/source files.
- The app is loopback-only and has no authentication or hosted deployment.
- Starlette's test client emits one upstream AnyIO deprecation warning.
