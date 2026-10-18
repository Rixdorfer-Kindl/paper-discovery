# Paper Discovery (seed papers → ranked new papers)

FastAPI backend + one static HTML page. Pre-filled with 5 Baker-lab protein-design seeds
(RFdiffusion, ProteinMPNN, RoseTTAFold, RoseTTAFold All-Atom, network hallucination).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys, then:
set -a; source .env; set +a
uvicorn app:app --reload    # http://localhost:8000
python test_smoke.py        # offline test, no keys needed
```
`GET /health` shows which keys the server sees.

## How it works (each step shows up in the "Agent trace" panel)
1. **Seeds** → resolved through OpenAlex (DOI, OpenAlex ID, or title).
2. **Candidates** → papers that cite a seed (most-cited + newest), papers a seed cites, OpenAlex related works.
   Every candidate remembers *which seeds* it links to and how; that is the "why it surfaced" line.
3. **Score** → graph score (share of seeds linked) + semantic score (embeddings; TF-IDF fallback without a key).
4. **LLM review** of the top slice → 0–10 relevance, short reason, one verbatim abstract quote.
   The quote is string-matched against the abstract; unmatched quotes are flagged ⚠.
5. **Tavily** → web leads for the question (new preprints) and code/GitHub links for the top 5 papers.

Final score = 0.6 × LLM relevance + 0.4 × (0.45 graph + 0.55 semantic). Weights are in `discover()`.

## Switching providers (after the hackathon)
All model calls are OpenAI-compatible and configured by env vars (see `.env.example`):
`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` and `EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL`.
No code change needed. Embeddings can come from a different provider than the chat model.

## Deploy on Nebius
The `Dockerfile` serves the app on port 8000. Deploy it as a Serverless **Endpoint** following
https://docs.nebius.com/serverless/overview, passing the env vars from `.env.example` as secrets.
Model inference stays on Token Factory, so the app and the model endpoint are hosted separately.

## Not verified yet (check these first)
- Real OpenAlex / Token Factory / Tavily calls were not run (the build sandbox blocked those hosts).
  Only the pipeline logic was tested, with mocked services.
- `EMBED_MODEL` default is a guess from the docs; confirm the exact id in the Token Factory model list.
- `LLM_MODEL` default is the model named in the workshop notebook. If it emits long reasoning text and
  `max_tokens` (900) cuts the JSON off, reviews show "unparsable output": raise `max_tokens` in `llm_json()`.
