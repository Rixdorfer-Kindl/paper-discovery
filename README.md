# Paper Discovery (seed papers → ranked new papers)

Give it a research question and a handful of papers you already know. It finds new papers around them,
ranks them, and shows why each one surfaced and what supports that explanation.
Built for the Research Paper Discovery track of Nebius Build Week Hackathon in partnership with NVIDIA.
Pre-filled example: protein AI design, seeded with five
Baker-lab papers (RFdiffusion, ProteinMPNN, RoseTTAFold, RoseTTAFold All-Atom, network hallucination).
I also built it as a practical tool for my own protein AI design project

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
TBD


