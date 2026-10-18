"""Paper discovery: from a seed collection to a ranked, traceable list of new papers.

Pipeline (deterministic, every step is logged into `trace`):
  1. Resolve seeds (DOI / OpenAlex ID / title) via OpenAlex.
  2. Gather candidates from the citation graph: papers citing a seed, papers a seed
     cites, and OpenAlex "related works". Track WHICH seeds each candidate links to.
  3. Score: graph score (share of seeds linked) + semantic score (embeddings, or a
     TF-IDF fallback when no embedding key is set).
  4. LLM review of the top candidates: relevance 0-10, a short reason, and a verbatim
     abstract quote. The quote is checked against the abstract programmatically.
  5. Tavily web search for extra context / code links (optional).

Everything model-related is configured by env vars, so switching from Nebius Token
Factory to any other OpenAI-compatible provider needs no code change.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- config
OPENALEX = "https://api.openalex.org"
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")  # optional, gives OpenAlex's polite pool

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("NEBIUS_API_KEY", "")).strip()
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", LLM_BASE_URL)
EMBED_API_KEY = (os.getenv("EMBED_API_KEY") or LLM_API_KEY).strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")  # check Token Factory's model list

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

SELECT = (
    "id,doi,title,publication_year,publication_date,authorships,abstract_inverted_index,"
    "referenced_works,related_works,cited_by_count,primary_location"
)
REL_TEXT = {
    "cites": "cites seed",
    "cited_by_seed": "is cited by seed",
    "openalex_related": "is OpenAlex-related to seed",
}

app = FastAPI(title="Paper Discovery")


# --------------------------------------------------------------------------- helpers
def sid(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def abstract_of(work: dict) -> str:
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def to_paper(w: dict) -> dict:
    loc = w.get("primary_location") or {}
    venue = (loc.get("source") or {}).get("display_name")
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    authors = [(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []]
    return {
        "id": sid(w["id"]),
        "title": (w.get("title") or "").strip(),
        "year": w.get("publication_year"),
        "date": w.get("publication_date") or "",
        "authors": [a for a in authors if a],
        "venue": venue,
        "doi": doi,
        "cited_by_count": w.get("cited_by_count") or 0,
        "abstract": abstract_of(w),
        "refs": [sid(x) for x in w.get("referenced_works") or []],
        "related": [sid(x) for x in w.get("related_works") or []],
        "url": f"https://doi.org/{doi}" if doi else (loc.get("landing_page_url") or w["id"]),
        "openalex_url": w["id"],
    }


_cache: dict[str, Any] = {}


def oa_get(path: str, **params: Any) -> Any:
    """GET from OpenAlex with a small in-memory cache and retries."""
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    key = path + "?" + json.dumps(params, sort_keys=True)
    if key in _cache:
        return _cache[key]
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.get(OPENALEX + path, params=params, timeout=30, follow_redirects=True)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            _cache[key] = r.json()
            return _cache[key]
        except httpx.HTTPError as e:
            last = e
            time.sleep(1)
    raise RuntimeError(f"OpenAlex request failed: {last or 'rate limited'}")


def fetch_by_ids(ids: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        data = oa_get("/works", filter="openalex_id:" + "|".join(batch), per_page=50, select=SELECT)
        out += (data or {}).get("results", [])
    return out


def resolve_seed(s: str) -> dict | None:
    s = s.strip()
    if not s:
        return None
    m = re.search(r"10\.\d{4,9}/\S+", s)
    if m:
        w = oa_get(f"/works/https://doi.org/{m.group(0).rstrip('.,;')}", select=SELECT)
    elif re.search(r"\bW\d{6,}\b", s):
        w = oa_get(f"/works/{re.search(r'W\d{6,}', s).group(0)}", select=SELECT)
    else:
        data = oa_get("/works", search=s, per_page=1, select=SELECT)
        res = (data or {}).get("results") or []
        w = res[0] if res else None
    return to_paper(w) if w else None


# --------------------------------------------------------------------------- step 2: candidates
def gather_candidates(seeds: list[dict], per_seed: int, include_refs: bool, trace: list[str]) -> dict:
    seed_ids = {s["id"] for s in seeds}
    seed_titles = {norm_title(s["title"]) for s in seeds}
    cands: dict[str, dict] = {}

    def add(work: dict, seed: dict, relation: str) -> None:
        p = to_paper(work)
        if not p["title"] or p["id"] in seed_ids or norm_title(p["title"]) in seed_titles:
            return
        c = cands.setdefault(p["id"], {"paper": p, "links": {}})
        c["links"].setdefault(seed["id"], set()).add(relation)

    tasks = []  # (seed, relation, zero-arg callable returning works)
    for s in seeds:
        for sort in ("cited_by_count:desc", "publication_date:desc"):
            tasks.append((s, "cites", lambda s=s, sort=sort: (oa_get(
                "/works", filter=f"cites:{s['id']}", sort=sort, per_page=per_seed, select=SELECT
            ) or {}).get("results", [])))
        if include_refs:
            tasks.append((s, "cited_by_seed", lambda s=s: fetch_by_ids(s["refs"][:150])))
            tasks.append((s, "openalex_related", lambda s=s: fetch_by_ids(s["related"][:10])))

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda t: t[2](), tasks))
    counts: Counter = Counter()
    for (seed, relation, _), works in zip(tasks, results):
        counts[relation] += len(works)
        for w in works:
            add(w, seed, relation)
    trace.append(
        f"Candidates: {len(cands)} unique papers (raw hits: "
        + ", ".join(f"{REL_TEXT[k]}={v}" for k, v in counts.items()) + ")"
    )

    # merge preprint/published duplicates by normalised title
    by_title: dict[str, dict] = {}
    for c in cands.values():
        k = norm_title(c["paper"]["title"])
        if k not in by_title:
            by_title[k] = c
            continue
        keep, other = by_title[k], c
        if (other["paper"]["cited_by_count"], bool(other["paper"]["abstract"])) > (
            keep["paper"]["cited_by_count"], bool(keep["paper"]["abstract"])
        ):
            keep, other = other, keep
        for sd, rels in other["links"].items():
            keep["links"].setdefault(sd, set()).update(rels)
        by_title[k] = keep
    merged = list(by_title.values())
    if len(merged) != len(cands):
        trace.append(f"Merged {len(cands) - len(merged)} duplicate records (same title) -> {len(merged)}")
    return {c["paper"]["id"]: c for c in merged}


def cap_pool(cands: dict, cap: int) -> list[dict]:
    """Keep strongly-linked papers, but reserve 30% of slots for the newest ones."""
    items = sorted(cands.values(), key=lambda c: (len(c["links"]), c["paper"]["cited_by_count"]), reverse=True)
    if len(items) <= cap:
        return items
    n_top = int(cap * 0.7)
    rest = sorted(items[n_top:], key=lambda c: c["paper"]["date"], reverse=True)
    return items[:n_top] + rest[: cap - n_top]


# --------------------------------------------------------------------------- step 3: scoring
STOP = set(
    "a an the of and or in on for to with by from is are was were be as at this that these those we our it its "
    "using use used based via into than which can have has not also such their they more most been".split()
)


def _toks(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", t.lower()) if w not in STOP]


def _tfidf(docs: list[str]) -> list[dict]:
    tfs = [Counter(_toks(d)) for d in docs]
    df: Counter = Counter()
    for c in tfs:
        df.update(c.keys())
    n = len(docs)
    return [{w: (1 + math.log(f)) * math.log((n + 1) / (df[w] + 0.5)) for w, f in c.items()} for c in tfs]


def _cos_sparse(a: dict, b: dict) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY, timeout=60)
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        r = client.embeddings.create(model=EMBED_MODEL, input=[t[:1500] for t in texts[i : i + 32]])
        out += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
    return out


def minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs, default=0.0), max(xs, default=0.0)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


def semantic_scores(question: str, seeds: list[dict], pool: list[dict], trace: list[str]) -> list[float]:
    """Similarity to the seed centroid (60%) and to the question (40%)."""
    seed_txt = [f"{s['title']}. {s['abstract']}"[:1500] for s in seeds]
    cand_txt = [f"{c['paper']['title']}. {c['paper']['abstract']}"[:1500] for c in pool]
    ns = len(seeds)
    try:
        if not EMBED_API_KEY:
            raise RuntimeError("no embedding API key set")
        vecs = embed([question] + seed_txt + cand_txt)
        q, sv, cv = vecs[0], vecs[1 : 1 + ns], vecs[1 + ns :]
        centroid = [sum(col) / ns for col in zip(*sv)]
        sims = [0.6 * _cos(v, centroid) + 0.4 * _cos(v, q) for v in cv]
        trace.append(f"Semantic score: embeddings ({EMBED_MODEL}) over {len(cv)} candidates")
    except Exception as e:  # noqa: BLE001 - any failure falls back to lexical scoring
        vecs = _tfidf([question] + seed_txt + cand_txt)
        q, sv, cv = vecs[0], vecs[1 : 1 + ns], vecs[1 + ns :]
        centroid: dict[str, float] = {}
        for d in sv:
            for k, v in d.items():
                centroid[k] = centroid.get(k, 0.0) + v
        sims = [0.6 * _cos_sparse(v, centroid) + 0.4 * _cos_sparse(v, q) for v in cv]
        trace.append(f"WARNING embeddings unavailable ({str(e)[:120]}); fell back to TF-IDF similarity")
    return sims


# --------------------------------------------------------------------------- step 4: LLM review
SYSTEM = (
    "You are a careful research assistant helping a protein-design researcher triage literature. "
    "Use ONLY the information given. A citation link alone is not evidence of relevance. "
    "Answer with a single JSON object and nothing else."
)


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def llm_json(prompt: str) -> dict | None:
    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=90)
    r = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=900,
        temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    )
    return parse_json(r.choices[0].message.content or "")


def _flat(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’`]", "", t or "")).strip().lower()


def evidence_verified(quote: str, abstract: str) -> bool:
    q = _flat(quote)
    return len(q) >= 8 and q in _flat(abstract)


def connection_text(c: dict, seed_by_id: dict) -> tuple[str, list[dict]]:
    links, parts = [], []
    for seed_id, rels in c["links"].items():
        for r in sorted(rels):
            title = seed_by_id[seed_id]["title"]
            links.append({"seed": title, "relation": r})
            parts.append(f"{REL_TEXT[r]} “{title[:70]}”")
    return "; ".join(parts), links


def review_one(question: str, seeds: list[dict], c: dict, connection: str, lang: str) -> dict:
    p = c["paper"]
    if not p["abstract"]:
        return {"relevance": None, "reason": "", "evidence": "", "verified": False,
                "uncertainty": "No abstract available, so relevance could not be judged from text."}
    seed_list = "\n".join(f"- {s['title']} ({s['year']})" for s in seeds)
    prompt = f"""Research question: {question}

Seed papers (the researcher's collection):
{seed_list}

Candidate paper:
Title: {p['title']} ({p['year']})
Abstract: {p['abstract'][:2500]}
Connection to seeds (from citation data): {connection}

Return JSON with exactly these keys:
"relevance": integer 0-10 (0-2 unrelated; 3-5 same broad area but different problem or method; 6-8 directly useful method or evidence for the question; 9-10 core work for the question),
"reason": 1-2 sentences in {lang} explaining why it is or is not worth reading for this question,
"evidence": ONE verbatim quote (max 30 words) copied exactly from the abstract that supports the reason, or "" if none,
"uncertainty": one short sentence in {lang} on what the abstract does not tell us."""
    try:
        d = llm_json(prompt)
    except Exception as e:  # noqa: BLE001
        return {"relevance": None, "reason": "", "evidence": "", "verified": False,
                "uncertainty": f"LLM call failed: {str(e)[:100]}"}
    if not d:
        return {"relevance": None, "reason": "", "evidence": "", "verified": False,
                "uncertainty": "LLM returned unparsable output."}
    try:
        rel = max(0, min(10, int(round(float(d.get("relevance"))))))
    except (TypeError, ValueError):
        rel = None
    ev = str(d.get("evidence") or "").strip()
    return {"relevance": rel, "reason": str(d.get("reason") or "").strip(), "evidence": ev,
            "verified": bool(ev) and evidence_verified(ev, p["abstract"]),
            "uncertainty": str(d.get("uncertainty") or "").strip()}


# --------------------------------------------------------------------------- step 5: Tavily
def tavily_search(query: str, k: int = 3) -> list[dict]:
    from tavily import TavilyClient

    r = TavilyClient(api_key=TAVILY_API_KEY).search(query, max_results=k)
    return [{"title": x.get("title") or x.get("url"), "url": x.get("url")} for x in r.get("results", [])]


def safe_tavily(query: str, k: int = 3) -> list[dict]:
    try:
        return tavily_search(query, k)
    except Exception:  # noqa: BLE001 - enrichment is optional
        return []


# --------------------------------------------------------------------------- pipeline
class DiscoverRequest(BaseModel):
    question: str
    seeds: list[str] = Field(min_length=1)
    top_n: int = Field(10, ge=1, le=30)
    per_seed: int = Field(40, ge=5, le=100)
    include_references: bool = True
    lang: str = "中文"


def discover(req: DiscoverRequest) -> dict:
    t0 = time.time()
    trace: list[str] = []

    # 1. seeds
    resolved, unresolved = [], []
    for raw in req.seeds:
        p = resolve_seed(raw)
        (resolved if p else unresolved).append(p or raw)
    seen = set()
    seeds = [s for s in resolved if not (s["id"] in seen or seen.add(s["id"]))]
    trace.append(f"Seeds: resolved {len(seeds)}/{len(req.seeds)}" + (f"; NOT found: {unresolved}" if unresolved else ""))
    if not seeds:
        raise HTTPException(400, "None of the seeds could be resolved in OpenAlex. Use DOIs or exact titles.")
    seed_by_id = {s["id"]: s for s in seeds}

    # 2. candidates
    cands = gather_candidates(seeds, req.per_seed, req.include_references, trace)
    pool = cap_pool(cands, 250)
    trace.append(f"Pool for scoring: {len(pool)} candidates")
    if not pool:
        raise HTTPException(404, "No candidates found around these seeds.")

    # 3. scoring
    graph = minmax([len(c["links"]) / len(seeds) for c in pool])
    sem = minmax(semantic_scores(req.question, seeds, pool, trace))
    for c, g, s in zip(pool, graph, sem):
        c["graph"], c["semantic"] = g, s
        c["pre"] = 0.45 * g + 0.55 * s
    pool.sort(key=lambda c: c["pre"], reverse=True)

    # 4. LLM review of the top slice
    llm_on = bool(LLM_API_KEY)
    review_k = min(len(pool), req.top_n + 4) if llm_on else 0
    conns = {c["paper"]["id"]: connection_text(c, seed_by_id) for c in pool}
    if llm_on:
        with ThreadPoolExecutor(4) as ex:
            reviews = list(ex.map(
                lambda c: review_one(req.question, seeds, c, conns[c["paper"]["id"]][0], req.lang), pool[:review_k]))
        for c, rv in zip(pool[:review_k], reviews):
            c["review"] = rv
        ok = sum(1 for r in reviews if r["relevance"] is not None)
        trace.append(f"LLM review ({LLM_MODEL}): {ok}/{review_k} scored; "
                     f"{sum(1 for r in reviews if r['verified'])} quotes verified against abstracts")
    else:
        trace.append("WARNING no LLM key set: ranked by graph + semantic score only, no explanations")

    for c in pool:
        rv = c.get("review")
        if rv is None:
            c["final"] = c["pre"] if not llm_on else 0.2 * c["pre"]  # unreviewed -> below reviewed ones
        elif rv["relevance"] is None:
            c["final"] = 0.4 * c["pre"]
        else:
            c["final"] = 0.6 * rv["relevance"] / 10 + 0.4 * c["pre"]
    pool.sort(key=lambda c: c["final"], reverse=True)
    top = pool[: req.top_n]

    # 5. Tavily enrichment
    web_leads: list[dict] = []
    if TAVILY_API_KEY:
        with ThreadPoolExecutor(4) as ex:
            lead_f = ex.submit(safe_tavily, f"{req.question} new preprint bioRxiv arXiv", 5)
            enrich = list(ex.map(lambda c: safe_tavily(f"{c['paper']['title']} code github", 3), top[:5]))
            web_leads = lead_f.result()
        for c, links in zip(top[:5], enrich):
            c["web"] = links
        trace.append(f"Tavily: {len(web_leads)} web leads for the question, enrichment for top {len(enrich)} papers")
    else:
        trace.append("Tavily skipped (no TAVILY_API_KEY)")

    results = []
    for rank, c in enumerate(top, 1):
        p, rv = c["paper"], c.get("review") or {}
        text, links = conns[p["id"]]
        results.append({
            "rank": rank, "title": p["title"], "year": p["year"], "authors": p["authors"][:6],
            "venue": p["venue"], "doi": p["doi"], "url": p["url"], "openalex_url": p["openalex_url"],
            "cited_by_count": p["cited_by_count"], "abstract": p["abstract"][:1500],
            "connection": text, "links": links,
            "scores": {"graph": round(c["graph"], 2), "semantic": round(c["semantic"], 2),
                       "llm": rv.get("relevance"), "final": round(c["final"], 2)},
            "reason": rv.get("reason", ""), "evidence": rv.get("evidence", ""),
            "evidence_verified": rv.get("verified", False), "uncertainty": rv.get("uncertainty", ""),
            "web": c.get("web", []),
        })
    trace.append(f"Done in {time.time() - t0:.1f}s")
    return {
        "question": req.question,
        "seeds": [{"title": s["title"], "year": s["year"], "doi": s["doi"], "url": s["url"]} for s in seeds],
        "unresolved_seeds": unresolved, "results": results, "web_leads": web_leads, "trace": trace,
        "config": {"llm": LLM_MODEL if llm_on else None, "embeddings": EMBED_MODEL if EMBED_API_KEY else None,
                   "tavily": bool(TAVILY_API_KEY)},
    }


# --------------------------------------------------------------------------- routes
@app.post("/api/discover")
def api_discover(req: DiscoverRequest) -> dict:
    try:
        return discover(req)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"ok": True, "llm": bool(LLM_API_KEY), "llm_model": LLM_MODEL, "embeddings": bool(EMBED_API_KEY),
            "embed_model": EMBED_MODEL, "tavily": bool(TAVILY_API_KEY)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")
