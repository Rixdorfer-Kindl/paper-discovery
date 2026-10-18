"""Offline smoke test: mocks OpenAlex + LLM + embeddings, checks the pipeline logic.

Run:  python test_smoke.py
Missing third-party modules (fastapi/httpx) are stubbed so this also runs in a bare sandbox;
when they are installed the real ones are used.
"""
import sys
import types

try:
    import fastapi  # noqa: F401
except ImportError:
    fa = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code, detail=""):
            self.status_code, self.detail = status_code, detail

    class FastAPI:
        def __init__(self, **kw): pass
        def post(self, *a, **k): return lambda f: f
        def get(self, *a, **k): return lambda f: f

    fa.FastAPI, fa.HTTPException = FastAPI, HTTPException
    resp = types.ModuleType("fastapi.responses")
    resp.HTMLResponse = object
    sys.modules.update({"fastapi": fa, "fastapi.responses": resp})

try:
    import httpx  # noqa: F401
except ImportError:
    hx = types.ModuleType("httpx")
    hx.HTTPError = type("HTTPError", (Exception,), {})
    hx.get = lambda *a, **k: (_ for _ in ()).throw(hx.HTTPError("network disabled in test"))
    sys.modules["httpx"] = hx

import app  # noqa: E402


def W(i, title, year, abstract="", refs=(), related=(), cited=0, doi=None):
    inv: dict = {}
    for pos, word in enumerate(abstract.split()):
        inv.setdefault(word, []).append(pos)
    return {
        "id": f"https://openalex.org/{i}", "doi": f"https://doi.org/{doi}" if doi else None, "title": title,
        "publication_year": year, "publication_date": f"{year}-06-01",
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "abstract_inverted_index": inv or None,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "related_works": [f"https://openalex.org/{r}" for r in related],
        "cited_by_count": cited, "primary_location": {"source": {"display_name": "Nature"}},
    }


ABS_A = "We present a diffusion model for protein backbone design that produces binders validated by crystallography."
ABS_B = "A graph neural network predicts amino acid sequences for fixed protein backbones with high recovery."
SEEDS = {
    "S1": W("S1", "Diffusion for protein design", 2023, ABS_A, refs=["R1"], doi="10.1000/s1", cited=900),
    "S2": W("S2", "Sequence design with message passing", 2022, ABS_B, refs=["R1", "R2"], doi="10.1000/s2", cited=700),
}
WORKS = {
    "C1": W("C1", "Antibody design with diffusion and sequence models", 2025,
            "We combine diffusion backbone generation with graph neural network sequence design for antibodies.", cited=40),
    "C2": W("C2", "Single-cell RNA atlas of melanoma", 2024, "We profile immune cells in melanoma tumours.", cited=900),
    "C3": W("C3", "Diffusion for protein design", 2022, ABS_A, cited=5),            # preprint twin of seed S1
    "C4": W("C4", "Protein language model embeddings", 2024, "", cited=10),          # no abstract
    "R1": W("R1", "Deep learning for protein structure prediction", 2020,
            "Neural networks predict protein structure from sequence.", cited=5000),
    "R2": W("R2", "Rosetta energy functions", 2017, "Physics based protein design energy functions.", cited=3000),
}
CITES = {"S1": ["C1", "C2", "C3"], "S2": ["C1", "C4"]}


def fake_oa_get(path, **p):
    if path.startswith("/works/https://doi.org/"):
        doi = path.split("doi.org/")[1]
        return next((w for w in SEEDS.values() if w["doi"].endswith(doi)), None)
    if path == "/works":
        f = p.get("filter", "")
        if f.startswith("cites:"):
            return {"results": [WORKS[i] for i in CITES.get(f[6:], [])]}
        if f.startswith("openalex_id:"):
            return {"results": [WORKS[i] for i in f.split(":")[1].split("|") if i in WORKS]}
        if "search" in p:
            return {"results": [SEEDS["S2"]] if "message" in p["search"] else []}
    return None


app.oa_get = fake_oa_get
app.LLM_API_KEY = app.EMBED_API_KEY = app.TAVILY_API_KEY = ""


def run(**kw):
    req = app.DiscoverRequest(question="diffusion and sequence design for protein binders",
                              seeds=["10.1000/s1", "https://doi.org/10.1000/s2"], top_n=4, **kw)
    return app.discover(req)


# 1. no keys: graph + TF-IDF fallback --------------------------------------------------------
out = run()
titles = [r["title"] for r in out["results"]]
assert out["results"][0]["title"].startswith("Antibody design"), titles       # cites BOTH seeds
assert "Diffusion for protein design" not in titles, "seed / preprint twin must be excluded"
assert len(titles) == len(set(titles)), "duplicates"
assert any("WARNING" in t for t in out["trace"]), out["trace"]
assert out["results"][0]["links"] and len(out["results"][0]["links"]) == 2
print("1 ok  (no keys) ranking:", titles)

# 2. LLM on + fake embeddings: quote verification and rank influence -------------------------
app.LLM_API_KEY = "x"
app.EMBED_API_KEY = "x"
app.embed = lambda texts: [[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts]


def fake_llm(prompt):
    if "Antibody design" in prompt:
        return {"relevance": 9, "reason": "directly combines both methods",
                "evidence": "diffusion backbone generation with graph neural network sequence design",
                "uncertainty": "no wet-lab data in abstract"}
    if "melanoma" in prompt:
        return {"relevance": 1, "reason": "different field", "evidence": "invented sentence not in abstract",
                "uncertainty": ""}
    return {"relevance": 5, "reason": "background", "evidence": "", "uncertainty": "n/a"}


app.llm_json = fake_llm
out = run()
r0 = out["results"][0]
assert r0["title"].startswith("Antibody") and r0["scores"]["llm"] == 9 and r0["evidence_verified"] is True, r0
mel = next(r for r in out["results"] if "melanoma" in r["title"])
assert mel["evidence_verified"] is False and mel["scores"]["llm"] == 1
noabs = [r for r in out["results"] if r["title"].startswith("Protein language")]
assert not noabs or noabs[0]["scores"]["llm"] is None
assert any("embeddings" in t for t in out["trace"])
print("2 ok  (LLM on) trace:", *out["trace"], sep="\n    ")

# 3. parsing + evidence helpers --------------------------------------------------------------
assert app.parse_json('<think>hmm {x}</think>```json\n{"relevance": 7}\n```') == {"relevance": 7}
assert app.parse_json("no json here") is None
assert app.evidence_verified("Graph neural network  predicts", "A graph neural network predicts amino acids")
assert not app.evidence_verified("short", "short text")
print("3 ok  (parsers)")

# 4. unresolved seed handling ----------------------------------------------------------------
out = app.discover(app.DiscoverRequest(question="q", seeds=["10.1000/s1", "10.9999/nope"], top_n=2))
assert out["unresolved_seeds"] == ["10.9999/nope"]
try:
    app.discover(app.DiscoverRequest(question="q", seeds=["10.9999/nope"]))
    raise SystemExit("expected failure")
except Exception as e:  # HTTPException
    assert getattr(e, "status_code", None) == 400
print("4 ok  (unresolved seeds)")
print("ALL PASSED")
