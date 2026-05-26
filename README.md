# Game of Thrones Lore RAG

A retrieval-augmented question-answering system over ~24k sentences scraped from
Wikipedia articles about *A Song of Ice and Fire* / *Game of Thrones*. Hybrid
BM25 + dense-vector retrieval, local LLM inference, no API keys, no paid
services.

Built as a course project (CU Boulder INFO 4614, Information & Data Retrieval),
then cleaned up for portfolio use.

## Architecture

```
Wikipedia (68 articles)
        │  wikipedia-api + NLTK sentence tokenizer
        ▼
~24k sentences
        │  sentence-transformers/all-MiniLM-L6-v2 (384-d)
        ▼
Elasticsearch index `got_lore`
  ├─ sentence (text, porter-stemmed analyzer)   → BM25
  ├─ embedding (dense_vector, cosine)           → KNN
  └─ doc_title / category (keyword)             → filters
        │
        │  query → embed → bool{ match SHOULD knn }
        ▼
Top-k sentences  →  prompt template  →  Ollama (llama3.2)  →  answer
```

Three notebooks, run in order:

1. **`01_build_index.ipynb`** — fetches Wikipedia articles, sentence-tokenizes,
   embeds in batches of 256, bulk-indexes into Elasticsearch with refresh paused
   during ingest.
2. **`02_query_demo.ipynb`** — keyword, BM25, KNN, and hybrid queries against
   the index.
3. **`03_rag_pipeline.ipynb`** — full RAG loop: hybrid retrieval, prompt
   construction with explicit grounding rules, generation via Ollama. Includes a
   side-by-side comparison of hybrid vs. term-only retrieval and a failure-case
   query (out-of-corpus question) to show the model declining cleanly.

## Setup

Prereqs: Python 3.11+, Docker, [Ollama](https://ollama.com).

```bash
# 1. Elasticsearch (single-node, with security on)
docker run -d --name es \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "ELASTIC_PASSWORD=changeme" \
  docker.elastic.co/elasticsearch/elasticsearch:9.3.1

# 2. Ollama model
ollama pull llama3.2

# 3. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Config
cp .env.example .env   # then edit ES_PASSWORD to match the one above

# 5. Run the notebooks
jupyter lab
```

Then execute `01_build_index.ipynb` → `02_query_demo.ipynb` → `03_rag_pipeline.ipynb`.
The first notebook takes ~5–10 min on a laptop (most of it is sentence embedding).

## Design notes

- **Hybrid retrieval matters.** Pure BM25 returns near-duplicate sentences and
  misses paraphrases; pure KNN returns semantic neighbors that miss exact-name
  matches. The bool/should combination scores both. Notebook 3 shows a direct
  comparison on the same query.
- **Sentence-level granularity.** Splitting articles into sentences keeps each
  retrieved chunk tightly on-topic and lets a small context window hold ~10
  diverse hits. Trade-off: loses some inter-sentence context (a deliberate
  choice — see "Limitations").
- **Grounded prompting.** The system prompt enforces answering only from
  `CONTEXT`, declining when insufficient, and prefixing with `Answer:`. The
  failure-case query in notebook 3 confirms the model abstains rather than
  hallucinating from training data.
- **Indexing throughput.** Embeddings are batched (256 sentences/call); index
  refresh is paused during bulk insert and restored after.

## Limitations (honest)

- Sentence-level chunking drops cross-sentence coreference. A larger system
  would use sliding-window chunks or hierarchical retrieval.
- Llama 3.2 (3B) sometimes over-hedges on context that clearly contains the
  answer (see the Jon Snow query in notebook 3). A larger model or a
  reranker would help.
- Wikipedia article quality varies — some "character" articles are stubs and a
  few redirects resolve to unrelated content (the `Yara Greyjoy` page returns
  ~1900 sentences because it redirects to a larger compilation). A production
  ingest would validate canonical URLs.
- No reranking, no query rewriting, no eval harness. This is a working pipeline,
  not a tuned one.

## Stack

`sentence-transformers` · `elasticsearch-py` (KNN + BM25) · `ollama` (llama3.2) ·
`wikipedia-api` · `nltk` · Jupyter
