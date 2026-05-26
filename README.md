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

## Example: hybrid vs. term-only retrieval

Same query against the same index, retrieval strategy swapped. Both runs ask:
**"How did Daenerys Targaryen rise to power?"**

### Hybrid retrieval (BM25 + KNN)

```
CONTEXT:
 * Tywin's death upsets the balance of power in King's Landing, namely by
   allowing the rise to power of the High Sparrow and the Faith Militant.
 * The Prince discusses the use of amoral ways and "how to do wrong" to gain power.
 * Her rise to power is aided by the historic birth of three dragons, hatched
   from eggs given to her as wedding gifts.
 * Tyrion arrives in Pentos, where Varys reveals that he has been conspiring to
   restore House Targaryen to power, and asks Tyrion to journey with him to
   meet Daenerys Targaryen in Meereen.
 * Petyr helps Eddard expose the secret parentage of the royal children, but
   advises him to abet Joffrey's rise to power in order to consolidate their own.
 * Later, she conquers Yunkai and Meereen, the latter Daenerys settles in to
   learn how to rule.
 * After Daenerys conquers the city she continues to rule it as its queen to
   learn how to rule.
 * The season also features other storylines: Daenerys Targaryen begins her
   rise in power in Essos; Ned's illegitimate son, Jon Snow, goes undercover
   beyond the Wall; ... and all of Westeros is informed of the reawakened
   threat of the Army of the Dead.
 * It doesn't mean that she's insincere in her genuine hope to do charitable
   work and it's just an interesting new comment on how you handle the masses
   and how you win power that maybe we haven't seen in Westeros before.
 * Daenerys realizes this makes Jon the heir to House Targaryen.
```

> **Answer:** Daenerys Targaryen rose to power after the death of her uncle,
> Tywin Lannister, which upset the balance of power in King's Landing and
> allowed for the rise of the High Sparrow and the Faith Militant. Her own
> path to power was aided by a unique circumstance — she had been given three
> dragon eggs as wedding gifts, and upon hatching, these dragons made her a
> formidable force on her own.

### Term-only retrieval (BM25 only)

```
CONTEXT:
 * Tywin's death upsets the balance of power in King's Landing, namely by
   allowing the rise to power of the High Sparrow and the Faith Militant.
 * The Prince discusses the use of amoral ways and "how to do wrong" to gain power.
 * Her rise to power is aided by the historic birth of three dragons, hatched
   from eggs given to her as wedding gifts.
 * Tyrion arrives in Pentos, where Varys reveals that he has been conspiring to
   restore House Targaryen to power, and asks Tyrion to journey with him to
   meet Daenerys Targaryen in Meereen.
 * Petyr helps Eddard expose the secret parentage of the royal children, but
   advises him to abet Joffrey's rise to power in order to consolidate their own.
 * Later, she conquers Yunkai and Meereen, the latter Daenerys settles in to
   learn how to rule.
 * After Daenerys conquers the city she continues to rule it as its queen to
   learn how to rule.
 * After Daenerys conquers the city she continues to rule it as its queen to
   learn how to rule.
 * After Daenerys conquers the city she continues to rule it as its queen to
   learn how to rule.
 * The season also features other storylines: Daenerys Targaryen begins her
   rise in power in Essos; ...
```

> **Answer:** Daenerys Targaryen's rise to power was aided by the historic
> birth of three dragons, hatched from eggs given to her as wedding gifts.
> After acquiring these dragon eggs, she became known for her bravery and
> determination, eventually conquering Yunkai and Meereen, the latter being a
> city where she settled to learn how to rule.

### What this shows

Term-only retrieval returns the same sentence three times in a row —
*"After Daenerys conquers the city she continues to rule it as its queen to
learn how to rule."* — because three different Wikipedia articles contain that
exact sentence and BM25 scores each occurrence identically. The model is then
working with a context window that's effectively 7 unique sentences instead
of 10.

Hybrid retrieval dedupes those by score (each near-duplicate gets a slightly
different KNN distance from the query embedding) and pulls in semantically
adjacent material the term query misses — the Jon-as-heir line, the
"three-headed dragon" reference, the "rise in power in Essos" sentence — none
of which share strong keyword overlap with the query.

The downstream answer reflects this. The hybrid response cites both the
political vacuum and the dragons; the term-only response leans on the dragons
plus the duplicated conquering sentence and produces a thinner answer.

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
