# Game of Thrones Lore RAG

A retrieval-augmented question-answering system over ~24k sentences scraped
from Wikipedia articles about *A Song of Ice and Fire* and *Game of Thrones*.
Hybrid BM25 + dense-vector retrieval, local LLM inference, no paid APIs.

Built as a course project for INFO 4614 (Information & Data Retrieval) at CU
Boulder, then cleaned up for portfolio use.

**Stack:** `sentence-transformers` (all-MiniLM-L6-v2) · Elasticsearch 9.x (BM25 +
dense_vector KNN) · Ollama running Llama 3.2 · `wikipedia-api` · NLTK ·
Jupyter.

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

1. `01_build_index.ipynb` fetches the Wikipedia articles, splits them into
   sentences, embeds in batches of 256, and bulk-loads them into Elasticsearch
   with index refresh paused during ingest.
2. `02_query_demo.ipynb` shows keyword, BM25, KNN, and hybrid queries against
   the index.
3. `03_rag_pipeline.ipynb` is the full RAG loop: hybrid retrieval, prompt
   construction, generation via Ollama. It also has a side-by-side comparison
   of hybrid vs. term-only retrieval and a deliberately-out-of-corpus question
   used to check that the model declines instead of making things up.

## Example: hybrid vs. term-only retrieval

Same query against the same index, swapping retrieval strategy. Both ask:
**"How did Daenerys Targaryen rise to power?"**

### Hybrid (BM25 + KNN)

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
> path to power was aided by a unique circumstance: she had been given three
> dragon eggs as wedding gifts, and upon hatching, these dragons made her a
> formidable force on her own.

### Term-only (BM25)

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

The term-only run returns the same sentence three times in a row,
*"After Daenerys conquers the city she continues to rule it as its queen to
learn how to rule,"* because three different Wikipedia articles contain that
exact sentence and BM25 scores each copy identically. Out of a 10-sentence
context budget, three slots get wasted on a duplicate. The hybrid run dedupes
those by score (KNN gives each near-duplicate a slightly different distance)
and brings in semantically adjacent material the term query misses: the
Jon-as-heir line, the Targaryen-conquest reference, the "rise in power in
Essos" sentence. The hybrid answer reflects that wider context, citing both
the political vacuum and the dragons. The term-only answer mostly recycles the
dragon eggs.

## Design choices

**Hybrid retrieval.** The query is run as a single Elasticsearch `bool` with
both a `match` clause (BM25 over a porter-stemmed analyzer) and a `knn` clause
on the dense vector field, both as `should`. BM25 alone misses paraphrases
and over-rewards near-duplicate sentences. KNN alone misses exact-name
matches. Running them together and letting Elasticsearch sum the scores is
about as simple as hybrid retrieval gets, and the worked example above shows
why it's worth doing.

**Sentence-level chunks.** Splitting articles into sentences keeps each
retrieved chunk tightly on-topic and lets a 10-result context window cover a
lot of ground. The trade-off is lost coreference across sentences. A larger
system would use sliding-window chunks or a hierarchical retriever; this one
doesn't.

**Grounded prompting.** The system prompt instructs the model to answer only
from `CONTEXT`, decline when there isn't enough information, and prefix
responses with `Answer:`. The thing that convinced me this was working was the
out-of-corpus query about the *Game of Thrones* theme song lyrics. The
retrieved context was full of unrelated trivia (tourism campaigns, themed
whiskies). The model could have easily made something up from its
pre-training. Instead it said:

> Answer: The context provided does not include any information about the
> lyrics to the Game of Thrones theme song. The information in CONTEXT is
> limited to general statements about the series' portrayal of medieval
> realism and its connection to George R.R. Martin's novel series A Song of
> Ice and Fire, as well as some specific references to tourism campaigns and
> themed whiskies.

That refusal is a real win for the grounding rules. A 3B model is very
willing to confabulate when given a question it sort of half-remembers, and
the rules held.

**Indexing throughput.** Embeddings are batched 256 sentences per
`model.encode` call. Index refresh is set to `-1` during the bulk insert and
restored to `1s` after. Without those two changes the initial build takes
multiple times longer.

## Limitations

- Sentence-level chunking loses cross-sentence coreference.
- Llama 3.2 (3B) occasionally over-hedges in the other direction and refuses
  to answer questions the context *does* support. The Jon Snow query in
  notebook 3 is the clearest case: the context contains "Jon Snow is the
  bastard son of Eddard Stark" and the model still says it cannot speculate.
- Some Wikipedia pages redirect to compilation articles. The `Yara Greyjoy`
  lookup returns ~1,900 sentences because that title redirects to a much
  larger list page, and several other titles silently 404. A production
  ingest would resolve canonical URLs first.
- No reranker, no query rewriting, no eval harness. The hybrid-vs-term
  comparison in the README is the closest thing to evaluation in the repo.

## Setup

Prereqs: Python 3.11+, Docker, [Ollama](https://ollama.com).

```bash
# 1. Elasticsearch (single-node, security on)
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

Then run `01_build_index.ipynb` → `02_query_demo.ipynb` → `03_rag_pipeline.ipynb`.
The first notebook takes 5–10 minutes on a laptop, mostly sentence embedding.
