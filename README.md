# Legal Document RAG Pipeline

> Query legal and regulatory documents using natural language.
> Two implementations — one with LangChain, one built from scratch using raw Python libraries.

---

## What This Project Does

This project builds a **Retrieval-Augmented Generation (RAG)** system over public legal
and regulatory documents (EU AI Act, NIST AI RMF, CFPB Supervision Manual, and more).

Instead of asking an LLM to answer from memory — where it might hallucinate or have
outdated knowledge — we:

1. Pre-process and embed all documents into a vector store
2. At query time, retrieve the most relevant passages
3. Inject those passages into the LLM prompt so it answers *from the documents*

```
User Query
    │
    ▼
[Embed query] ──► [Search vector store] ──► [Top-k relevant chunks]
                                                       │
                                                       ▼
                                          [Build grounded prompt]
                                                       │
                                                       ▼
                                         [LLM generates cited answer]
```

---

## Two Implementations

### `rag_with_langchain/`
Uses **LangChain** + **FAISS** + **HuggingFace Embeddings**.
Good for fast prototyping — less boilerplate, more abstractions.

### `rag_from_scratch/`
Uses **pypdf** + **sentence-transformers** + **FAISS** + raw **numpy**.
No frameworks. Every step is explicit and fully understood.
Good for learning, customization, and production control.

---

## Documents Used

All documents are free, public, and require no login to download.

| Document | Description | Pages |
|---|---|---|
| [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) | AI Risk Management Framework — Govern, Map, Measure, Manage | ~50 |
| [NIST GenAI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | GenAI-specific risk companion to AI RMF | ~60 |
| [EU AI Act](https://artificialintelligenceact.eu/wp-content/uploads/2024/04/TA-9-2024-0138_EN.pdf) | World's first comprehensive AI regulation (2024) | ~144 |
| [CFPB Supervision Manual](https://files.consumerfinance.gov/f/documents/cfpb_supervision-and-examination-manual.pdf) | US financial consumer protection examination guide | ~900 |

---

## Project Structure

```
legal-rag-pipeline/
│
│
├── docs/                        # downloaded PDFs go here (gitignored)
├── indexes/                     # saved FAISS indexes (gitignored)
├── notebooks/
│   ├── RAG_with_LangChain.ipynb
│   └── RAG_from_Scratch.ipynb
│
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/legal-rag-pipeline.git
cd legal-rag-pipeline
```

### 2. Install dependencies

**For the LangChain version:**
```bash
pip install langchain langchain-community langchain-text-splitters \
            langchain-openai faiss-cpu pypdf sentence-transformers tiktoken
```

**For the from-scratch version:**
```bash
pip install pypdf sentence-transformers faiss-cpu numpy requests \
            openai anthropic google-generativeai
```

### 3. Download documents

```bash
python rag_from_scratch/01_download_docs.py
# Downloads all PDFs into docs/
```

### 4. Build the index

```bash
# From scratch
python rag_from_scratch/08_pipeline.py --build-index

# LangChain
python rag_with_langchain/03_embed_and_store.py
```

### 5. Query it

```python
from rag_from_scratch.pipeline import rag_query

result = rag_query(
    query="What are the obligations for providers of high-risk AI systems?",
    llm="gemini"   # or "openai" | "claude"
)

print(result["answer"])
for s in result["sources"]:
    print(f"  • {s['file']} | Page {s['page']} | Score {s['score']:.3f}")
```

---

## Pipeline Walkthrough (From Scratch)

### Step 1 — Parse PDFs

Extract text page-by-page using `pypdf`. Each page carries metadata (filename, page number)
so every retrieved chunk is traceable to its exact source.

```python
reader = pypdf.PdfReader("docs/eu_ai_act.pdf")
pages = [{"text": p.extract_text(), "page": i+1} for i, p in enumerate(reader.pages)]
```

### Step 2 — Chunk

Split pages into overlapping windows of ~1000 characters with 150-character overlap.
Overlap prevents context from being lost at chunk boundaries.

```python
# Sliding window: start=0, end=1000, next start=850 (1000-150 overlap)
chunks = chunk_pages(pages, chunk_size=1000, overlap=150)
```

### Step 3 — Embed

Convert each chunk into a 384-dimensional dense vector using a local HuggingFace model.
No API key needed. Vectors are L2-normalized so cosine similarity = dot product.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, normalize_embeddings=True)
# shape: (num_chunks, 384)
```

### Step 4 — Index with FAISS

Store all embeddings in a FAISS `IndexFlatIP` (exact inner-product search).
Save to disk so you never re-embed unless documents change.

```python
import faiss

index = faiss.IndexFlatIP(384)
index.add(embeddings)
faiss.write_index(index, "indexes/legal_docs.faiss")
```

### Step 5 — Retrieve

At query time, embed the user's question and find the k most similar chunks.

```python
query_vec = embedding_model.encode([query], normalize_embeddings=True)
scores, indices = index.search(query_vec, k=5)
# Returns top-5 chunks by cosine similarity
```

### Step 6 — Generate

Inject retrieved chunks into a strict prompt template and send to an LLM.
The grounding instruction prevents the model from using outside knowledge.

```python
prompt = f"""Answer using ONLY the context below. Cite source and page number.
If the answer is not in the context, say "I cannot find this in the documents."

{context_block}

Question: {query}
Answer:"""
```

---

## Supported LLMs

The generation step is modular — swap LLMs with a single parameter.

| LLM | Parameter | Notes |
|---|---|---|
| Google Gemini 1.5 Pro | `llm="gemini"` | Free tier available via Google AI Studio |
| OpenAI GPT-4o mini | `llm="openai"` | Cheap, very capable |
| Anthropic Claude Haiku | `llm="claude"` | Fast and cost-efficient |
| Ollama (local) | `llm="ollama"` | Fully offline, no API key needed |

---

## LangChain vs From Scratch — When to Use Which

| Concern | LangChain | From Scratch |
|---|---|---|
| Speed to prototype | Fast | Slower |
| Understanding internals | Abstracted | Fully visible |
| Customizing chunking logic | Limited | Complete control |
| Debugging retrieval | Hard (black box) | Easy (inspect every step) |
| Production flexibility | Framework-dependent | No lock-in |
| Adding new PDF types | Easy via loaders | Write your own parser |

**Recommendation:** Start with LangChain to validate the idea. Rewrite from scratch
when you need to debug, optimize, or control exactly what happens at each step.

---

## Key Design Decisions

**Why `all-MiniLM-L6-v2` for embeddings?**
It's free, runs on CPU, downloads once at ~90MB, and produces strong general-purpose
embeddings. For domain-specific legal text at scale, consider `BAAI/bge-large-en-v1.5`.

**Why FAISS `IndexFlatIP` and not an approximate index?**
Exact search is fast enough for document sets under ~500K chunks. For larger corpora,
switch to `IndexIVFFlat` (cluster-based) or `IndexHNSWFlat` (graph-based).

**Why chunk at 1000 characters with 150 overlap?**
Legal text has long sentences and complex clause structures. Smaller chunks (300–500)
lose context; larger chunks (2000+) dilute the embedding signal. 150-char overlap
ensures no sentence is split between two chunks without representation.

**Why `temperature=0` on the LLM?**
RAG for legal/regulatory Q&A is a factual retrieval task, not a creative one.
Temperature 0 makes the LLM deterministic and less likely to embellish beyond the context.

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| `HTTPError 403` on PDF download | Server blocks `urllib` user-agent | Use `requests` with a browser user-agent |
| `ModuleNotFoundError: langchain.text_splitter` | LangChain restructured packages | `pip install langchain-text-splitters` |
| `AttributeError: no attribute 'encode'` | Variable name collision (`model` reused) | Use `embedding_model` vs `gemini_model` vs `openai_client` |
| `AttributeError: 'Anthropic' has no 'chat'` | `client` overwritten by Anthropic client | Use `openai_client` and `claude_client` separately |
| Empty chunks from PDF | Image-based/scanned PDF | Use `pytesseract` OCR |
| Garbled text from PDF | Complex font encoding | Switch `pypdf` → `pymupdf` |

---

## What to Build Next

- **Hybrid search** — combine FAISS (semantic) with BM25 (keyword) using Reciprocal
  Rank Fusion. Catches exact article numbers like "Article 9" that semantic search misses.
- **Re-ranking** — after retrieving top-10 chunks, use a `CrossEncoder` to re-score and
  return the true top-3. Improves precision significantly.
- **Conversation memory** — include prior Q&A turns in the prompt for follow-up questions.
- **Streaming** — use `stream=True` in OpenAI/Claude APIs for token-by-token UI output.
- **Metadata filtering** — pre-filter FAISS search by source document before similarity search.

---

## References

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [EU Artificial Intelligence Act — Official Journal](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [CFPB Supervision Manual](https://www.consumerfinance.gov/compliance/supervision-examinations/)

---

## License

MIT — use freely, modify openly, attribute kindly.

---
## Contact

For any questions or feedback, please contact [Siddharamyya M](mailto:msidrm455@gmail.com).

🌐 [my Website](https://siddharamayya.in)

🌐 [Portfolio](https://portfolio.siddharamayya.in)

📞 +91 9740671620
