# Legal Clause Intelligence Demo for GitHub Pages

This version upgrades the browser demo with stronger search and better clause visibility.

## What changed

The app now includes:

- **advanced hybrid retrieval**
  - browser embeddings with cosine similarity
  - BM25-style sparse lexical ranking
  - reciprocal rank fusion
  - title-aware boosts
  - clause-type-aware boosts
  - query expansion using legal synonyms
  - MMR reranking to reduce duplicate or overly similar results

- **better clause taxonomy**
  - clause type chips in the clause review panel
  - clause type chips in search results
  - a left-side facet panel showing the indexed clause types and counts
  - clause-type filtering for search results

- **better upload behavior**
  - clearer upload status messages
  - explicit parsing feedback by file
  - TXT, MD, PDF, and DOCX parsing on the client side

- **live-demo-friendly packaging**
  - static hosting ready
  - includes `.nojekyll` for GitHub Pages
  - works on GitHub Pages or Vercel

## Important technical note on FAISS

FAISS is an excellent production-grade vector indexing library, but it is primarily a server-side tool. A static GitHub Pages site cannot run `faiss-python` directly in the browser.

This demo therefore uses a **FAISS-like retrieval pattern** for a browser-native environment:
- local embedding generation
- in-memory vector storage
- cosine similarity ranking
- hybrid fusion with lexical retrieval
- reranking for diversity

That makes the public demo portable while still reflecting the architecture you would use in production.

### Production upgrade path

For a production or interview follow-up version, the architecture should be:

- frontend: static site or React app
- backend: Python or Node service
- vector index: **FAISS** or another ANN index
- retrieval: hybrid sparse plus dense search
- reranking: cross-encoder or MMR
- LLM layer: Azure OpenAI or OpenAI for grounded clause analysis

## Files

- `index.html` : main app shell
- `styles.css` : styling
- `app.js` : parsing, chunking, embeddings, retrieval, reranking
- `sample_contracts.js` : built-in sample contracts
- `.nojekyll` : prevents GitHub Pages from applying the Jekyll pipeline

## Local test

Use any static server.

### Python
```bash
python -m http.server 8000
```

Then open:
```bash
http://localhost:8000
```

## Best demo flow

1. Start with the sample contracts.
2. Search for:
   - `limitation of liability`
   - `confidential information`
   - `subprocessor obligations`
   - `termination for convenience`
   - `non-compete`
3. Click different clause-type facets like `liability`, `privacy`, or `termination`.
4. Show that the search results and clause inventory reflect those categories.
5. Switch to **Upload your own** and load a text-based PDF or DOCX.
6. Re-run search and export the JSON result.

## Resume-friendly description

Built a browser-native legal clause intelligence demo with structure-aware chunking, semantic embeddings, advanced hybrid retrieval, clause taxonomy filters, and reviewer-oriented clause analysis for contracts and legal documents.
