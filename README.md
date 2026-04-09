# Legal Clause Intelligence

An AI-powered contract review tool for extracting, analyzing, and semantically searching legal clauses across a contract corpus. Built with RAG architecture, vector embeddings, overlapping chunking, and FAISS-style cosine similarity search.

---

## Overview

Legal teams spend significant time manually reviewing contracts for risky or non-standard clauses. This tool accelerates that process by:

- Parsing contracts into semantically meaningful chunks
- Embedding each chunk into a high-dimensional vector space
- Enabling natural language semantic search across all contracts simultaneously
- Surfacing AI-generated summaries, risk ratings, and negotiation flags for each clause

The demo ships as a single self-contained HTML file with no external dependencies, making it instantly deployable on any static host or runnable locally in a browser.

---

## Features

### Semantic Search
Type natural language queries like *"termination without cause"* or *"IP ownership assignment"* and the engine retrieves the most relevant clauses using cosine similarity against 768-dimensional embeddings — not keyword matching.

### RAG Pipeline
Each query follows a full Retrieval-Augmented Generation pipeline:
```
PDF parse → Chunk (256 tokens / 64 overlap) → Embed (768-dim) → FAISS index → Cosine retrieval → LLM analysis
```

### Advanced Chunking
Documents are split using an overlapping sliding window strategy — 256-token chunks with a 64-token overlap — ensuring clauses that span chunk boundaries are never lost.

### Vector Embeddings
Every clause is represented as a 768-dimensional vector. The detail panel visualises the top 5 semantic dimensions for each clause (termination, liability, confidentiality, payment, IP/ownership), making the embedding space interpretable.

### FAISS-style Index
Retrieval uses an IVFFlat cosine similarity index (simulated in the demo, replaceable with a real FAISS backend). All results are ranked by similarity score shown as a percentage match bar.

### AI Clause Analysis
Each clause surfaces:
- Plain-English summary of the clause's legal effect
- Risk level (high / medium / low)
- Specific negotiation flags and market standard comparisons
- Key tags (e.g. *"5-year tail"*, *"fee acceleration"*, *"no pre-existing IP carve-out"*)

### Contract Corpus
5 sample contracts included out of the box:

| Contract | Type | Clauses |
|---|---|---|
| Acme Corp NDA | NDA | Confidentiality, Governing Law |
| CloudSoft MSA | MSA | Termination, Liability, Indemnification |
| DevWorks SOW | SOW | Payment, IP / Ownership |
| Holloway Employment | Employment | Non-Compete, Termination, Confidentiality |
| Zenith Office Lease | Lease | Termination, Liability |

### Filtering & Scoping
- Filter by clause type (Termination, Liability, Confidentiality, Payment, IP, Indemnification, Governing Law, Non-Compete)
- Toggle between searching all contracts or only the active document
- Click any contract in the sidebar to scope the corpus

---

## Getting Started

### Run locally
No build step or server required. Simply open the file in a browser:

```bash
open legal_clause_intelligence.html
```

Or serve it with any static file server:

```bash
npx serve .
# or
python3 -m http.server 8080
```

### Deploy to a static host

**Netlify**
```bash
# Drag and drop the HTML file onto netlify.com/drop
```

**GitHub Pages**
```bash
git init
git add legal_clause_intelligence.html
git commit -m "initial deploy"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
# Enable GitHub Pages in repo Settings → Pages → Deploy from branch: main
```

**AWS S3**
```bash
aws s3 cp legal_clause_intelligence.html s3://your-bucket/ --acl public-read
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Contract Corpus                       │
│   NDA · MSA · SOW · Employment · Lease                  │
└────────────────────────┬────────────────────────────────┘
                         │
                    Document Parse
                         │
              ┌──────────▼──────────┐
              │   Chunking Engine   │
              │  256 tokens / 64    │
              │  token overlap      │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Embedding Model    │
              │  768-dim vectors    │
              │  (text-embedding-   │
              │   3-small / Cohere) │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    FAISS Index      │
              │    IVFFlat          │
              │    Cosine sim       │
              └──────────┬──────────┘
                         │
                   Query / Search
                         │
              ┌──────────▼──────────┐
              │   Top-K Retrieval   │
              │   Ranked results    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │    LLM Analysis     │
              │    Summary · Risk   │
              │    Flags · Tags     │
              └─────────────────────┘
```

---

## Productionising

The demo ships with a simulated embedding engine (`pseudoEmbed()`) for zero-dependency portability. To connect real AI services, replace the following:

### 1. Real embeddings (OpenAI)

```javascript
async function embed(text) {
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: text,
    }),
  });
  const data = await res.json();
  return data.data[0].embedding; // 1536-dim float array
}
```

### 2. Real FAISS index (Python backend)

```python
import faiss
import numpy as np

# Build index
dimension = 1536
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim on normalised vectors

# Add clause embeddings
embeddings = np.array(clause_embeddings, dtype=np.float32)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Query
def search(query_embedding, k=10):
    q = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, k)
    return scores[0], indices[0]
```

### 3. Real document parsing

```python
# PDF extraction with Azure Document Intelligence
from azure.ai.formrecognizer import DocumentAnalysisClient

client = DocumentAnalysisClient(endpoint, credential)
poller = client.begin_analyze_document("prebuilt-document", document)
result = poller.result()

# Or with LangChain PDF loader
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("contract.pdf")
pages = loader.load_and_split()
```

### 4. LLM clause analysis

```python
from anthropic import Anthropic

client = Anthropic()

def analyze_clause(clause_text, clause_type):
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analyze this {clause_type} clause from a legal contract.
            
Clause: {clause_text}

Return a JSON object with:
- summary: plain-English explanation (2-3 sentences)
- risk: "high" | "med" | "low"
- flags: list of negotiation concerns
- tags: list of 2-4 key terms"""
        }]
    )
    return message.content[0].text
```

### 5. Recommended stack for production

| Layer | Technology |
|---|---|
| Document parsing | Azure Document Intelligence / AWS Textract |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-3-small` or Cohere `embed-v3` |
| Vector store | FAISS (local) or Pinecone / Weaviate (managed) |
| LLM analysis | Claude claude-opus-4-6 via Anthropic API |
| Backend API | FastAPI (Python) |
| Frontend | React + TypeScript |
| Auth | Auth0 / Azure AD |
| Storage | Azure Blob / S3 for contract PDFs |

---

## Clause Types Supported

| Type | Risk Indicators |
|---|---|
| Termination | Notice period, fee acceleration, cause vs. convenience |
| Liability | Cap amount, consequential exclusions, carve-outs |
| Confidentiality | Duration, scope, public domain exceptions |
| Payment | Net terms, interest rate, dispute window |
| IP / Ownership | Work-for-hire, assignment, pre-existing IP |
| Indemnification | Scope, defense obligation, cap interaction |
| Governing Law | Jurisdiction, arbitration, venue |
| Non-Compete | Duration, geography, enforceability |

---

## File Structure

```
legal_clause_intelligence.html   # Complete self-contained demo (HTML + CSS + JS)
README.md                        # This file
```

---

## Extending the Corpus

To add new contracts and clauses, edit the `CONTRACTS` and `CLAUSES` arrays in the `<script>` block:

```javascript
// Add a contract
CONTRACTS.push({
  id: 'c6',
  name: 'Vendor Supply Agreement',
  type: 'SUPPLY',
  badge: 'badge-sow',      // reuse an existing badge style
  date: '2025-01-20',
  parties: 'Acme Corp & GlobalSupply Ltd',
  clauses: ['cl13', 'cl14'],
});

// Add a clause
CLAUSES.push({
  id: 'cl13',
  contract: 'c6',
  type: 'Termination',
  risk: 'med',
  text: 'Either party may terminate this Agreement...',
  summary: 'Standard termination clause with 30-day notice...',
  tags: ['30-day notice', 'mutual'],
  chunk: 3,
  totalChunks: 8,
});
```

---

## License

MIT — free to use, modify, and deploy.

---

## Contributing

Issues and pull requests welcome. For major changes, please open an issue first to discuss the proposed change.
