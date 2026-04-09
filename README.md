# Legal Clause Intelligence v2 for GitHub Pages

This is a browser-native legal AI demo designed to be hosted on GitHub Pages.

## What makes this version different

This version does not require a Python backend. It runs entirely in the browser and still demonstrates:
- client-side contract parsing
- structure-aware clause chunking
- browser-based vector embeddings
- hybrid retrieval with dense plus sparse search
- lightweight clause review and summarization
- uploads for TXT, MD, PDF, and DOCX

## How the AI portion works

The app uses `Transformers.js` in the browser with the `Xenova/all-MiniLM-L6-v2` embedding model.
That means visitors can see semantic retrieval working on GitHub Pages without exposing an API key.

The first load may take a little longer because the embedding model is downloaded into the browser.

## Files

- `index.html` : the main app shell
- `styles.css` : styling
- `app.js` : client-side parsing, chunking, embeddings, and retrieval
- `sample_contracts.js` : built-in sample contracts

## Deploy to GitHub Pages

1. Create a GitHub repo.
2. Upload all files from this package into the repo root.
3. Commit and push.
4. In GitHub, open **Settings > Pages**.
5. Under **Build and deployment**, choose **Deploy from a branch**.
6. Select the main branch and the root folder.
7. Save.

After GitHub publishes the site, your demo will be live.

## Local test

You can test with a simple static server.

### Python
```bash
python -m http.server 8000
```

Then open:
```bash
http://localhost:8000
```

### VS Code Live Server
You can also right click `index.html` and open with Live Server.

## Best demo flow

1. Start with the sample contracts.
2. Search for:
   - `limitation of liability`
   - `data breach notification`
   - `termination for convenience`
   - `non-compete`
3. Show the clause inventory and the clause review panel.
4. Upload a PDF or DOCX contract and re-run search live.

## Important note

This is a strong GitHub Pages demo, but it is still a static site. It does not perform privileged server-side LLM calls.

If you want a production-style version next, the best upgrade is:
- GitHub Pages frontend
- serverless API for LLM summarization and redlining
- optional Azure OpenAI or OpenAI backend
- citations and clause-grounded answer generation
