import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
import { SAMPLE_CONTRACTS } from './sample_contracts.js';

const state = {
  corpus: { ...SAMPLE_CONTRACTS },
  chunks: [],
  embeddings: [],
  extractor: null,
  selectedContract: null,
  selectedChunkIndex: 0,
};

const RISK_PATTERNS = {
  high: [
    /as is/i,
    /revocable license/i,
    /terminate for convenience/i,
    /binding arbitration/i,
    /non-compete/i,
    /audit/i,
    /without undue delay/i,
  ],
  medium: [
    /limitation of liability/i,
    /assignment/i,
    /indemnify/i,
    /governing law/i,
    /late payments/i,
    /subprocessor/i,
  ],
};

const CLAUSE_KEYWORDS = {
  confidentiality: ['confidential', 'proprietary', 'non-public', 'trade secrets'],
  liability: ['liability', 'damages', 'consequential', 'punitive', 'cap'],
  indemnification: ['indemnify', 'hold harmless', 'defend'],
  termination: ['terminate', 'termination', 'term', 'breach', 'convenience'],
  governing_law: ['governing law', 'laws of', 'jurisdiction'],
  security: ['security', 'encryption', 'safeguards', 'access controls', 'logging'],
  privacy: ['personal data', 'processor', 'controller', 'subprocessor', 'breach'],
  ip: ['intellectual property', 'inventions', 'works of authorship', 'non-infringement'],
  payment: ['payment', 'fees', 'invoices', 'interest'],
  employment: ['at will', 'non-compete', 'employment', 'arbitration'],
};

const el = {
  corpusMode: document.getElementById('corpusMode'),
  uploadWrap: document.getElementById('uploadWrap'),
  fileInput: document.getElementById('fileInput'),
  contractSelect: document.getElementById('contractSelect'),
  searchInput: document.getElementById('searchInput'),
  searchBtn: document.getElementById('searchBtn'),
  reindexBtn: document.getElementById('reindexBtn'),
  status: document.getElementById('status'),
  contractsStat: document.getElementById('contractsStat'),
  chunksStat: document.getElementById('chunksStat'),
  embedStat: document.getElementById('embedStat'),
  contractViewer: document.getElementById('contractViewer'),
  chunkSelect: document.getElementById('chunkSelect'),
  clauseTitle: document.getElementById('clauseTitle'),
  clauseTokens: document.getElementById('clauseTokens'),
  clauseTypes: document.getElementById('clauseTypes'),
  riskLevel: document.getElementById('riskLevel'),
  clauseText: document.getElementById('clauseText'),
  clauseSummary: document.getElementById('clauseSummary'),
  whyMatters: document.getElementById('whyMatters'),
  reviewSuggestions: document.getElementById('reviewSuggestions'),
  inventoryTableWrap: document.getElementById('inventoryTableWrap'),
  results: document.getElementById('results'),
  exportBtn: document.getElementById('exportBtn'),
};

function normalize(text) {
  return text.replace(/\s+/g, ' ').trim();
}

function estimateTokens(text) {
  return Math.max(1, Math.ceil(text.split(/\s+/).filter(Boolean).length * 1.3));
}

function splitIntoSections(contractText) {
  const parts = contractText.trim().split(/\n(?=(?:\d+\.|\d+\)|Section\s+\d+|[A-Z][A-Z \-/&]{4,})[^\n]*)/g).filter(Boolean);
  let cursor = 0;
  const results = parts.map((section) => {
    const clean = section.trim();
    const lines = clean.split('\n').map(s => s.trim()).filter(Boolean);
    const first = lines[0] || 'General';
    const m = first.match(/^((?:\d+\.|\d+\)|Section\s+\d+)\s*[^\n]+)/i);
    const title = m ? m[1] : first.slice(0, 90);
    const startIdx = cursor;
    const endIdx = cursor + clean.length;
    cursor = endIdx + 1;
    return { title, text: clean, startIdx, endIdx };
  });

  if (results.length <= 1) {
    const paras = contractText.split(/\n\s*\n/g).map(p => p.trim()).filter(Boolean);
    let c = 0;
    return paras.map((p, i) => {
      const item = { title: `Paragraph ${i + 1}`, text: p, startIdx: c, endIdx: c + p.length };
      c += p.length + 2;
      return item;
    });
  }

  return results;
}

function chunkSection(title, sectionText, maxWords = 120, overlapWords = 25) {
  const body = normalize(sectionText);
  const words = body.split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) {
    return [`${title} | ${body}`];
  }
  const chunks = [];
  let start = 0;
  while (start < words.length) {
    const end = Math.min(words.length, start + maxWords);
    chunks.push(`${title} | ${words.slice(start, end).join(' ')}`);
    if (end === words.length) break;
    start = Math.max(0, end - overlapWords);
  }
  return chunks;
}

function buildChunks(corpus) {
  const chunks = [];
  Object.entries(corpus).forEach(([contractName, text]) => {
    splitIntoSections(text).forEach(({ title, text: sectionText, startIdx }) => {
      let offset = startIdx;
      chunkSection(title, sectionText).forEach((piece) => {
        chunks.push({
          contractName,
          clauseTitle: title,
          text: piece,
          startIdx: offset,
          endIdx: offset + piece.length,
          tokensEst: estimateTokens(piece),
        });
        offset += piece.length;
      });
    });
  });
  return chunks;
}

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9_]+/g) || []);
}

function buildBm25(chunks) {
  const docs = chunks.map(c => tokenize(c.text));
  const df = new Map();
  const N = docs.length;

  docs.forEach(tokens => {
    const unique = new Set(tokens);
    unique.forEach(t => df.set(t, (df.get(t) || 0) + 1));
  });

  const avgdl = docs.reduce((a, d) => a + d.length, 0) / Math.max(1, N);
  return { docs, df, N, avgdl, k1: 1.5, b: 0.75 };
}

function bm25Scores(query, bm25) {
  const q = tokenize(query);
  return bm25.docs.map((docTokens) => {
    const tf = new Map();
    docTokens.forEach(t => tf.set(t, (tf.get(t) || 0) + 1));
    let score = 0;
    q.forEach(term => {
      const termDf = bm25.df.get(term) || 0;
      if (!termDf) return;
      const idf = Math.log(1 + (bm25.N - termDf + 0.5) / (termDf + 0.5));
      const freq = tf.get(term) || 0;
      const denom = freq + bm25.k1 * (1 - bm25.b + bm25.b * (docTokens.length / bm25.avgdl));
      score += idf * ((freq * (bm25.k1 + 1)) / Math.max(1e-9, denom));
    });
    return score;
  });
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function l2norm(v) {
  const mag = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
  return v.map(x => x / mag);
}

async function loadModel() {
  el.status.textContent = 'Loading browser embedding model. The first run may take a minute while model files download.';
  state.extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  el.status.textContent = 'Embedding model loaded. Building index...';
  el.embedStat.textContent = 'Ready';
}

async function embedTexts(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i++) {
    const result = await state.extractor(texts[i], { pooling: 'mean', normalize: true });
    const arr = Array.from(result.data);
    out.push(l2norm(arr));
    el.status.textContent = `Embedding chunk ${i + 1} of ${texts.length}...`;
  }
  return out;
}

async function rebuildIndex() {
  el.status.textContent = 'Preparing clause chunks...';
  state.chunks = buildChunks(state.corpus);
  state.bm25 = buildBm25(state.chunks);
  el.contractsStat.textContent = String(Object.keys(state.corpus).length);
  el.chunksStat.textContent = String(state.chunks.length);

  if (!state.extractor) {
    await loadModel();
  }

  const texts = state.chunks.map(c => c.text);
  state.embeddings = await embedTexts(texts);
  el.status.textContent = 'Index ready. You can search contracts now.';
  renderContractSelect();
  renderInventory();
  updateSelectedContract(Object.keys(state.corpus)[0]);
}

function detectClauseTypes(text) {
  const lower = text.toLowerCase();
  const found = Object.entries(CLAUSE_KEYWORDS)
    .filter(([, terms]) => terms.some(term => lower.includes(term)))
    .map(([k]) => k);
  return found.length ? found : ['general'];
}

function riskAssessment(text) {
  const why = [];
  const high = RISK_PATTERNS.high.filter(rx => rx.test(text)).map(rx => rx.source);
  const medium = RISK_PATTERNS.medium.filter(rx => rx.test(text)).map(rx => rx.source);
  let riskLevel = 'Low';
  if (high.length) riskLevel = 'High';
  else if (medium.length) riskLevel = 'Medium';

  const lower = text.toLowerCase();
  if (lower.includes('as is')) why.push('Contains broad warranty disclaimer language.');
  if (lower.includes('terminate for convenience')) why.push('Includes convenience termination, which may create continuity risk.');
  if (lower.includes('binding arbitration')) why.push('Routes disputes to arbitration and may affect dispute strategy.');
  if (lower.includes('non-compete')) why.push('Contains post-employment restrictions that may require jurisdiction-specific review.');
  if (lower.includes('liability')) why.push('Limits financial exposure and should be checked for cap structure and carve-outs.');
  if (lower.includes('indemn')) why.push('Creates third-party claim allocation obligations.');

  return {
    riskLevel,
    triggers: { high, medium },
    explanation: why.length ? why : ['No major heuristic risk signals detected in this clause.'],
  };
}

function summarizeClause(text) {
  const sentences = normalize(text).split(/(?<=[.!?])\s+/);
  const summary = sentences.slice(0, 2).join(' ');
  return summary.length > 320 ? `${summary.slice(0, 317)}...` : summary;
}

function reviewClause(text) {
  const lower = text.toLowerCase();
  const recommendations = [];
  if (lower.includes('liability') && !lower.includes('gross negligence')) {
    recommendations.push('Consider adding carve-outs for gross negligence, willful misconduct, confidentiality, data breach, and IP infringement.');
  }
  if (lower.includes('terminate for convenience')) {
    recommendations.push('Validate notice period, transition support, and any early termination obligations.');
  }
  if (lower.includes('as is')) {
    recommendations.push('Assess whether service levels, uptime commitments, or express warranties should be added.');
  }
  if (lower.includes('audit')) {
    recommendations.push('Limit audit frequency, scope, timing, and cost allocation.');
  }
  if (lower.includes('subprocessor')) {
    recommendations.push('Confirm objection rights, notice obligations, and downstream data protection commitments.');
  }
  if (lower.includes('arbitration')) {
    recommendations.push('Confirm governing rules, venue, confidentiality, and emergency relief language.');
  }
  if (!recommendations.length) {
    recommendations.push('Clause appears generally standard, but confirm alignment with fallback language and negotiation playbooks.');
  }

  return {
    clauseTypes: detectClauseTypes(text),
    summary: summarizeClause(text),
    risk: riskAssessment(text),
    recommendations,
  };
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function renderContractSelect() {
  el.contractSelect.innerHTML = Object.keys(state.corpus)
    .map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
    .join('');
}

function renderChunkSelect(contractName) {
  const chunks = state.chunks.filter(c => c.contractName === contractName);
  el.chunkSelect.innerHTML = chunks
    .map((c, i) => `<option value="${i}">${escapeHtml(`${c.clauseTitle} | ${c.text.slice(0, 90)}...`)}</option>`)
    .join('');
}

function renderClause(contractName, localChunkIndex = 0) {
  const chunks = state.chunks.filter(c => c.contractName === contractName);
  if (!chunks.length) return;
  const chosen = chunks[Math.min(localChunkIndex, chunks.length - 1)];
  const review = reviewClause(chosen.text);

  el.clauseTitle.textContent = chosen.clauseTitle;
  el.clauseTokens.textContent = String(chosen.tokensEst);
  el.clauseTypes.textContent = review.clauseTypes.join(', ');
  el.riskLevel.textContent = review.risk.riskLevel;
  el.clauseText.textContent = chosen.text;
  el.clauseSummary.textContent = review.summary;

  el.whyMatters.innerHTML = review.risk.explanation.map(x => `<li>${escapeHtml(x)}</li>`).join('');
  el.reviewSuggestions.innerHTML = review.recommendations.map(x => `<li>${escapeHtml(x)}</li>`).join('');
}

function renderInventory() {
  const rows = [];
  Object.entries(state.corpus).forEach(([contractName, text]) => {
    splitIntoSections(text).forEach(({ title, text: sectionText }) => {
      const review = reviewClause(sectionText);
      rows.push({
        contractName,
        clauseTitle: title,
        riskLevel: review.risk.riskLevel,
        clauseTypes: review.clauseTypes.join(', '),
        summary: review.summary.slice(0, 180),
      });
    });
  });

  el.inventoryTableWrap.innerHTML = `
    <table class="inventory-table">
      <thead>
        <tr>
          <th>Contract</th>
          <th>Clause</th>
          <th>Risk</th>
          <th>Type</th>
          <th>Summary</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => `
          <tr>
            <td>${escapeHtml(row.contractName)}</td>
            <td>${escapeHtml(row.clauseTitle)}</td>
            <td>${escapeHtml(row.riskLevel)}</td>
            <td>${escapeHtml(row.clauseTypes)}</td>
            <td>${escapeHtml(row.summary)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

function updateSelectedContract(contractName) {
  state.selectedContract = contractName;
  el.contractSelect.value = contractName;
  el.contractViewer.value = state.corpus[contractName] || '';
  renderChunkSelect(contractName);
  renderClause(contractName, 0);
}

async function hybridSearch(query, topK = 6) {
  if (!query.trim()) return [];
  el.status.textContent = 'Embedding query and ranking results...';

  const bm25 = bm25Scores(query, state.bm25);
  const qResult = await state.extractor(query, { pooling: 'mean', normalize: true });
  const qVec = l2norm(Array.from(qResult.data));
  const dense = state.embeddings.map(v => dot(v, qVec));

  const sparseOrder = bm25.map((score, idx) => ({ idx, score })).sort((a, b) => b.score - a.score).slice(0, 10);
  const denseOrder = dense.map((score, idx) => ({ idx, score })).sort((a, b) => b.score - a.score).slice(0, 10);

  const combined = new Map();
  sparseOrder.forEach((item, rank) => {
    const current = combined.get(item.idx) || { sparse: 0, dense: 0 };
    current.sparse = 1 / (rank + 21);
    combined.set(item.idx, current);
  });
  denseOrder.forEach((item, rank) => {
    const current = combined.get(item.idx) || { sparse: 0, dense: 0 };
    current.dense = 1 / (rank + 21);
    combined.set(item.idx, current);
  });

  const ranked = Array.from(combined.entries())
    .map(([idx, score]) => ({ idx, total: 0.45 * score.sparse + 0.55 * score.dense }))
    .sort((a, b) => b.total - a.total)
    .slice(0, topK)
    .map(item => ({ ...state.chunks[item.idx], score: item.total }));

  el.status.textContent = 'Search complete.';
  return ranked;
}

function renderResults(results) {
  if (!results.length) {
    el.results.innerHTML = '<div class="results-empty">No results found. Try a broader query.</div>';
    return;
  }

  el.results.innerHTML = results.map((r, i) => {
    const review = reviewClause(r.text);
    return `
      <div class="result">
        <div class="result-top">
          <div>
            <div class="result-title">Result ${i + 1}: ${escapeHtml(r.contractName)}</div>
            <div class="result-meta">${escapeHtml(r.clauseTitle)}</div>
          </div>
          <div class="result-score">score=${r.score.toFixed(4)}</div>
        </div>
        <div class="subsection">
          <h4>Chunk</h4>
          <p>${escapeHtml(r.text)}</p>
        </div>
        <div class="subsection">
          <h4>Auto-summary</h4>
          <p>${escapeHtml(review.summary)}</p>
        </div>
        <div class="subsection">
          <h4>Detected type and risk</h4>
          <p>${escapeHtml(review.clauseTypes.join(', '))} | ${escapeHtml(review.risk.riskLevel)}</p>
        </div>
      </div>
    `;
  }).join('');
}

async function parseUploadedFile(file) {
  const ext = file.name.toLowerCase().split('.').pop();

  if (ext === 'txt' || ext === 'md') {
    return await file.text();
  }

  if (ext === 'pdf') {
    const pdfjsLib = await import('https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.min.mjs');
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.min.mjs';
    const buffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
    let fullText = '';
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const pageText = content.items.map(item => item.str || '').join(' ');
      fullText += pageText + '\n';
    }
    return fullText;
  }

  if (ext === 'docx') {
    const buffer = await file.arrayBuffer();
    const result = await window.mammoth.extractRawText({ arrayBuffer: buffer });
    return result.value;
  }

  throw new Error('Unsupported file type');
}

async function handleUploads(files) {
  const corpus = {};
  for (const file of files) {
    try {
      el.status.textContent = `Reading ${file.name}...`;
      corpus[file.name] = await parseUploadedFile(file);
    } catch (err) {
      console.error(err);
      alert(`Could not parse ${file.name}`);
    }
  }
  if (Object.keys(corpus).length) {
    state.corpus = corpus;
    await rebuildIndex();
  }
}

function exportCurrentAnalysis() {
  const chunks = state.chunks.filter(c => c.contractName === state.selectedContract);
  const chosen = chunks[Number(el.chunkSelect.value || 0)];
  const review = reviewClause(chosen.text);
  const payload = {
    contract: state.selectedContract,
    clauseTitle: chosen.clauseTitle,
    clauseText: chosen.text,
    analysis: review,
    exportedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'legal_clause_analysis.json';
  a.click();
  URL.revokeObjectURL(url);
}

el.corpusMode.addEventListener('change', async (e) => {
  const mode = e.target.value;
  el.uploadWrap.classList.toggle('hidden', mode !== 'upload');
  if (mode === 'sample') {
    state.corpus = { ...SAMPLE_CONTRACTS };
    await rebuildIndex();
  }
});

el.contractSelect.addEventListener('change', (e) => {
  updateSelectedContract(e.target.value);
});

el.chunkSelect.addEventListener('change', (e) => {
  renderClause(state.selectedContract, Number(e.target.value));
});

el.searchBtn.addEventListener('click', async () => {
  const results = await hybridSearch(el.searchInput.value);
  renderResults(results);
});

el.reindexBtn.addEventListener('click', async () => {
  await rebuildIndex();
});

el.fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  if (files.length) await handleUploads(files);
});

el.exportBtn.addEventListener('click', exportCurrentAnalysis);

window.addEventListener('DOMContentLoaded', async () => {
  try {
    await rebuildIndex();
  } catch (err) {
    console.error(err);
    el.status.textContent = 'Could not initialize the embedding model. Check browser console or network access.';
    el.embedStat.textContent = 'Error';
  }
});
