// ══════════════════════════════════════════════════════════════════════════
//  PaperPal — Dashboard JavaScript
// ══════════════════════════════════════════════════════════════════════════

// ── Tab Switching ─────────────────────────────────────────────────────────

const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels  = document.querySelectorAll('.tab-panel');

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.tab;

        tabButtons.forEach(b => b.classList.remove('active'));
        tabPanels.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        document.getElementById(`panel-${target}`).classList.add('active');

        // Auto-refresh KB when switching to that tab
        if (target === 'kb') loadKnowledgeBase();
    });
});


// ── Toast Notifications ──────────────────────────────────────────────────

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast toast-${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3500);
}


// ── Loading State Helper ─────────────────────────────────────────────────

function setLoading(btn, loading) {
    if (loading) {
        btn.classList.add('loading');
        btn.disabled = true;
    } else {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}


// ── Skeleton Loader ──────────────────────────────────────────────────────

function showSkeletons(container, count = 3) {
    container.innerHTML = Array(count)
        .fill('<div class="skeleton skeleton-card"></div>')
        .join('');
}


// ══════════════════════════════════════════════════════════════════════════
//  SEARCH TAB
// ══════════════════════════════════════════════════════════════════════════

const searchInput   = document.getElementById('search-input');
const searchBtn     = document.getElementById('search-btn');
const searchResults = document.getElementById('search-results');

// Store last search results for ingest
let lastSearchResults = [];

searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    setLoading(searchBtn, true);
    showSkeletons(searchResults, 3);

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 }),
        });
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        lastSearchResults = data.results;
        renderSearchResults(data.results);
        showToast(`Found ${data.results.length} papers`, 'success');
    } catch (err) {
        searchResults.innerHTML = `<div class="kb-empty">Error: ${err.message}</div>`;
        showToast('Search failed', 'error');
    } finally {
        setLoading(searchBtn, false);
    }
}

function renderSearchResults(results) {
    if (!results.length) {
        searchResults.innerHTML = '<div class="kb-empty">No results found. Try a different query.</div>';
        return;
    }

    searchResults.innerHTML = results.map((paper, i) => `
        <div class="paper-card" id="card-${i}">
            <div class="paper-card-header">
                <div class="paper-title">
                    <a href="${paper.url}" target="_blank" rel="noopener">${escapeHtml(paper.title)}</a>
                </div>
                <div class="paper-score">Score: ${paper.score}</div>
            </div>
            <div class="paper-authors">${escapeHtml(paper.authors)}</div>
            <div class="paper-abstract">${escapeHtml(paper.abstract)}</div>
            <div class="paper-actions">
                <button class="btn btn-secondary btn-ingest" onclick="ingestSingle(${i})">
                    <span class="btn-text">Ingest Paper</span>
                    <span class="btn-loader"></span>
                </button>
            </div>
        </div>
    `).join('');
}


// ── Ingest a Single Paper ────────────────────────────────────────────────

async function ingestSingle(index) {
    const paper = lastSearchResults[index];
    if (!paper) return;

    const btn = document.querySelector(`#card-${index} .btn-ingest`);
    setLoading(btn, true);

    try {
        const res = await fetch('/api/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ papers: [paper] }),
        });
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        const summary = data.results[0];
        btn.outerHTML = `<span style="color: var(--success); font-size: 0.85rem; font-weight: 500;">Ingested (${summary.chunks_stored} chunks)</span>`;
        showToast(`Ingested: ${paper.title.substring(0, 50)}...`, 'success');
    } catch (err) {
        showToast(`Ingest failed: ${err.message}`, 'error');
        setLoading(btn, false);
    }
}


// ══════════════════════════════════════════════════════════════════════════
//  KNOWLEDGE BASE TAB
// ══════════════════════════════════════════════════════════════════════════

const kbStats          = document.getElementById('kb-stats');
const kbTableContainer = document.getElementById('kb-table-container');

async function loadKnowledgeBase() {
    try {
        const res = await fetch('/api/kb');
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        renderKBStats(data.papers.length, data.total_chunks);
        renderKBTable(data.papers);
    } catch (err) {
        kbStats.innerHTML = '';
        kbTableContainer.innerHTML = `<div class="kb-empty">Error loading knowledge base: ${err.message}</div>`;
    }
}

function renderKBStats(paperCount, chunkCount) {
    kbStats.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${paperCount}</div>
            <div class="stat-label">Papers Ingested</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${chunkCount}</div>
            <div class="stat-label">Total Chunks</div>
        </div>
    `;
}

function renderKBTable(papers) {
    if (!papers.length) {
        kbTableContainer.innerHTML = `
            <div class="kb-empty">
                <div class="kb-empty-icon">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                    </svg>
                </div>
                <p>No papers ingested yet.</p>
                <p style="margin-top: 0.5rem; font-size: 0.8rem;">Use the Search tab to find and ingest papers.</p>
            </div>
        `;
        return;
    }

    kbTableContainer.innerHTML = `
        <table class="kb-table">
            <thead>
                <tr>
                    <th>Paper ID</th>
                    <th>Title</th>
                    <th>Chunks</th>
                </tr>
            </thead>
            <tbody>
                ${papers.map(p => `
                    <tr>
                        <td><a href="${p.url}" target="_blank" rel="noopener" style="color: var(--accent); text-decoration: none;">${escapeHtml(p.paper_id)}</a></td>
                        <td class="td-title">${escapeHtml(p.title)}</td>
                        <td class="td-chunks">${p.chunks}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}


// ══════════════════════════════════════════════════════════════════════════
//  ASK TAB
// ══════════════════════════════════════════════════════════════════════════

const askInput  = document.getElementById('ask-input');
const askBtn    = document.getElementById('ask-btn');
const askResult = document.getElementById('ask-result');

askBtn.addEventListener('click', performAsk);
askInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') performAsk();
});

async function performAsk() {
    const question = askInput.value.trim();
    if (!question) return;

    setLoading(askBtn, true);
    askResult.innerHTML = `
        <div class="skeleton skeleton-card" style="height: 200px;"></div>
        <div class="skeleton skeleton-card" style="height: 80px;"></div>
    `;

    try {
        const res = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await res.json();

        if (data.error) throw new Error(data.error);

        renderAnswer(data);
        showToast('Answer generated', 'success');
    } catch (err) {
        askResult.innerHTML = `<div class="kb-empty">Error: ${err.message}</div>`;
        showToast('Failed to generate answer', 'error');
    } finally {
        setLoading(askBtn, false);
    }
}

function renderAnswer(data) {
    let html = `
        <div class="answer-box">
            <div class="answer-label">Answer</div>
            <div class="answer-text">${escapeHtml(data.answer)}</div>
        </div>
    `;

    if (data.citations && data.citations.length > 0) {
        html += `
            <div class="citations-box">
                <div class="citations-label">Sources</div>
                ${data.citations.map((c, i) => `
                    <div class="citation-card">
                        <div class="citation-number">${i + 1}</div>
                        <div class="citation-info">
                            <div class="citation-title">${escapeHtml(c.title || 'Unknown')}</div>
                            <div class="citation-authors">${escapeHtml(c.authors || '')}</div>
                        </div>
                        ${c.url ? `<a href="${c.url}" target="_blank" rel="noopener" class="citation-link">PDF</a>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    askResult.innerHTML = html;
}


// ── Utilities ─────────────────────────────────────────────────────────────

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// ── Init ──────────────────────────────────────────────────────────────────

// Load KB data on page load (in case user goes to KB tab first)
loadKnowledgeBase();
