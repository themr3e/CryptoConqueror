"""Live application logs endpoint — view recent logs without Railway dashboard."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.utils.log_buffer import log_buffer

router = APIRouter(tags=["logs"])


@router.get("/logs/data")
async def logs_data(
    limit: int = Query(200, ge=1, le=2000),
    level: str = Query(None, description="Minimum level: DEBUG, INFO, WARNING, ERROR"),
    search: str = Query(None, description="Search in message text"),
):
    """Return recent log entries as JSON (newest first)."""
    entries = log_buffer.get_entries(limit=limit, level=level, search=search)
    return {"count": len(entries), "entries": entries}


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """HTML page for viewing live application logs."""
    return LOGS_HTML


LOGS_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Logs — QuantLive</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #c9d1d9;
      font-family: "SF Mono", "Fira Code", "Cascadia Code", Menlo, monospace;
      font-size: 13px;
      min-height: 100vh;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 24px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    header h1 { font-size: 18px; color: #58a6ff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    nav a {
      color: #8b949e;
      text-decoration: none;
      margin-left: 20px;
      font-size: 13px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    nav a:hover { color: #c9d1d9; }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 24px;
      background: #161b22;
      border-bottom: 1px solid #21262d;
      flex-wrap: wrap;
    }
    .toolbar label {
      font-size: 11px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-family: -apple-system, sans-serif;
    }
    .toolbar select, .toolbar input[type="text"] {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #c9d1d9;
      font-size: 13px;
      padding: 6px 10px;
      outline: none;
    }
    .toolbar select:focus, .toolbar input:focus { border-color: #58a6ff; }
    .toolbar input[type="text"] { width: 220px; }

    .btn {
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-family: -apple-system, sans-serif;
    }
    .btn:hover { background: #30363d; }
    .btn-live {
      background: rgba(63,185,80,0.15);
      border-color: rgba(63,185,80,0.4);
      color: #3fb950;
    }
    .btn-live.active {
      background: rgba(63,185,80,0.3);
      border-color: #3fb950;
    }

    .count-badge {
      font-size: 12px;
      color: #8b949e;
      font-family: -apple-system, sans-serif;
      margin-left: auto;
    }

    #log-container {
      padding: 8px 0;
      overflow-y: auto;
      height: calc(100vh - 110px);
    }
    .log-entry {
      padding: 3px 24px;
      line-height: 1.6;
      border-left: 3px solid transparent;
      transition: background 0.1s;
    }
    .log-entry:hover { background: #161b22; }
    .log-entry.new { animation: flash 1s ease-out; }

    @keyframes flash {
      0% { background: rgba(88,166,255,0.12); }
      100% { background: transparent; }
    }

    .ts { color: #484f58; }
    .lvl { font-weight: 700; min-width: 70px; display: inline-block; }
    .mod { color: #8b949e; }
    .msg { color: #c9d1d9; }
    .exc { color: #f85149; font-size: 12px; margin-left: 80px; display: block; white-space: pre-wrap; }

    .lvl-DEBUG    { color: #8b949e; }
    .lvl-INFO     { color: #58a6ff; }
    .lvl-SUCCESS  { color: #3fb950; }
    .lvl-WARNING  { color: #e3b341; }
    .lvl-ERROR    { color: #f85149; }
    .lvl-CRITICAL { color: #f85149; font-weight: 900; }

    .log-entry.level-ERROR,
    .log-entry.level-CRITICAL { border-left-color: #f85149; }
    .log-entry.level-WARNING { border-left-color: #e3b341; }

    .empty {
      text-align: center;
      padding: 60px 24px;
      color: #484f58;
      font-family: -apple-system, sans-serif;
      font-size: 15px;
    }
  </style>
</head>
<body>

<header>
  <h1>QuantLive Logs</h1>
  <nav>
    <a href="/dashboard">Dashboard</a>
    <a href="/chart/">Chart</a>
    <a href="/status">Status</a>
    <a href="/health">Health</a>
  </nav>
</header>

<div class="toolbar">
  <label>Level</label>
  <select id="level-filter">
    <option value="">All</option>
    <option value="DEBUG">DEBUG</option>
    <option value="INFO" selected>INFO</option>
    <option value="WARNING">WARNING</option>
    <option value="ERROR">ERROR</option>
  </select>

  <label>Search</label>
  <input type="text" id="search-input" placeholder="Filter messages..." />

  <button class="btn btn-live" id="live-btn" onclick="toggleLive()">Live</button>
  <button class="btn" onclick="loadLogs()">Refresh</button>

  <span class="count-badge" id="count-badge">—</span>
</div>

<div id="log-container"></div>

<script>
  let liveMode = true;
  let liveTimer = null;
  let knownCount = 0;

  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function fmtTime(iso) {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    const ms = String(d.getMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss}.${ms}`;
  }

  function renderEntry(e, isNew) {
    const cls = `log-entry level-${e.level}${isNew ? ' new' : ''}`;
    let html = `<div class="${cls}">`;
    html += `<span class="ts">${fmtTime(e.timestamp)}</span> `;
    html += `<span class="lvl lvl-${e.level}">${e.level.padEnd(8)}</span> `;
    html += `<span class="mod">${escHtml(e.module)}:</span> `;
    html += `<span class="msg">${escHtml(e.message)}</span>`;
    if (e.exception) {
      html += `<span class="exc">${escHtml(e.exception)}</span>`;
    }
    html += '</div>';
    return html;
  }

  async function loadLogs() {
    const level = document.getElementById('level-filter').value;
    const search = document.getElementById('search-input').value.trim();
    const params = new URLSearchParams({ limit: '500' });
    if (level) params.set('level', level);
    if (search) params.set('search', search);

    try {
      const resp = await fetch('/logs/data?' + params);
      const data = await resp.json();
      const container = document.getElementById('log-container');
      const badge = document.getElementById('count-badge');

      if (!data.entries || data.entries.length === 0) {
        container.innerHTML = '<div class="empty">No log entries matching filters</div>';
        badge.textContent = '0 entries';
        knownCount = 0;
        return;
      }

      const isNew = data.count > knownCount && knownCount > 0;
      container.innerHTML = data.entries.map((e, i) =>
        renderEntry(e, isNew && i < (data.count - knownCount))
      ).join('');
      badge.textContent = data.count + ' entries';
      knownCount = data.count;

    } catch (err) {
      console.error('Log fetch error', err);
    }
  }

  function toggleLive() {
    liveMode = !liveMode;
    const btn = document.getElementById('live-btn');
    btn.classList.toggle('active', liveMode);
    btn.textContent = liveMode ? 'Live' : 'Paused';
    if (liveMode) startLive(); else stopLive();
  }

  function startLive() {
    stopLive();
    liveTimer = setInterval(loadLogs, 3000);
  }

  function stopLive() {
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  }

  // Reload on filter change
  document.getElementById('level-filter').addEventListener('change', () => { knownCount = 0; loadLogs(); });
  let searchTimeout;
  document.getElementById('search-input').addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => { knownCount = 0; loadLogs(); }, 400);
  });

  // Initial load + start live
  loadLogs();
  toggleLive();
</script>
</body>
</html>
"""
