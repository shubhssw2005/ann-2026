/* ── ANN Signal Dashboard ─────────────────────────────────────────────────── */

const REFRESH = 30_000;
let equityChart = null;
let modalChart = null;
let allSignals = [];
let activeFilter = 'ALL';

// ── Chart global defaults ─────────────────────────────────────────────────────

Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.font.size = 10;
Chart.defaults.color = '#aaa';

const CHART_COLORS = ['#0f0f0f', '#555', '#999', '#bbb', '#ddd'];

function baseChartOpts(extra = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 900,
            easing: 'easeInOutQuart',
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#fff',
                borderColor: '#e0e0e0',
                borderWidth: 1,
                titleColor: '#888',
                bodyColor: '#0f0f0f',
                padding: 10,
                cornerRadius: 6,
                callbacks: {
                    label: ctx => ` ${Number(ctx.parsed.y).toFixed(5)}`
                }
            },
            ...extra.plugins,
        },
        scales: {
            x: { display: false },
            y: {
                grid: { color: '#f4f4f4', drawBorder: false },
                border: { display: false },
                ticks: { color: '#bbb', maxTicksLimit: 4, font: { size: 9 } }
            },
            ...extra.scales,
        },
        ...extra,
    };
}

// ── Clock ─────────────────────────────────────────────────────────────────────

function startClock() {
    const el = document.getElementById('clock');
    const tick = () => {
        const now = new Date();
        el.textContent = now.toUTCString().slice(5, 25) + ' UTC';
    };
    tick();
    setInterval(tick, 1000);
}

// ── Status ────────────────────────────────────────────────────────────────────

async function checkStatus() {
    try {
        const r = await fetch('/api/status');
        const d = await r.json();
        const badge = document.getElementById('live-badge');
        const text = document.getElementById('live-text');
        if (d.signals_ready) {
            badge.classList.add('active');
            text.textContent = 'LIVE';
        } else {
            text.textContent = d.model_trained ? 'MODEL READY' : 'TRAINING';
        }
    } catch { /* offline */ }
}

// ── Signals ───────────────────────────────────────────────────────────────────

async function loadSignals() {
    try {
        const r = await fetch('/api/signals');
        if (!r.ok) return;
        allSignals = await r.json();
        renderSignals();
        updateSummarySignals();
    } catch { }
}

function renderSignals() {
    const data = activeFilter === 'ALL'
        ? allSignals
        : allSignals.filter(d => d.signal === activeFilter);

    const tbody = document.getElementById('signals-body');
    tbody.innerHTML = '';

    data.forEach((row, i) => {
        const tr = document.createElement('tr');
        tr.style.animationDelay = `${Math.min(i * 20, 400)}ms`;

        const p = row.probs;
        const confPct = Math.round(row.confidence * 100);
        const sigClass = row.signal === 'BUY' ? 'sig-buy'
            : row.signal === 'SELL' ? 'sig-sell' : 'sig-hold';

        tr.innerHTML = `
      <td style="color:var(--muted2);font-variant-numeric:tabular-nums">${i + 1}</td>
      <td style="font-weight:700;letter-spacing:0.02em;font-family:var(--sans)">${row.symbol.replace('USDT', '')}<span style="color:var(--muted2);font-weight:400">USDT</span></td>
      <td><span class="sig ${sigClass}">${row.signal}</span></td>
      <td>
        <div class="conf-bar">
          <div class="conf-track">
            <div class="conf-fill" style="width:0%" data-w="${confPct}"></div>
          </div>
          <span class="conf-num">${confPct}%</span>
        </div>
      </td>
      <td style="color:var(--muted)">${(p.SELL * 100).toFixed(1)}</td>
      <td style="color:var(--muted2)">${(p.HOLD * 100).toFixed(1)}</td>
      <td style="color:var(--text2)">${(p.BUY * 100).toFixed(1)}</td>
      <td style="font-variant-numeric:tabular-nums;color:var(--muted)">${fmtPrice(row.close)}</td>
    `;
        tr.addEventListener('click', () => openModal(row.symbol));
        tbody.appendChild(tr);
    });

    // Animate confidence bars after paint
    requestAnimationFrame(() => {
        document.querySelectorAll('.conf-fill[data-w]').forEach(el => {
            setTimeout(() => { el.style.width = el.dataset.w + '%'; }, 50);
        });
    });
}

// ── Backtest ──────────────────────────────────────────────────────────────────

async function loadBacktest() {
    try {
        const r = await fetch('/api/backtest');
        if (!r.ok) return;
        const data = await r.json();
        renderBtTable(data.coins);
        renderEquityChart(data.coins);
        updateSummaryBacktest(data.summary);
    } catch { }
}

function renderBtTable(coins) {
    const tbody = document.getElementById('bt-body');
    tbody.innerHTML = '';
    coins.slice(0, 40).forEach((c, i) => {
        const tr = document.createElement('tr');
        tr.style.animationDelay = `${Math.min(i * 18, 400)}ms`;
        const rc = c.total_return >= 0 ? 'pos' : 'neg';
        tr.innerHTML = `
      <td style="font-weight:700;font-family:var(--sans)">${c.symbol.replace('USDT', '')}<span style="color:var(--muted2);font-weight:400">USDT</span></td>
      <td class="${rc}">${c.total_return >= 0 ? '+' : ''}${c.total_return.toFixed(2)}%</td>
      <td style="color:var(--text2)">${c.sharpe.toFixed(2)}</td>
      <td style="color:var(--muted)">${c.win_rate.toFixed(1)}%</td>
      <td class="neg">${c.max_drawdown.toFixed(2)}%</td>
      <td style="color:var(--muted2)">${c.n_trades}</td>
    `;
        tr.addEventListener('click', () => openModal(c.symbol));
        tbody.appendChild(tr);
    });
}

function renderEquityChart(coins) {
    const ctx = document.getElementById('equity-chart').getContext('2d');
    if (equityChart) equityChart.destroy();

    const top = coins.slice(0, 5);
    const maxLen = Math.max(...top.map(c => c.equity_curve.length));

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: maxLen }, (_, i) => i),
            datasets: top.map((c, i) => ({
                label: c.symbol,
                data: c.equity_curve,
                borderColor: CHART_COLORS[i],
                borderWidth: i === 0 ? 2 : 1,
                pointRadius: 0,
                tension: 0.4,
                fill: false,
            }))
        },
        options: {
            ...baseChartOpts(),
            plugins: {
                ...baseChartOpts().plugins,
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: '#aaa',
                        boxWidth: 8,
                        boxHeight: 2,
                        padding: 12,
                        font: { size: 9 }
                    }
                }
            }
        }
    });
}

// ── Summary ───────────────────────────────────────────────────────────────────

function updateSummarySignals() {
    const buy = allSignals.filter(s => s.signal === 'BUY').length;
    const sell = allSignals.filter(s => s.signal === 'SELL').length;
    const hold = allSignals.filter(s => s.signal === 'HOLD').length;
    countUp('sv-coins', allSignals.length);
    countUp('sv-buy', buy);
    countUp('sv-sell', sell);
    countUp('sv-hold', hold);
}

function updateSummaryBacktest(s) {
    const sign = s.avg_return >= 0 ? '+' : '';
    document.getElementById('sv-return').textContent = sign + s.avg_return.toFixed(2) + '%';
    document.getElementById('sv-sharpe').textContent = s.avg_sharpe.toFixed(2);
    document.getElementById('sv-wr').textContent = s.avg_win_rate.toFixed(1) + '%';
    document.getElementById('sv-prof').textContent = s.profitable_coins + ' / ' + s.n_coins;
}

function countUp(id, target) {
    const el = document.getElementById(id);
    if (!el) return;
    let cur = 0;
    const dur = 600;
    const step = 16;
    const inc = target / (dur / step);
    const t = setInterval(() => {
        cur = Math.min(cur + inc, target);
        el.textContent = Math.round(cur);
        if (cur >= target) clearInterval(t);
    }, step);
}

// ── Modal ─────────────────────────────────────────────────────────────────────

async function openModal(symbol) {
    try {
        const r = await fetch(`/api/coin/${symbol}`);
        if (!r.ok) return;
        const c = await r.json();

        document.getElementById('modal-title').textContent = symbol;
        document.getElementById('modal').classList.remove('hidden');

        const rc = c.total_return >= 0 ? 'pos' : 'neg';
        document.getElementById('modal-stats').innerHTML = `
      <div class="modal-stat">
        <div class="val ${rc}">${c.total_return >= 0 ? '+' : ''}${c.total_return.toFixed(2)}%</div>
        <div class="lbl">Total Return</div>
      </div>
      <div class="modal-stat">
        <div class="val">${c.sharpe.toFixed(3)}</div>
        <div class="lbl">Sharpe Ratio</div>
      </div>
      <div class="modal-stat">
        <div class="val">${c.win_rate.toFixed(1)}%</div>
        <div class="lbl">Win Rate</div>
      </div>
      <div class="modal-stat">
        <div class="val neg">${c.max_drawdown.toFixed(2)}%</div>
        <div class="lbl">Max Drawdown</div>
      </div>
    `;

        // Equity chart
        const mctx = document.getElementById('modal-chart').getContext('2d');
        if (modalChart) modalChart.destroy();
        modalChart = new Chart(mctx, {
            type: 'line',
            data: {
                labels: c.timestamps,
                datasets: [{
                    data: c.equity_curve,
                    borderColor: '#0f0f0f',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: {
                        target: 'origin',
                        above: 'rgba(15,15,15,0.04)',
                        below: 'rgba(15,15,15,0)',
                    }
                }]
            },
            options: baseChartOpts()
        });

        // Trades
        const tbody = document.getElementById('trades-body');
        tbody.innerHTML = '';
        (c.trades || []).slice(-25).reverse().forEach(t => {
            const tr = document.createElement('tr');
            const pc = t.pnl >= 0 ? 'pos' : 'neg';
            tr.innerHTML = `
        <td style="font-weight:600;color:${t.side === 'long' ? 'var(--text)' : 'var(--muted)'}">${t.side.toUpperCase()}</td>
        <td style="color:var(--muted);font-variant-numeric:tabular-nums">${fmtPrice(t.entry)}</td>
        <td style="color:var(--muted);font-variant-numeric:tabular-nums">${fmtPrice(t.exit)}</td>
        <td class="${pc}" style="font-variant-numeric:tabular-nums">${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(3)}%</td>
        <td style="color:var(--muted2)">${t.hit_sl ? '✓' : '—'}</td>
        <td style="color:var(--muted2)">${t.hit_tp ? '✓' : '—'}</td>
      `;
            tbody.appendChild(tr);
        });
    } catch (e) { console.warn(e); }
}

document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal').addEventListener('click', e => {
    if (e.target.id === 'modal') closeModal();
});
function closeModal() {
    document.getElementById('modal').classList.add('hidden');
    if (modalChart) { modalChart.destroy(); modalChart = null; }
}

// ── Filters ───────────────────────────────────────────────────────────────────

document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.f;
        renderSignals();
    });
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtPrice(p) {
    if (p >= 1000) return p.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (p >= 1) return p.toFixed(4);
    return p.toFixed(6);
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
    startClock();
    await checkStatus();
    await Promise.all([loadSignals(), loadBacktest()]);
    setInterval(() => { loadSignals(); checkStatus(); }, REFRESH);
}

init();
