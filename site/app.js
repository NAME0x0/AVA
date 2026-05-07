// AVA — interactivity (vanilla)

(function () {
  'use strict';

  // -------------------------------------------------------------
  // Benchmark data — full 17, with categories
  // -------------------------------------------------------------
  const BM = [
    { name: 'ARC-Easy',                   cat: 'reasoning',  n: 2376,  acc: 92.0, ci: '[90.8, 93.0]' },
    { name: 'ARC-Challenge',              cat: 'reasoning',  n: 1172,  acc: 82.0, ci: '[79.7, 84.1]', star: true },
    { name: 'PIQA',                       cat: 'reasoning',  n: 1838,  acc: 75.9, ci: '[73.9, 77.8]' },
    { name: 'BoolQ',                      cat: 'knowledge',  n: 3270,  acc: 75.0, ci: '[73.5, 76.5]' },
    { name: 'MMLU (5-shot)',              cat: 'knowledge',  n: 14042, acc: 59.2, ci: '[58.4, 60.1]', star: true },
    { name: 'HellaSwag',                  cat: 'reasoning',  n: 10042, acc: 56.8, ci: '[55.8, 57.8]' },
    { name: 'WinoGrande XL',              cat: 'reasoning',  n: 1267,  acc: 56.4, ci: '[53.7, 59.1]' },
    { name: 'TruthfulQA-MC1',             cat: 'knowledge',  n: 817,   acc: 47.5, ci: '[44.1, 50.9]' },
    { name: 'GSM8K self-cons (k=5)',      cat: 'math',       n: 200,   acc: 44.0, ci: '[37.3, 50.9]' },
    { name: 'MBPP+',                      cat: 'code',       n: 378,   acc: 35.7, ci: '[31.0, 40.7]' },
    { name: 'Agentic GSM8K',              cat: 'math',       n: 1319,  acc: 35.4, ci: '[32.9, 38.0]' },
    { name: 'GSM8K (greedy)',             cat: 'math',       n: 1319,  acc: 35.3, ci: '[32.8, 38.0]' },
    { name: 'MGSM (en/es/fr)',            cat: 'other',      n: 750,   acc: 34.4, ci: '[31.1, 37.9]' },
    { name: 'IFEval (strict)',            cat: 'other',      n: 541,   acc: 31.6, ci: '[27.8, 35.6]' },
    { name: 'MMLU-Pro',                   cat: 'knowledge',  n: 12032, acc: 30.9, ci: '[30.1, 31.8]' },
    { name: 'HumanEval+',                 cat: 'code',       n: 164,   acc: 19.5, ci: '[14.2, 26.3]' },
    { name: 'MATH-500',                   cat: 'math',       n: 500,   acc: 18.8, ci: '[15.6, 22.5]' },
  ];

  // ARC-Challenge across small models (from README)
  const ARC = [
    { m: 'TinyLlama 1.1B',         v: 30.1 },
    { m: 'SmolLM2 1.7B',           v: 52.0 },
    { m: 'Qwen2.5 1.5B',           v: 54.7 },
    { m: 'Mistral 7B-IT',          v: 55.5 },
    { m: 'Gemma2 2B',              v: 55.7 },
    { m: 'Llama 3.2 1B-IT',        v: 59.4 },
    { m: 'Llama 3.2 3B',           v: 69.1 },
    { m: 'Llama 3.2 3B-IT',        v: 78.6 },
    { m: 'AVA v2 2B',              v: 82.0, ava: true },
    { m: 'Phi-4-mini 3.8B',        v: 83.7 },
    { m: 'Phi-3.5-mini 3.8B',      v: 84.6 },
  ];

  const GSM = [
    { m: 'TinyLlama 1.1B',         v: 2.0 },
    { m: 'Gemma2 2B',              v: 24.3 },
    { m: 'Qwen3.5 2B Base',        v: 28.0 },
    { m: 'AVA v2 (greedy)',        v: 35.3, ava: true },
    { m: 'AVA v2 (self-cons k=5)', v: 44.0, ava: true },
    { m: 'Llama 3.2 1B-IT',        v: 44.4 },
    { m: 'SmolLM2 1.7B',           v: 48.2 },
    { m: 'Mistral 7B-IT',          v: 52.2 },
    { m: 'Qwen2.5 1.5B',           v: 68.5 },
    { m: 'Llama 3.2 3B-IT',        v: 77.7 },
    { m: 'Qwen2.5 3B',             v: 79.1 },
    { m: 'Phi-3.5 3.8B',           v: 86.2 },
    { m: 'Phi-4 3.8B',             v: 88.6 },
  ];

  // -------------------------------------------------------------
  // Comparison bar chart
  // -------------------------------------------------------------
  function buildBars(host, rows, max) {
    const m = max || 100;
    host.innerHTML = '';
    rows.forEach((r) => {
      const row = document.createElement('div');
      row.className = 'bar-row' + (r.ava ? ' is-ava' : '');
      row.innerHTML = `
        <div class="lbl">${r.m}</div>
        <div class="bar" style="--w:${(r.v / m) * 100}%"></div>
        <div class="val">${r.v.toFixed(1)}</div>
      `;
      host.appendChild(row);
    });
  }

  function initBars() {
    const arcHost = document.querySelector('[data-bars="arc"]');
    const gsmHost = document.querySelector('[data-bars="gsm"]');
    if (arcHost) buildBars(arcHost, ARC);
    if (gsmHost) buildBars(gsmHost, GSM);

    const block = document.getElementById('cmp-block');
    if (!block) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          block.classList.add('revealed', 'in');
          io.disconnect();
        }
      });
    }, { threshold: 0.18 });
    io.observe(block);
  }

  // -------------------------------------------------------------
  // Benchmark table — sortable + filterable
  // -------------------------------------------------------------
  let bmState = {
    sort: { key: 'acc', dir: 'desc' },
    filter: 'all',
  };
  const MAX_BAR = 100;

  function renderBM() {
    const tbody = document.querySelector('#bm-table tbody');
    if (!tbody) return;
    let rows = BM.slice();
    if (bmState.filter !== 'all') {
      rows = rows.filter((r) => r.cat === bmState.filter);
    }
    rows.sort((a, b) => {
      const k = bmState.sort.key;
      let x = a[k], y = b[k];
      if (k === 'ci') { x = parseFloat(a.ci.replace('[','').split(',')[0]); y = parseFloat(b.ci.replace('[','').split(',')[0]); }
      if (typeof x === 'string') {
        const r = x.localeCompare(y);
        return bmState.sort.dir === 'asc' ? r : -r;
      }
      return bmState.sort.dir === 'asc' ? x - y : y - x;
    });

    tbody.innerHTML = rows.map((r, i) => `
      <tr data-cat="${r.cat}">
        <td class="col-name col-acc">${r.star ? '<span class="star">★</span>' : ''}${r.name}</td>
        <td class="col-cat is-mobile-hidden">${r.cat}</td>
        <td>${r.n.toLocaleString()}</td>
        <td class="col-acc">${r.acc.toFixed(1)}%</td>
        <td class="col-ci is-mobile-hidden">${r.ci}</td>
        <td class="col-bar is-mobile-hidden"><div class="mini" style="--w:${(r.acc / MAX_BAR) * 100}%; --d:${i * 25}ms;"></div></td>
      </tr>
    `).join('');

    // sort indicators
    document.querySelectorAll('#bm-table thead th').forEach((th) => {
      const key = th.getAttribute('data-sort');
      const arr = th.querySelector('.arr');
      if (!key) return;
      if (key === bmState.sort.key) {
        th.setAttribute('aria-sort', bmState.sort.dir === 'asc' ? 'ascending' : 'descending');
        if (arr) arr.textContent = bmState.sort.dir === 'asc' ? '▲' : '▼';
      } else {
        th.removeAttribute('aria-sort');
        if (arr) arr.textContent = '';
      }
    });
  }

  function initBM() {
    document.querySelectorAll('#bm-table thead th[data-sort]').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort');
        if (bmState.sort.key === key) {
          bmState.sort.dir = bmState.sort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          bmState.sort.key = key;
          bmState.sort.dir = (key === 'name' || key === 'cat') ? 'asc' : 'desc';
        }
        renderBM();
      });
    });
    document.querySelectorAll('[data-filter-group] button').forEach((b) => {
      b.addEventListener('click', () => {
        document.querySelectorAll('[data-filter-group] button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
        b.setAttribute('aria-pressed', 'true');
        bmState.filter = b.getAttribute('data-filter');
        renderBM();
      });
    });
    renderBM();
  }

  // -------------------------------------------------------------
  // Try-it tabs + copy buttons
  // -------------------------------------------------------------
  function initTabs() {
    document.querySelectorAll('.try-tabs button[data-tab]').forEach((b) => {
      b.addEventListener('click', () => {
        const t = b.getAttribute('data-tab');
        document.querySelectorAll('.try-tabs button[data-tab]').forEach((x) => x.setAttribute('aria-pressed', 'false'));
        b.setAttribute('aria-pressed', 'true');
        document.querySelectorAll('.try-pane').forEach((p) => {
          if (p.getAttribute('data-tab') === t) p.setAttribute('data-active', '');
          else p.removeAttribute('data-active');
        });
      });
    });
  }

  function initCopy() {
    document.querySelectorAll('.copy-btn').forEach((b) => {
      b.addEventListener('click', async () => {
        const id = b.getAttribute('data-copy');
        const el = document.getElementById(id);
        if (!el) return;
        const text = el.innerText;
        try {
          await navigator.clipboard.writeText(text);
        } catch (e) {
          const ta = document.createElement('textarea');
          ta.value = text; document.body.appendChild(ta); ta.select();
          try { document.execCommand('copy'); } catch (_) {}
          document.body.removeChild(ta);
        }
        const prev = b.textContent;
        b.classList.add('copied');
        b.textContent = '✓ Copied';
        setTimeout(() => { b.classList.remove('copied'); b.textContent = prev; }, 1400);
      });
    });
  }

  // -------------------------------------------------------------
  // Code modal — opens Try section
  // -------------------------------------------------------------
  function initModal() {
    const modal = document.getElementById('modal');
    const body = document.getElementById('modal-body');
    if (!modal || !body) return;

    function openModal() {
      // clone the try-block markup into modal
      const src = document.getElementById('try-block');
      if (!src) return;
      body.innerHTML = '';
      const clone = src.cloneNode(true);
      clone.id = 'try-block-modal';
      body.appendChild(clone);
      // re-wire tabs and copy in clone
      clone.querySelectorAll('.try-tabs button[data-tab]').forEach((b) => {
        b.addEventListener('click', () => {
          const t = b.getAttribute('data-tab');
          clone.querySelectorAll('.try-tabs button[data-tab]').forEach((x) => x.setAttribute('aria-pressed','false'));
          b.setAttribute('aria-pressed','true');
          clone.querySelectorAll('.try-pane').forEach((p) => {
            if (p.getAttribute('data-tab') === t) p.setAttribute('data-active','');
            else p.removeAttribute('data-active');
          });
        });
      });
      clone.querySelectorAll('.copy-btn').forEach((b) => {
        b.addEventListener('click', async () => {
          const id = b.getAttribute('data-copy');
          // grab text from the visible source (may exist in multiple places)
          const el = clone.querySelector('#' + id) || document.getElementById(id);
          if (!el) return;
          try { await navigator.clipboard.writeText(el.innerText); } catch (_) {}
          const prev = b.textContent;
          b.classList.add('copied');
          b.textContent = '✓ Copied';
          setTimeout(()=>{ b.classList.remove('copied'); b.textContent = prev; }, 1400);
        });
      });
      modal.setAttribute('data-open','');
      document.body.style.overflow = 'hidden';
    }
    function closeModal() {
      modal.removeAttribute('data-open');
      document.body.style.overflow = '';
      body.innerHTML = '';
    }
    document.querySelectorAll('[data-action="open-try"]').forEach((b) => b.addEventListener('click', openModal));
    document.querySelectorAll('[data-close-modal]').forEach((b) => b.addEventListener('click', closeModal));
    modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.hasAttribute('data-open')) closeModal(); });
  }

  // -------------------------------------------------------------
  // Roadmap chevrons
  // -------------------------------------------------------------
  const PHASES = [
    { id: 'P0',  s: 'done',   label: 'Scaffolding',          eta: 'Q4 2025',
      desc: 'Repository layout, design doc, configs, engine stubs, MCP catalog, and bring-up scripts. Done — the foundations are checked in.' },
    { id: 'P1',  s: 'active', label: 'Teacher download',     eta: 'Q1 2026',
      desc: 'Pull Qwen 3.6 35B-A3B, set up streamed loading on the laptop, validate teacher logits across the distillation prompt set.' },
    { id: 'P2',  s: 'queued', label: 'Ternary linear',       eta: 'Q1 2026',
      desc: 'Implement the ternary linear primitive for FFN experts (BitDistiller-style {-1, 0, +1} with learnable per-channel scale).' },
    { id: 'P3',  s: 'queued', label: 'MoTE-FFN',             eta: 'Q1 2026',
      desc: 'Mixture of Ternary Experts: ternary FFN experts, BF16 router, one BF16 shared expert. Smoke-test on a 6-8B base.' },
    { id: 'P4',  s: 'queued', label: 'Gated DeltaNet',       eta: 'Q2 2026',
      desc: 'Wire the 3:1 hybrid: three Gated DeltaNet linear-time layers per one full-attention layer. Decode-time linear scaling target.' },
    { id: 'P5',  s: 'queued', label: 'BF16 warmup',          eta: 'Q2 2026',
      desc: 'BF16 warmup of the student so the architecture trains stably before introducing ternary QAT.' },
    { id: 'P6',  s: 'queued', label: 'Ternary QAT distill',  eta: 'Q2 2026',
      desc: 'BitDistiller 3-stage QAT: cold ternary, asymmetric quantization, KD against the Qwen 3.6 teacher. Track loss and ARC/GSM8K every 200 steps.' },
    { id: 'P7',  s: 'queued', label: 'SFT + DPO',            eta: 'Q3 2026',
      desc: 'Supervised fine-tune on the v2 corpus extended for tools and long-form, then DPO alignment for instruction-following quality.' },
    { id: 'P8',  s: 'queued', label: 'MCP server',           eta: 'Q3 2026',
      desc: 'FastMCP 3.0 tool server + XGrammar constrained decoding. Calculator, code, retrieval, file. Make tool-use non-latent.' },
    { id: 'P9',  s: 'queued', label: 'GGUF export',          eta: 'Q3 2026',
      desc: 'Convert merged student to GGUF, verify Q4_K_M and Q8_0 retain QAT gains within tolerance.' },
    { id: 'P10', s: 'queued', label: 'Full eval',            eta: 'Q4 2026',
      desc: 'Run the same 17-benchmark / 16,872-task suite on v3 and publish a side-by-side report with v2.' },
    { id: 'P11', s: 'queued', label: 'Release',              eta: 'Q4 2026',
      desc: 'Model card, GGUF release, documentation, and a research write-up of what changed and why.' },
  ];

  function initRoadmap() {
    const row = document.getElementById('chev-row');
    const detail = document.getElementById('chev-detail');
    if (!row || !detail) return;
    row.innerHTML = PHASES.map((p, i) => `
      <div class="chev ${p.s === 'done' ? 'is-done' : ''} ${p.s === 'active' && i === 1 ? 'is-active' : ''}" data-i="${i}">
        <span class="id">${p.id}</span>
        <span>${p.label}</span>
      </div>
    `).join('');

    function show(i) {
      const p = PHASES[i];
      detail.innerHTML = `
        <div class="meta">
          <b>${p.id} · ${p.s.toUpperCase()}</b>
          target&nbsp;${p.eta}
        </div>
        <div>
          <h4>${p.label}</h4>
          <p>${p.desc}</p>
        </div>
      `;
      row.querySelectorAll('.chev').forEach((c) => c.classList.remove('is-active'));
      row.querySelector('.chev[data-i="' + i + '"]').classList.add('is-active');
    }

    row.querySelectorAll('.chev').forEach((c) => {
      c.addEventListener('click', () => show(parseInt(c.getAttribute('data-i'), 10)));
    });
    show(1);
  }

  // -------------------------------------------------------------
  // Hero typewriter (only when hero=typing)
  // -------------------------------------------------------------
  let typingTimer = null;
  function applyHero() {
    const h1 = document.getElementById('hero-h1');
    const tw = document.getElementById('hero-typewriter');
    const frame = document.getElementById('hero-frame');
    const mode = document.documentElement.getAttribute('data-hero') || 'ascii';
    if (!h1 || !tw) return;

    const full = h1.getAttribute('data-text');
    if (typingTimer) { clearInterval(typingTimer); typingTimer = null; }

    if (mode === 'typing') {
      tw.textContent = '';
      let i = 0;
      typingTimer = setInterval(() => {
        if (i >= full.length) { clearInterval(typingTimer); typingTimer = null; return; }
        tw.textContent = full.slice(0, ++i);
      }, 38);
    } else {
      tw.textContent = full;
    }

    if (frame) frame.style.display = (mode === 'static') ? 'none' : '';
  }

  // -------------------------------------------------------------
  // Scroll reveal
  // -------------------------------------------------------------
  function initReveal() {
    const els = document.querySelectorAll('.fade-in');
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 });
    els.forEach((el) => io.observe(el));
  }

  // -------------------------------------------------------------
  // Public hook for tweaks panel
  // -------------------------------------------------------------
  window.AVA = {
    setMode(m)    { document.documentElement.setAttribute('data-mode', m); },
    setType(t)    { document.documentElement.setAttribute('data-type', t); },
    setDensity(d) { document.documentElement.setAttribute('data-density', d); },
    setHero(h)    { document.documentElement.setAttribute('data-hero', h); applyHero(); },
  };

  // -------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    initBars();
    initBM();
    initTabs();
    initCopy();
    initModal();
    initRoadmap();
    applyHero();
    initReveal();
  });
})();
