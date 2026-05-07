# AVA marketing + research site

Static, single-page. Deployed to GitHub Pages from `main` via `.github/workflows/pages.yml`.

## Files

| File | Purpose |
|---|---|
| `index.html` | full page — all interactivity inline (vanilla JS) |
| `styles.css` | all styles (light/dark, type pairings, density variants) |
| `.nojekyll` | tells GitHub Pages to skip Jekyll processing |
| `app.js` | dev-only mirror of inline script (not loaded by `index.html`) |
| `tweaks-app.jsx`, `tweaks-panel.jsx` | dev-only React preview tools (not loaded by `index.html`) |

## Local preview

```bash
cd site
python -m http.server 8000
# open http://localhost:8000
```

Or any static server (`npx serve .`, `caddy file-server`, etc.).

## Editing content

All copy lives in `index.html`. Benchmark numbers live in two arrays inside the inline `<script>`:

- `BM` — full 17-benchmark table data
- `ARC` / `GSM` — cross-model comparison bars

Update those arrays + the numbers in the section copy when re-evaluating. The phase data for the v3 roadmap is in `PHASES`.

## Deploy

Pushed to `main` → workflow at `.github/workflows/pages.yml` uploads `site/` and publishes. Live URL: `https://name0x0.github.io/AVA/`.
