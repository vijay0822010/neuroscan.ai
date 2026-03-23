# 🧠 NeuroScan AI

Neurological risk assessment using handwriting + speech analysis.
**One-click deploy to Railway — full stack, single URL.**

---

## Deploy to Railway (5 minutes)

### Step 1 — Push to GitHub

```bash
cd neuroscan
git init
git add .
git commit -m "Initial commit"
# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/neuroscan-ai.git
git push -u origin main
```

### Step 2 — Create Railway project

1. Go to **[railway.app](https://railway.app)** → sign in
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your `neuroscan-ai` repository
4. Railway detects the `Dockerfile` and starts building automatically

### Step 3 — Add environment variables

In Railway dashboard → your service → **Variables** tab, add:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | `your_groq_api_key_here` |
| `GROQ_API_URL` | `https://api.groq.com/openai/v1/chat/completions` |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` |

> Railway sets `PORT` automatically — do NOT add it manually.

### Step 4 — Get your URL

1. In Railway dashboard → your service → **Settings** → **Networking**
2. Click **Generate Domain** (free `*.railway.app` domain)
3. Your app is live at e.g. `https://neuroscan-ai-production.up.railway.app`

---

## How the deployment works

Railway uses the **Dockerfile**:
1. **Stage 1** (Node 20): runs `npm install && npm run build` → creates `frontend/dist/`
2. **Stage 2** (Python 3.11): installs Python deps, copies built frontend
3. Starts `python run.py` → FastAPI serves both API and React from one port

---

## Run locally

```bash
# Install Python deps
pip install -r requirements.txt

# Build and run (single command)
bash build.sh

# OR run separately:
cd frontend && npm install && npm run build && cd ..
python run.py
# → open http://localhost:8000
```

---

## Project structure

```
neuroscan/
├── app/
│   ├── main.py                  ← FastAPI app + serves React SPA
│   ├── core/config.py           ← Settings (env vars / .env)
│   ├── routers/
│   │   ├── analysis.py          ← POST /api/v1/analyse
│   │   └── report.py            ← GET  /api/v1/report/{id}
│   └── services/
│       ├── resnet_service.py    ← ResNet-512 pipeline
│       ├── wav2vec2_service.py  ← Wav2Vec2-768 pipeline
│       ├── fusion_service.py    ← NSS computation
│       └── report_service.py   ← PDF generation
├── frontend/
│   ├── src/App.jsx              ← React UI
│   ├── package.json
│   └── vite.config.js
├── Dockerfile                   ← Railway build (multi-stage)
├── railway.toml                 ← Railway config
├── nixpacks.toml                ← Nixpacks fallback config
├── requirements.txt             ← Python deps
├── run.py                       ← Server entry point
└── .env                        ← Local dev only (NOT for Railway)
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check (Railway probe) |
| POST | `/api/v1/analyse` | Run full assessment |
| GET | `/api/v1/report/{id}` | Download PDF report |
| GET | `/docs` | Swagger UI |

---

## How analysis works

```
Image  → ResNet-512 (16 residual blocks)
         → 512-dim embedding + 8 stroke biomarkers

Audio  → Wav2Vec2-768 (7 CNN + 12 transformer layers)
         → 768-dim embedding + 8 acoustic biomarkers

Fusion → Calibrated biomarker pathology score
         → NSS = Sigmoid(score × 0.6)
         → ≥0.75 = LOW | 0.50-0.75 = MODERATE | <0.50 = HIGH

Analysis → Groq API (primary) → model fallback (if API fails)
```

---

## Medical disclaimer

For screening purposes only. Not a medical diagnosis.
Consult a qualified neurologist for clinical decisions.
