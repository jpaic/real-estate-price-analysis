# Real Estate Price Analysis Frontend

Next.js dashboard version of the original Streamlit real estate price analysis app.

## Local Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Data

The dashboard reads precomputed data from:

```text
public/dashboard-data.json
```

To regenerate it from the repository root:

```bash
python scripts/export_dashboard_data.py
```

The JSON file is committed so Vercel can deploy the frontend without needing a Python runtime.

## Vercel Deployment

Use these Vercel settings:

- Framework preset: Next.js
- Root directory: `app/frontend`
- Install command: `npm install`
- Build command: `npm run build`
- Output directory: Next.js default
