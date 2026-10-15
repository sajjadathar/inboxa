# Inboxa

An AI inbox assistant for small businesses. It reads customer emails, sorts them, answers the simple ones, drafts replies for the hard ones, and logs everything to a CRM.

**Stack:** n8n · FastAPI · Postgres + pgvector · Next.js · Docker Compose

## Folders

| Folder | What's inside |
|---|---|
| `backend/` | FastAPI AI service (classify, draft replies, lead scoring) |
| `frontend/` | Next.js dashboard |
| `n8n/workflows/` | Exported n8n workflows (JSON) |
| `infra/` | Database setup and Dockerfiles |

---

## Phase 1 setup, step by step

### Step 1. Install the tools
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **Node.js 22**: https://nodejs.org (needed once to create the Next.js app)
- **Git**

Check that they work:
```bash
docker --version
docker compose version
node --version
```

### Step 2. Create your `.env` file
```bash
cp .env.example .env
```
Open `.env` and change `POSTGRES_PASSWORD`. Leave everything else as it is for now.

### Step 3. Create the Next.js app (one time only)
From the project root:
```bash
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*" --use-npm
```
If it asks about Turbopack or other options, accept the defaults.

### Step 4. Start everything
```bash
docker compose up -d --build
```
The first run downloads images, so it takes a few minutes.

### Step 5. Check each part

| Open this | You should see |
|---|---|
| http://localhost:8000/health | `"database": "ok"` and `"pgvector": "ok"` |
| http://localhost:8000/docs | FastAPI's interactive API page |
| http://localhost:3000 | The Next.js starter page |
| http://localhost:5678 | n8n. Create your owner account here. |

If all four work, Phase 1 is done.

### Step 6 (optional). Free local AI with Ollama
Only needed if you want to use Ollama instead of a free API tier.
```bash
docker compose --profile local-ai up -d
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text
```

### Step 7. Save to GitHub
```bash
git init
git add .
git commit -m "Phase 1: project setup"
```
Then create an empty repo on GitHub and push to it.

---

## Accounts to create (needed from Phase 3)

All free:
1. **Test Gmail account.** Use a new one, never your real inbox.
2. **Google Cloud project.** Enable the Gmail API and create OAuth credentials (n8n's Gmail node explains how).
3. **AI key.** Pick one: Gemini (Google AI Studio) or Groq. Or skip it and use Ollama.
4. **Slack.** A free workspace plus an incoming webhook for a `#sales` channel.
5. **Airtable.** A free account and an empty base called "Inboxa".
6. **HubSpot.** A free developer account with a private app token.

## Useful commands

```bash
docker compose ps                 # what's running
docker compose logs -f backend    # watch backend logs
docker compose down               # stop everything (data is kept)
docker compose down -v            # stop and DELETE all data
```
