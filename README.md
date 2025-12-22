# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT 5.1, Google Gemini 3.0 Pro, Anthropic Claude Sonnet 4.5, xAI Grok 4, eg.c), you can group them into your "LLM Council". This repo is a simple, local web app that essentially looks like ChatGPT except it now uses the **Google Agent Development Kit (ADK)** to coordinate multiple Gemini-based agents. A chairman agent dispatches your query to sub agents (each backed by a different Gemini model), the agents review and rank each other's work, and finally the chairman produces the definitive response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The chairman asks each ADK sub-agent (configured with a unique Gemini model) to answer the user prompt. Their responses show up in a tab view so you can inspect every opinion.
2. **Stage 2: Review**. The same agents receive anonymized versions of their peers' answers and must critique and rank them using the strict `FINAL RANKING` format.
3. **Stage 3: Final response**. The chairman agent combines the best ideas, resolves disagreements, and delivers one polished answer.

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore and evaluate a number of LLMs side by side in the process of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). It's nice and useful to see multiple responses side by side, and also the cross-opinions of all LLMs on each other's outputs. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Keys

Create a `.env` file in the project root and add your Google Generative AI key (the ADK looks for `GOOGLE_API_KEY`). You can optionally override the built-in council roster with `ADK_COUNCIL_MEMBERS` (comma-separated `name:model` pairs).

```bash
GOOGLE_API_KEY=sk-your-gemini-key
# Optional: override the default Gemini pairings
# ADK_COUNCIL_MEMBERS="Orion:gemini-2.5-flash-preview-05-20,Lyra:gemini-2.0-flash-thinking-exp-01-21,Vega:gemini-1.5-pro-002"
```

### 3. Tune the ADK Council (Optional)

`backend/config.py` exposes additional toggles:

- `ADK_CHAIRMAN_MODEL` and `ADK_TITLE_MODEL` let you pick which Gemini model synthesizes the final answer/title.
- `ADK_COUNCIL_MEMBERS` (env var) defines the chair's sub-agents — same instructions, different models.
- `ADK_FILE_TOOL_MAX_BYTES` / `ADK_WEB_TOOL_MAX_CHARS` control how much text the built-in tools can surface.
- `ADK_ALLOWED_FILE_ROOT` lets you scope which part of the repo agents are allowed to read.

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), Google Agent Development Kit (Gemini), async httpx (for tool calls)
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript

## ADK Agents & Tools

- **Chairman agent:** Coordinates question routing, tracks stage output, and synthesizes the final message.
- **Sub agents:** Identical instructions but each runs on a different Gemini model to maximize diversity of thought.
- **Tooling:** Every agent can call `browse_web` (fetch/clean external URLs) and `read_repository_file` (inspect files or directories inside this repo). Use them when a prompt depends on fresh information or code context.
