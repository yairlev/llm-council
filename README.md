# LLM Council

![llmcouncil](header.jpg)

Instead of asking a question to a single LLM, group multiple models into your "LLM Council". This web app uses the **Google Agent Development Kit (ADK)** to coordinate multiple agents. Each agent can be backed by different models (Gemini, GPT, Claude via OpenRouter). The agents review and rank each other's work, and a chairman synthesizes the final response.

## How It Works

1. **Stage 1: Individual Responses** - Each agent answers the user's question independently. Responses appear in tabs so you can compare.
2. **Stage 2: Peer Review** - Agents receive anonymized versions of their peers' answers and must critique and rank them.
3. **Stage 3: Synthesis** - The chairman combines the best ideas and delivers one polished answer.

Conversations are stateful: follow-up questions include prior context.

## Features

- **Multi-model support** - Mix Gemini, GPT, Claude, and other models via OpenRouter
- **Real-time streaming** - See individual agent responses as they complete
- **RTL language support** - Proper display of Hebrew, Arabic, and other RTL languages
- **Tool usage** - Agents can search the web, browse URLs, and read repository files

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

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global

# Optional: for non-Google models (GPT, Claude, etc.)
OPENROUTER_API_KEY=your-openrouter-key
```

### 3. Configure Council Members (Optional)

The default council includes Gemini and GPT models. Override via environment variables:

```bash
# Custom council members (name:model pairs)
ADK_COUNCIL_MEMBERS=Lyra:gemini-3-flash-preview,Orion:gemini-3-pro-preview,Vega:openrouter/openai/gpt-5.2-chat

# Chairman model (synthesizes final answer)
ADK_CHAIRMAN_MODEL=gemini-3-pro-preview
```

See `.env.example` for all available options.

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

## Credits

Originally created by [Andrej Karpathy](https://github.com/karpathy/llm-council).
