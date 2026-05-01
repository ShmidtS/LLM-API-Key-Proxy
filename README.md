# Universal LLM API Proxy

**One proxy. Any LLM provider. Zero code changes.**

Self-hosted proxy that gives you a single OpenAI-compatible API endpoint for all your LLM providers. Works with Claude Code, Cursor, Continue, JanitorAI, and any app that supports custom OpenAI or Anthropic base URLs.

---

## Why Use This?

- **One endpoint for everything** — configure all your providers once, use one API key
- **Automatic failover** — if a key fails, it instantly tries another. No manual switching
- **Works with any app** — Claude Code, Cursor, VS Code extensions, chat UIs, custom code

---

## Quick Start

### Windows

```bash
git clone https://github.com/ShmidtS/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/proxy_app/main.py
```

Or just double-click `start_proxy.bat`.

### macOS / Linux

```bash
git clone https://github.com/ShmidtS/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/proxy_app/main.py
```

### Docker

```bash
cp .env.example .env
# edit .env and add your keys
docker compose up -d
```

> Make sure `key_usage.json` exists (run `touch key_usage.json`) before starting Docker.

---

## Connect Your App

Once running, the proxy is at:

| Setting | Value |
|---------|-------|
| **URL** | `http://127.0.0.1:8000/v1` |
| **API Key** | Your `PROXY_API_KEY` from `.env` |

### Model format: `provider/model_name`

```
gemini/gemini-2.5-flash          <- Gemini
openai/gpt-4o                    <- OpenAI
anthropic/claude-3-5-sonnet      <- Anthropic
```

Use `GET http://127.0.0.1:8000/v1/models` to see all available models.

---

## Usage Examples

<details>
<summary><b>Python (OpenAI library)</b></summary>

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="your-proxy-key")

response = client.chat.completions.create(
    model="gemini/gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary><b>curl</b></summary>

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{"model": "gemini/gemini-2.5-flash", "messages": [{"role": "user", "content": "Hello!"}]}'
```

</details>

<details>
<summary><b>Claude Code</b></summary>

Edit your Claude Code `settings.json`:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "your-proxy-key",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000"
  }
}
```

Now Claude Code talks to Gemini/OpenAI through the proxy.

</details>

<details>
<summary><b>Chat UIs (JanitorAI, SillyTavern, etc.)</b></summary>

1. Go to **API Settings**
2. Select **Custom OpenAI** mode
3. Set:
   - **URL:** `http://127.0.0.1:8000/v1`
   - **Key:** your `PROXY_API_KEY`
   - **Model:** `gemini/gemini-2.5-flash`
4. Save and chat

</details>

---

## Add Your Keys

The first time you run the proxy, an interactive menu opens. Pick **"Manage Credentials"** and follow the prompts.

Or edit `.env` directly:

```env
# Required: password for YOUR proxy (make it strong)
PROXY_API_KEY="my-secret-proxy-key"

# Add provider keys (use _1, _2, _3... for multiple keys per provider)
GEMINI_API_KEY_1="your-gemini-key"
OPENAI_API_KEY_1="your-openai-key"
ANTHROPIC_API_KEY_1="your-anthropic-key"
```

Copy `.env.example` as a starting point — it has all available variables.

### OAuth providers (Gemini CLI, Antigravity, Qwen, iFlow)

Run the credential tool and follow the browser login:

```bash
python -m rotator_library.credential_tool
```

For hosting without files (Render, Railway), run the tool locally, then pick **"Export to .env"** and copy the output to your platform.

---

## What It Does

- **Key rotation** — automatically picks the best available key for each request
- **Failover** — if a provider errors out, instantly retries with another key
- **Rate limit handling** — knows when keys are exhausted and skips them
- **Auto-cooldown** — temporarily disables failing keys, brings them back later
- **Daily reset** — usage counters reset automatically every day

---

## Supported Providers

Standard API keys: **Gemini, OpenAI, Anthropic, OpenRouter, Groq, Mistral, NVIDIA NIM, Cohere, Chutes, Z.AI, Colin, Firmware, NanoGPT, Kilo Code, OpenCode** — plus [any LiteLLM provider](https://docs.litellm.ai/docs/providers).

OAuth (login via browser): **Gemini CLI, Antigravity (Gemini 3 + Claude), Qwen Code, iFlow**.

Custom endpoints: set `<NAME>_API_BASE` + `<NAME>_API_KEY` and the proxy auto-discovers models.

See [Technical Documentation](DOCUMENTATION.md) for per-provider setup details.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `401 Unauthorized` | Check that `Authorization: Bearer` matches `PROXY_API_KEY` exactly |
| All keys on cooldown | Provider is down or keys are invalid; check logs in `logs/` |
| Model not found | Use format `provider/model_name` (e.g. `gemini/gemini-2.5-flash`) |
| OAuth fails | Make sure callback port isn't blocked by firewall |

Enable detailed logging: `python src/proxy_app/main.py --enable-request-logging`

---

## Docs

| Document | What's inside |
|----------|---------------|
| [Technical Docs](DOCUMENTATION.md) | Architecture, internals, all provider configs |
| [Library README](src/rotator_library/README.md) | Using the resilience library in your own code |
| [Deployment Guide](Deployment%20guide.md) | Hosting on Render, Railway, VPS |
| [.env.example](.env.example) | Full list of environment variables |

---

## License

- **Proxy** (`src/proxy_app/`) — MIT
- **Library** (`src/rotator_library/`) — LGPL-3.0
