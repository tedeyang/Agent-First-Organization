# 🧠 Arklex AI · Agent-First Framework for Intelligent Automation

![Arklex AI Logo](Arklex_AI__logo.jpeg)

[![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)](https://github.com/arklexai/Agent-First-Organization/releases)
[![PyPI](https://img.shields.io/pypi/v/arklex.svg)](https://pypi.org/project/arklex)
[![Python](https://img.shields.io/pypi/pyversions/arklex)](https://pypi.org/project/arklex)

---

## ⚡ TL;DR

**Arklex AI** is a modular, production-grade framework for building intelligent agents powered by LLMs, retrieval, and task graphs.

- 🧠 **Multi-agent orchestration** using structured DAGs
- 🧩 **Composable modules** for tools, databases, APIs, and browsers
- 🔌 **Model-agnostic**: OpenAI, Anthropic, Gemini, Mistral, Hugging Face
- 🧪 **Built-in evaluation**: synthetic tests, A/B runs, metrics tracking
- 🚀 **FastAPI backend** with observability, OpenAPI docs, and error handling

---

## ✨ Why Arklex?

**Arklex AI** empowers developers and researchers to build intelligent systems through structured task graphs, robust evaluation tools, and modular design.

Whether you're building customer support agents, data analysis workflows, or co-writing assistants, Arklex makes it easy to compose, run, and evaluate LLM-powered pipelines — at scale.

---

## 🚀 Quickstart

### 1. Install Arklex

```bash
pip install arklex
```

### 2. Set Up Environment

Create a `.env` file with your provider keys:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
# More env vars below ↓
```

### 3. Launch an Agent

```bash
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service

python run.py --input-dir ./examples/customer_service
```

✅ Done — your agent is live.

---

## 📚 Docs & Tutorials

- 📖 [Getting Started](https://arklexai.github.io/Agent-First-Organization/docs/intro)
- 🧪 [Tutorials](https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro)
- ⚙️ [API Reference](https://www.arklex.ai/qa/open-source)

---

## 🛠 Use Cases

| Use Case           | Description                                            |
|--------------------|--------------------------------------------------------|
| Customer Support   | RAG-powered agents with DB memory                      |
| Booking Systems    | Calendar integrations and multi-step scheduling        |
| Data Analysis      | LLM pipelines with code generation and visualization   |
| Content Generation | AI co-writing for docs, blogs, and editorial workflows |

▶️ [Video: Build a Customer Service Agent in 20 Minutes](https://youtu.be/y1P2Ethvy0I)

---

## 🧱 Architecture Overview

Arklex AI is designed for scalable, flexible agent development:

- **Task Graph** — Declarative DAG for agent workflows
- **Orchestrator** — Core runtime managing state and task flow
- **Workers** — Modular building blocks (e.g., RAG, DB, Browser)
- **Tools** — Atomic utilities for functional and logic extensions

---

## 🤖 Supported Providers

| Provider      | Models Supported                        |
|---------------|-----------------------------------------|
| OpenAI        | `gpt-4o`, `gpt-4o-mini`                 |
| Anthropic     | `claude-3-5-haiku`, `claude-3-5-sonnet` |
| Google        | `gemini-2.0-flash`                      |
| Mistral       | All `mistral-*` models                  |
| Hugging Face  | Any open-source models                  |

---

## 🔐 `.env` Config Example

```env
# LLM Providers
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
HUGGINGFACE_API_KEY=...

# LangChain (optional)
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_TRACING_V2=false

# Vector DB (e.g. Milvus)
MILVUS_URI=...

# SQL DB (MySQL)
MYSQL_USERNAME=...
MYSQL_PASSWORD=...
MYSQL_HOSTNAME=...
MYSQL_PORT=3306
MYSQL_DB_NAME=...
MYSQL_CONNECTION_TIMEOUT=10

# Web Search
TAVILY_API_KEY=...
```

---

## 🧪 Built-In Evaluation Tools

- 🔁 Synthetic Testing — Realistic user simulation
- 🧪 A/B Comparison — Compare models, chains, and prompts
- 📊 Metrics Dashboard — Track latency, success rates, and quality
- 🐛 Debug Suite — Logs, retries, tracebacks, and more

```bash
python eval.py \
  --model_api http://127.0.0.1:8000/eval/chat \
  --config ./examples/customer_service/customer_service_config.json \
  --documents_dir ./examples/customer_service \
  --output-dir ./examples/customer_service
```

---

## 🧰 CLI Essentials

```bash
# Create a new agent workflow
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o-mini

# Run the agent
python run.py \
  --input-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o

# Start the model API server
python model_api.py \
  --input-dir ./examples/customer_service
```

---

## 🛡️ Production Ready

- ✅ **Structured Logging** — JSON logs, trace IDs, rotation
- 📈 **Monitoring Hooks** — Health checks, metrics, fallbacks
- 🔐 **Secure API Server** — Auto docs, CORS, security headers
- ⚙️ **Robust Error Handling** — Typed exceptions, retries, fallbacks

---

## 🤝 Join the Community

We welcome contributions, questions, and feature ideas!

- 📘 [Contributing Guide](CONTRIBUTING.md)
- 🐛 [Report Issues](https://github.com/arklexai/Agent-First-Organization/issues)
- 💬 [Start a Discussion](https://github.com/arklexai/Agent-First-Organization/discussions)

---

## 📄 License

Arklex AI is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 💡 Need Help?

- 🌐 [Full Documentation](arklex.ai/docs)
- 📬 Reach out or open an issue — we're here to help.
