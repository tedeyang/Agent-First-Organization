# 🧠 Arklex AI · Agent-First Organization

![Arklex AI Logo](Arklex_AI__logo.jpeg)

<a href="https://github.com/arklexai/Agent-First-Organization/releases" target="_blank" rel="noopener noreferrer">![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)</a>
<a href="https://pypi.org/project/arklex" target="_blank" rel="noopener noreferrer">![PyPI](https://img.shields.io/pypi/v/arklex.svg)</a>
<a href="https://pypi.org/project/arklex" target="_blank" rel="noopener noreferrer">![Python](https://img.shields.io/pypi/pyversions/arklex)</a>

**Arklex AI** is a modular, agent-first framework built to orchestrate intelligent agents through structured task graphs. Designed for developers, researchers, and AI product teams, Arklex AI provides a powerful foundation for building multi-agent systems with LLMs, vector search, databases, and automated evaluation baked in.

---

## ✨ Key Features

- **Task Graph Orchestration** – Build agent workflows using DAG-based coordination
- **Composable Workers & Tools** – Modular architecture with plug-and-play components
- **Multi-LLM Compatibility** – Integrates seamlessly with OpenAI, Anthropic, Gemini, Mistral, Hugging Face
- **Built-in RAG + DB Access** – Native support for retrieval-augmented generation and SQL databases
- **Evaluation Toolkit** – Run synthetic tests, A/B experiments, and track performance
- **API Ready** – FastAPI backend with full monitoring, logging, and OpenAPI docs

---

## 🚀 Quick Start

### 1. Install

```bash
pip install arklex
```

### 2. Configure Environment

Create a `.env` file with your API keys and DB credentials:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
# See full list of supported variables below ↓
```

### 3. Create & Launch an Agent

```bash
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service

python run.py --input-dir ./examples/customer_service
```

✅ Your agent is live and operational.

---

## 📚 Documentation

- 📖 <a href="https://arklexai.github.io/Agent-First-Organization/docs/intro" target="_blank" rel="noopener noreferrer">Getting Started</a> — Install & configure
- 🧪 <a href="https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro" target="_blank" rel="noopener noreferrer">Tutorials</a> — Step-by-step walkthroughs
- ⚙️ <a href="https://www.arklex.ai/qa/open-source" target="_blank" rel="noopener noreferrer">API Reference</a> — CLI, Python APIs, and configs

---

## 🛠 Supported Use Cases

| Use Case           | Description                                                 |
|--------------------|-------------------------------------------------------------|
| Customer Support   | Conversational agents with RAG + DB memory                  |
| Booking Systems    | Calendar integration and multi-modal scheduling             |
| Data Analysis      | LLM-driven pipelines with code generation and visualization |
| Content Generation | Assistive co-writing tools for docs, blogs, and more        |

▶️ <a href="https://youtu.be/y1P2Ethvy0I" target="_blank" rel="noopener noreferrer">Watch: Build a Customer Service Agent in 20 Minutes</a>

---

## 🧱 Architecture Overview

Arklex AI combines central orchestration with distributed execution:

- **Task Graph** - Declarative workflows defining task dependencies
- **Orchestrator** - Central engine that routes logic and coordinates workers
- **Workers** - Pluggable units (DB, RAG, browser, API, etc.)
- **Tools** - Reusable atomic functions for agent execution

---

## ⚙️ Configuration

### Supported LLM Providers

| Provider      | Models Supported                        |
|---------------|-----------------------------------------|
| OpenAI        | `gpt-4o`, `gpt-4o-mini`                 |
| Anthropic     | `claude-3-5-haiku`, `claude-3-5-sonnet` |
| Google        | `gemini-2.0-flash`                      |
| Mistral       | All `mistral-*` models                  |
| Hugging Face  | Any open-source models                  |

### `.env` Variables

```env
# LLM API Keys
OPENAI_API_KEY=<your-openai-api-key>
GEMINI_API_KEY=<your-gemini-api-key>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
MISTRAL_API_KEY=<your-mistral-api-key>
HUGGINGFACE_API_KEY=<your-huggingface-api-key>

# LangChain (optional)
LANGCHAIN_API_KEY=<your-langchain-api-key>
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_TRACING_V2=false

# Vector Search (e.g., Milvus)
MILVUS_URI=<your-milvus-db-uri>

# SQL Database (MySQL)
MYSQL_USERNAME=<your-mysql-db-username>
MYSQL_PASSWORD=<your-mysql-db-password>
MYSQL_HOSTNAME=<your-mysql-db-hostname>
MYSQL_PORT=<your-mysql-db-port>
MYSQL_DB_NAME=<your-mysql-db-name>
MYSQL_CONNECTION_TIMEOUT=<your-mysql-db-timeout>

# Tavily (Web Search)
TAVILY_API_KEY=<your-tavily-api-key>
```

---

## 🧪 Evaluation & Testing

Built-in tools for robust validation and debugging:

- 🔄 Synthetic Conversations – Auto-generate realistic testing scenarios
- 📊 Metrics Dashboard – Track latency, task success, LLM output quality
- 🧪 A/B Testing – Compare model outputs or workflows side-by-side
- 🐞 Debugging Tools – Trace logs, retry logic, error reports

```bash
python eval.py \
  --model_api http://127.0.0.1:8000/eval/chat \
  --config ./examples/customer_service/customer_service_config.json \
  --documents_dir ./examples/customer_service \
  --output-dir ./examples/customer_service
```

---

## 🔌 CLI Essentials

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

## 🛡️ Production-Ready Features

- **Structured Logging** – Request-level trace IDs, JSON logs, log rotation
- **Monitoring** – Circuit breakers, metrics, health checks
- **API Server** – FastAPI w/ OpenAPI (/docs), CORS, security headers
- **Robust Error Handling** – Graceful retries, typed exceptions, fallbacks

---

## 🤝 Contributing

We’d love your input!

- 📖 <a href="CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">Contribution Guide</a>
- 📝 <a href="https://github.com/arklexai/Agent-First-Organization/issues" target="_blank" rel="noopener noreferrer">File an Issue</a>
- 💬 <a href="https://github.com/arklexai/Agent-First-Organization/discussions" target="_blank" rel="noopener noreferrer">Join the Discussion</a>

---

## 📄 License

Licensed under the MIT License. See <a href="LICENSE" target="_blank" rel="noopener noreferrer">LICENSE</a> for details.

---

## 📬 Support

- 🌐 <a href="arklex.ai/docs" target="_blank" rel="noopener noreferrer">Full Docs</a>
