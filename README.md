# Arklex Agent First Organization

![Arklex AI Logo](Arklex_AI__logo.jpeg)

[![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)](https://github.com/arklexai/Agent-First-Organization/releases)
[![PyPI](https://img.shields.io/pypi/v/arklex.svg)](https://pypi.org/project/arklex)
[![Python](https://img.shields.io/pypi/pyversions/arklex)](https://pypi.org/project/arklex)

**Arklex AI** is a modular framework for building AI agents that can handle complex, multi-step tasks through graph-based orchestration. Designed for developers, researchers, and AI product teams, Arklex enables intelligent agent workflows using customizable workers, multi-LLM support, and built-in evaluation.

---

## ✨ Highlights

- **Multi-Agent Graphs**: Coordinate agents through directed task graphs.
- **Modular Workers & Tools**: Compose agents with reusable components.
- **Multi-LLM Support**: Plug-and-play with OpenAI, Anthropic, Google Gemini, Mistral, and Hugging Face.
- **Built-in RAG & DB Support**: Out-of-the-box vector search and relational DB access.
- **Automated Evaluation**: Synthetic conversation testing and A/B comparison.
- **API-Ready**: Robust FastAPI backend with logging, monitoring, and OpenAPI docs.

---

## 🚀 Quick Start

### 1. Install Arklex AI

```bash
pip install arklex
```

### 2. Configure Your Environment

Create a `.env` file with your API keys and database credentials:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
# See full list of supported variables below ↓
```

### 3. Create and Run Your Agent

```bash
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service

python run.py --input-dir ./examples/customer_service
```

✅ Your agent is now live and handling tasks.

---

## Documentation

- [Getting Started Guide](https://arklexai.github.io/Agent-First-Organization/docs/intro) — Framework overview & setup
- [Tutorials](https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro) — Hands-on use cases
- [API Reference](https://www.arklex.ai/qa/open-source) — Full API and CLI documentation

---

## 🛠 Supported Use Cases

| Use Case           | Description                                                 |
|--------------------|-------------------------------------------------------------|
| Customer Support   | Conversational agent with RAG + DB memory                   |
| Booking Systems    | Schedule appointments with calendar tool integration        |
| Data Analysis      | Step-by-step analysis workflows with LLM + visualization    |
| Content Generation | Co-writing agents for blogs, documentation, and more        |

▶️ [Watch Tutorial: Build a Customer Service Agent in 20 Minutes](https://youtu.be/y1P2Ethvy0I)

---

## 🧱 Core Architecture

Arklex AI uses centralized orchestration + distributed execution:

- **Task Graph**: Defines task dependencies between nodes (workers)
- **Orchestrator**: Coordinates agents based on input and logic
- **Workers**: Modular components (e.g., DB, RAG, API, browser)
- **Tools**: Atomic functions usable across workflows

---

## ⚙️ Configuration

### Supported Models

| Provider      | Models Supported                      |
|---------------|---------------------------------------|
| OpenAI        | gpt-4o, gpt-4o-mini                   |
| Anthropic     | claude-3-5-haiku, claude-3-5-sonnet   |
| Google        | gemini-2.0-flash                      |
| Mistral       | All mistral-* models                  |
| Hugging Face  | Any open-source models                |

### `.env` Environment Variables

```env
OPENAI_API_KEY=<your-openai-api-key>
GEMINI_API_KEY = <your-gemini-api-key>
GOOGLE_API_KEY = <your-gemini-api-key>
ANTHROPIC_API_KEY = <your-anthropic-api-key>
HUGGINGFACE_API_KEY = <your-huggingface-api-key>
MISTRAL_API_KEY = <your-mistral-api-key>

LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_API_KEY=<your-langchain-api-key>

TAVILY_API_KEY=<your-tavily-api-key>

MYSQL_USERNAME=<your-mysql-db-username>
MYSQL_PASSWORD=<your-mysql-db-password>
MYSQL_HOSTNAME=<your-mysql-db-hostname>
MYSQL_PORT=<your-mysql-db-port>
MYSQL_DB_NAME=<your-mysql-db-name>
MYSQL_CONNECTION_TIMEOUT=<your-mysql-db-timeout>

MILVUS_URI=<your-milvus-db-uri>
```

---

## 🧪 Evaluation & Testing

Arklex AI includes a full testing suite:

- 🔄 Synthetic Conversations: Auto-generated test data
- 📊 Performance Metrics: Task accuracy, latency, and quality
- 🧪 A/B Testing: Compare agent variants
- 🐞 Debugging Tools: Logging, trace IDs, retries

Run an evaluation:

```bash
python eval.py \
  --model_api http://127.0.0.1:8000/eval/chat \
  --config ./examples/customer_service/customer_service_config.json \
  --documents_dir ./examples/customer_service \
  --output-dir ./examples/customer_service
```

---

## 🔌 Core Commands

```bash
# Create an agent workflow
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o-mini

# Run an agent
python run.py \
  --input-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o

# Start API server
python model_api.py \
  --input-dir ./examples/customer_service
```

---

## 🔐 Production-Grade Features

- **Logging**: Request-level tracing, log rotation, structured logs
- **Monitoring**: Metrics, health checks, circuit breakers
- **API**: FastAPI with `/docs`, `/redoc`, CORS, and security headers
- **Error Handling**: Retry logic, graceful fallbacks, typed exceptions

---

## 🤝 Contributing

We welcome community contributions!

- 📖 [Contributing Guide](CONTRIBUTING.md)
- 📝 [Open an Issue](https://github.com/arklexai/Agent-First-Organization/issues)
- 💬 [Join Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for full text.

---

## 📬 Support

- 🌐 [Documentation](arklex.ai/docs)
