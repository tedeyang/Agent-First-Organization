# 🧠 Arklex AI · Agent-First Framework for Intelligent Automation

![Arklex AI Logo](Arklex_AI__logo.jpeg)

[![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)](https://github.com/arklexai/Agent-First-Organization/releases)
[![PyPI](https://img.shields.io/pypi/v/arklex.svg)](https://pypi.org/project/arklex)
[![Python](https://img.shields.io/pypi/pyversions/arklex)](https://pypi.org/project/arklex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Use Cases](#-use-cases)
- [Supported Providers](#-supported-providers)
- [Evaluation & Testing](#-evaluation--testing)
- [Production Features](#-production-features)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Support](#-support)
- [License](#-license)

---

## 🎯 Overview

**Arklex AI** is a modular, production-grade framework for building intelligent agents powered by LLMs, retrieval, and task graphs. Designed for developers and researchers, Arklex makes it easy to compose, run, and evaluate LLM-powered pipelines at scale.

### ✨ Key Features

- 🧠 **Multi-agent orchestration** using structured DAGs
- 🧩 **Composable modules** for tools, databases, APIs, and browsers
- 🔌 **Model-agnostic** — OpenAI, Anthropic, Gemini, Mistral, Hugging Face
- 🧪 **Built-in evaluation** — synthetic tests, A/B runs, metrics tracking
- 🚀 **FastAPI backend** with observability, OpenAPI docs, and error handling
- 📊 **Production-ready** — structured logging, monitoring, and error handling

---

## 🧱 Architecture

Arklex AI is designed for scalable, flexible agent development:

- **Task Graph** — Declarative DAG for agent workflows
- **Orchestrator** — Core runtime managing state and task flow
- **Workers** — Modular building blocks (e.g., RAG, Database, Browser)
- **Tools** — Atomic utilities for functional and logic extensions

---

## 📋 Prerequisites

- **Python 3.10+** (required)
- **API Keys** for your chosen LLM providers
- **Optional**: Vector database (Milvus), SQL database (MySQL), web search API (Tavily)

---

## 🚀 Installation

### Basic Installation

```bash
pip install arklex
```

### Optional Dependencies

For specific integrations, install additional packages:

```bash
# Vector database support (Milvus)
pip install arklex[milvus]

# E-commerce integration (Shopify)
pip install arklex[shopify]

# CRM integration (HubSpot)
pip install arklex[hubspot]

# All optional dependencies
pip install arklex[milvus,shopify,hubspot]
```

---

## ⚡ Quick Start

### 1. Configure Environment

Create a `.env` file with your API keys:

```env
# Required: Choose at least one LLM provider
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here
# OR
GEMINI_API_KEY=your_gemini_key_here

# Optional: Additional services
TAVILY_API_KEY=your_tavily_key_here
MILVUS_URI=your_milvus_uri_here
```

### 2. Launch Your First Agent

```bash
# Create a customer service agent
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service

# Run the agent
python run.py --input-dir ./examples/customer_service
```

✅ Your agent is now live and ready to use!

▶️ [Video: Build a Customer Service Agent in 20 Minutes](https://youtu.be/y1P2Ethvy0I)

---

## ⚙️ Configuration

### Environment Variables

```env
# LLM Providers (choose one or more)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
HUGGINGFACE_API_KEY=...

# LangChain (optional)
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_TRACING_V2=false

# Vector Database (e.g., Milvus)
MILVUS_URI=...

# SQL Database (MySQL)
MYSQL_USERNAME=...
MYSQL_PASSWORD=...
MYSQL_HOSTNAME=...
MYSQL_PORT=3306
MYSQL_DB_NAME=...
MYSQL_CONNECTION_TIMEOUT=10

# Web Search
TAVILY_API_KEY=...
```

### CLI Commands

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

## 🛠 Use Cases

| Use Case           | Description                                            | Example |
|--------------------|--------------------------------------------------------|---------|
| Customer Support   | RAG-powered agents with database memory                | [Customer Service](./examples/customer_service/) |
| Booking Systems    | Calendar integrations and multi-step scheduling        | [Calendar](./examples/calendar/) |
| E-commerce         | Shopify integration for order management               | [Shopify](./examples/shopify/) |
| CRM Integration    | HubSpot contact and deal management                    | [HubSpot](./examples/hubspot/) |
| Data Analysis      | LLM pipelines with code generation and visualization   | [Multiple Choice](./examples/multiple_choice/) |
| Content Generation | AI co-writing for docs, blogs, and editorial workflows | [Syllabus Assistant](./examples/syllabus_assistant/) |

---

## 🤖 Supported Providers

| Provider      | Models Supported                        | Status |
|---------------|-----------------------------------------|--------|
| OpenAI        | `gpt-4o`, `gpt-4o-mini`                 | ✅ Stable |
| Anthropic     | `claude-3-5-haiku`, `claude-3-5-sonnet` | ✅ Stable |
| Google        | `gemini-2.0-flash`                      | ✅ Stable |
| Mistral       | All `mistral-*` models                  | ✅ Stable |
| Hugging Face  | Any open-source models                  | ✅ Stable |

---

## 🧪 Evaluation & Testing

Built-in tools for robust validation and debugging:

- 🔁 **Synthetic Testing** — Realistic user simulation
- 🧪 **A/B Comparison** — Compare models, chains, and prompts
- 📊 **Metrics Dashboard** — Track latency, success rates, and quality
- 🐛 **Debug Suite** — Logs, retries, tracebacks, and more

```bash
python eval.py \
  --model_api http://127.0.0.1:8000/eval/chat \
  --config ./examples/customer_service/customer_service_config.json \
  --documents_dir ./examples/customer_service \
  --output-dir ./examples/customer_service
```

---

## 🛡️ Production Features

- ✅ **Structured Logging** — JSON logs, trace IDs, log rotation
- 📈 **Monitoring Hooks** — Health checks, metrics, fallbacks
- 🔐 **Secure API Server** — Auto-generated docs, CORS, security headers
- ⚙️ **Robust Error Handling** — Typed exceptions, retries, fallbacks
- 🔄 **Auto-scaling** — Handle variable load with intelligent scaling
- 🛡️ **Security** — Input validation, rate limiting, authentication

---

## 📚 Examples

Explore our comprehensive examples to get started quickly:

- [Customer Service Agent](./examples/customer_service/) - RAG-powered support with database memory
- [Shopify Integration](./examples/shopify/) - E-commerce order management
- [HubSpot CRM](./examples/hubspot/) - Contact and deal management
- [Calendar Booking](./examples/calendar/) - Multi-step scheduling system
- [Syllabus Assistant](./examples/syllabus_assistant/) - Document processing and Q&A
- [Human-in-the-Loop](./examples/hitl_server/) - Interactive agent workflows

Each example includes:

- Complete configuration files
- Ready-to-run code
- Documentation and tutorials
- Best practices and patterns

---

## 📖 Documentation

- 📖 [Getting Started](https://arklexai.github.io/Agent-First-Organization/docs/intro)
- 🧪 [Tutorials](https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro)
- ⚙️ [API Reference](https://www.arklex.ai/qa/open-source)
- 🛠️ [Tools Documentation](https://arklexai.github.io/Agent-First-Organization/docs/Tools)
- 🏗️ [Task Graph Generation](https://arklexai.github.io/Agent-First-Organization/docs/Taskgraph/intro)
- 👥 [Workers Guide](https://arklexai.github.io/Agent-First-Organization/docs/Workers/intro)

---

## 🤝 Contributing

We welcome contributions, questions, and feature ideas!

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Resources

- 📘 [Contributing Guide](CONTRIBUTING.md)
- 🐛 [Report Issues](https://github.com/arklexai/Agent-First-Organization/issues)
- 💬 [Start a Discussion](https://github.com/arklexai/Agent-First-Organization/discussions)
- 📋 [Code of Conduct](CODE_OF_CONDUCT.md)

---

## 🆘 Support

### Getting Help

- 🌐 [Full Documentation](https://arklex.ai/docs)
- 📧 [Email Support](mailto:support@arklex.ai)
- 💬 [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)
- 🐛 [Bug Reports](https://github.com/arklexai/Agent-First-Organization/issues)

### Community

- 🐦 [Twitter](https://twitter.com/arklexai)
- 💼 [LinkedIn](https://linkedin.com/company/arklexai)
- 📺 [YouTube](https://youtube.com/@arklexai)

---

## 📄 License

Arklex AI is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Thanks to all our contributors and the open-source community for making this project possible!
