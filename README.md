# 🧠 Arklex AI · Agent-First Framework for Intelligent Automation

![Arklex AI Logo](Arklex_AI__logo.jpeg)

[![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)](https://github.com/arklexai/Agent-First-Organization/releases)
[![PyPI](https://img.shields.io/pypi/v/arklex.svg)](https://pypi.org/project/arklex)
[![Python](https://img.shields.io/pypi/pyversions/arklex)](https://pypi.org/project/arklex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)](https://github.com/arklexai/Agent-First-Organization)

## Overview

**Arklex AI** is a modular, production-grade framework for building intelligent agents powered by LLMs, retrieval, and task graphs. Designed for developers and researchers, Arklex makes it easy to compose, run, and evaluate LLM-powered pipelines at scale.

Whether you're building customer service bots, booking systems, or complex multi-agent workflows, Arklex provides the tools and infrastructure to create robust, scalable AI applications.

### Why Arklex AI?

- **🧠 Agent-First Design** — Built specifically for multi-agent orchestration
- **🚀 Production Ready** — Enterprise-grade features out of the box
- **🔌 Model Agnostic** — Works with any LLM provider
- **📊 Built-in Evaluation** — Comprehensive testing and metrics
- **🛡️ Security Focused** — Secure by design with proper validation

## Core Concepts

Arklex AI is built around four key architectural components:

### Task Graph

Declarative DAG (Directed Acyclic Graph) that defines agent workflows. Each node represents a task, and edges define dependencies and data flow.

### Orchestrator

Core runtime that manages state, task execution, and workflow coordination. Handles error recovery, retries, and monitoring.

### Workers

Modular building blocks for specific tasks:

- **RAG Worker** — Document retrieval and question answering
- **Database Worker** — SQL operations and data persistence
- **Browser Worker** — Web automation and scraping
- **Custom Workers** — Extensible for domain-specific needs

### Tools

Atomic utilities for functional and logic extensions:

- API integrations (Shopify, HubSpot, Google Calendar)
- Data processing and transformation
- External service connectors

## Key Features

- 🧠 **Multi-agent orchestration** using structured DAGs
- 🧩 **Composable modules** for tools, databases, APIs, and browsers
- 🔌 **Model-agnostic** — OpenAI, Anthropic, Gemini, Mistral, Hugging Face
- 🧪 **Built-in evaluation** — synthetic tests, A/B runs, metrics tracking
- 🚀 **FastAPI backend** with observability, OpenAPI docs, and error handling
- 📊 **Production-ready** — structured logging, monitoring, and error handling
- 🔄 **Auto-scaling** — Handle variable load with intelligent scaling
- 🛡️ **Security** — Input validation, rate limiting, authentication

## Use Cases

Arklex AI is designed for a wide range of intelligent automation scenarios:

### Customer Service & Support

- **RAG-powered chatbots** with document knowledge
- **Multi-step support workflows** with human-in-the-loop
- **Automated ticket routing** and resolution

### E-commerce & Retail

- **Order management** and inventory tracking
- **Customer onboarding** and account management
- **Product recommendations** and personalization

### Business Process Automation

- **Appointment scheduling** and calendar management
- **CRM integration** and lead management
- **Document processing** and data extraction

### Research & Development

- **Multi-agent research workflows**
- **Data analysis** and reporting automation
- **A/B testing** and model evaluation

## Installation

### System Requirements

- **Python 3.10+** (required)
- **API Keys** for your chosen LLM providers
- **Optional**: Vector database (Milvus), SQL database (MySQL), web search API (Tavily)

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

## Getting Started

Get up and running in under 5 minutes:

### Step 1: Set Up Environment

```bash
# Install Arklex
pip install arklex

# Set up your API key
export OPENAI_API_KEY="your-api-key-here"
```

### Step 2: Create Your First Agent

```bash
# Create a customer service agent
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o-mini
```

### Step 3: Run Your Agent

```bash
# Run the agent
python run.py \
  --input-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o
```

✅ Your agent is now live and ready to use!

### Step 4: Start the API Server (Optional)

```bash
# Start FastAPI server for programmatic access
python model_api.py --input-dir ./examples/customer_service
```

The server will be available at `http://localhost:8000` with auto-generated OpenAPI documentation.

▶️ [Video: Build a Customer Service Agent in 20 Minutes](https://youtu.be/y1P2Ethvy0I)

## Examples

Explore our comprehensive examples to get started quickly:

| Example | Description | Use Case |
|---------|-------------|----------|
| [Customer Service Agent](./examples/customer_service/) | RAG-powered support with database memory | Customer support automation |
| [Shopify Integration](./examples/shopify/) | E-commerce order management | E-commerce operations |
| [HubSpot CRM](./examples/hubspot/) | Contact and deal management | CRM automation |
| [Calendar Booking](./examples/calendar/) | Multi-step scheduling system | Appointment booking |
| [Syllabus Assistant](./examples/syllabus_assistant/) | Document processing and Q&A | Content management |
| [Human-in-the-Loop](./examples/hitl_server/) | Interactive agent workflows | Complex decision making |

Each example includes:

- Complete configuration files
- Ready-to-run code
- Documentation and tutorials
- Best practices and patterns

## Configuration

### Environment Variables

Create a `.env` file with your API keys:

```env
# Required: Choose at least one LLM provider
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here
# OR
GEMINI_API_KEY=your_gemini_key_here
# OR
MISTRAL_API_KEY=your_mistral_key_here
# OR
HUGGINGFACE_API_KEY=your_huggingface_key_here

# LangChain (optional)
LANGCHAIN_API_KEY=your_langchain_key_here
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_TRACING_V2=false

# Vector Database (e.g., Milvus)
MILVUS_URI=your_milvus_uri_here

# SQL Database (MySQL)
MYSQL_USERNAME=your_mysql_username
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOSTNAME=localhost
MYSQL_PORT=3306
MYSQL_DB_NAME=your_database_name
MYSQL_CONNECTION_TIMEOUT=10

# Web Search
TAVILY_API_KEY=your_tavily_key_here
```

## API Reference

### Core Components

- **`create.py`** — Generate agent workflows from configuration
- **`run.py`** — Execute agent workflows with input data
- **`model_api.py`** — Start FastAPI server for agent interactions
- **`eval.py`** — Run evaluation and testing suites

### Key Classes

- **`Orchestrator`** — Main runtime for agent execution
- **`TaskGraph`** — DAG representation of agent workflows
- **`Worker`** — Modular components for specific tasks
- **`Tool`** — Atomic utilities for agent operations

For detailed API documentation, visit our [API Reference](https://www.arklex.ai/qa/open-source).

## Evaluation & Testing

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

## Production Deployment

Arklex AI includes enterprise-grade features for production deployments:

### Monitoring & Observability

- ✅ **Structured Logging** — JSON logs, trace IDs, log rotation
- 📈 **Monitoring Hooks** — Health checks, metrics, fallbacks
- 🔍 **Distributed Tracing** — Track requests across services

### Security & Reliability

- 🔐 **Secure API Server** — Auto-generated docs, CORS, security headers
- ⚙️ **Robust Error Handling** — Typed exceptions, retries, fallbacks
- 🛡️ **Input Validation** — Rate limiting, authentication, sanitization

### Scalability

- 🔄 **Auto-scaling** — Handle variable load with intelligent scaling
- 📊 **Performance Monitoring** — Real-time metrics and alerts
- 🚀 **High Availability** — Fault tolerance and failover support

## Supported Providers

| Provider      | Models Supported                        | Status |
|---------------|-----------------------------------------|--------|
| OpenAI        | `gpt-4o`, `gpt-4o-mini`                 | ✅ Stable |
| Anthropic     | `claude-3-5-haiku`, `claude-3-5-sonnet` | ✅ Stable |
| Google        | `gemini-2.0-flash`                      | ✅ Stable |
| Mistral       | All `mistral-*` models                  | ✅ Stable |
| Hugging Face  | Any open-source models                  | ✅ Stable |

## Development

### Prerequisites for Development

```bash
# Clone the repository
git clone https://github.com/arklexai/Agent-First-Organization.git
cd Agent-First-Organization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/env/
pytest tests/orchestrator/
pytest tests/utils/
```

### Code Quality

```bash
# Linter
ruff arklex/
```

## Troubleshooting

### Common Issues

**API Key Errors**

```bash
# Ensure your API key is set correctly
export OPENAI_API_KEY="your-actual-key-here"
echo $OPENAI_API_KEY  # Verify it's set
```

**Import Errors**

```bash
# Reinstall with all dependencies
pip uninstall arklex
pip install arklex[all]
```

**Database Connection Issues**

```bash
# Check your MySQL connection
mysql -u username -p -h hostname -P port database_name
```

## Contributing

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

## Support

### Getting Help

- 🌐 [Full Documentation](https://arklex.ai/docs)
- 📧 [Email Support](mailto:support@arklex.ai)
- 💬 [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)
- 🐛 [Bug Reports](https://github.com/arklexai/Agent-First-Organization/issues)
- 📖 [Getting Started Guide](https://arklexai.github.io/Agent-First-Organization/docs/intro)
- 🧪 [Tutorials](https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro)
- 🛠️ [Tools Documentation](https://arklexai.github.io/Agent-First-Organization/docs/Tools)

### Community

- 🐦 [Twitter](https://twitter.com/arklexai)
- 💼 [LinkedIn](https://www.linkedin.com/company/arklex)
- 📺 [YouTube](https://youtube.com/@arklexai)

## License

Arklex AI is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Thanks to all our contributors and the open-source community for making this project possible!
