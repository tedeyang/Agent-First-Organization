# 🧠 Arklex AI · Agent-First Framework for Intelligent Automation

<div align="center">

![Arklex AI Logo](Arklex_AI__logo.jpeg)

**Build, deploy, and scale intelligent AI agents with enterprise-grade reliability**

[![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)](https://github.com/arklexai/Agent-First-Organization/releases)
[![PyPI](https://img.shields.io/pypi/v/arklex.svg)](https://pypi.org/project/arklex)
[![Python](https://img.shields.io/pypi/pyversions/arklex)](https://pypi.org/project/arklex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)](https://github.com/arklexai/Agent-First-Organization)
[![Discord](https://img.shields.io/badge/discord-join%20community-7289da?logo=discord)](https://discord.gg/arklex)
[![Twitter](https://img.shields.io/badge/twitter-follow%20us-1da1f2?logo=twitter)](https://twitter.com/arklexai)

[Quick Start](#-quick-start) • [Documentation](https://arklex.ai/docs) • [Examples](./examples/) • [Community](https://github.com/arklexai/Agent-First-Organization/discussions)

</div>

---

## 📋 Table of Contents

- [🚀 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [💡 Use Cases](#-use-cases)
- [⚡ Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [📚 Examples](#-examples)
- [🔍 API Reference](#-api-reference)
- [🧪 Evaluation & Testing](#-evaluation--testing)
- [🚀 Production Deployment](#-production-deployment)
- [🤝 Contributing](#-contributing)
- [📞 Support](#-support)

---

## 🚀 Overview

**Arklex AI** is the definitive framework for building production-ready intelligent agents. Whether you're creating customer service bots, booking systems, or complex multi-agent workflows, Arklex provides the enterprise-grade tools and infrastructure you need to succeed.

### Why Choose Arklex AI?

<div align="center">

| Feature | Arklex AI | Other Frameworks |
|---------|-----------|------------------|
| **Multi-Agent Orchestration** | ✅ Built-in DAG workflows | ❌ Manual coordination |
| **Production Ready** | ✅ Enterprise features | ❌ Basic functionality |
| **Model Agnostic** | ✅ Any LLM provider | ❌ Vendor lock-in |
| **Built-in Evaluation** | ✅ Comprehensive testing | ❌ External tools needed |
| **Security First** | ✅ Enterprise security | ❌ Basic validation |

</div>

### 🎯 What Makes Arklex Special?

- **🧠 Agent-First Design** — Purpose-built for multi-agent orchestration with declarative workflows
- **🚀 Production Ready** — Enterprise-grade features including monitoring, logging, and error handling
- **🔌 Model Agnostic** — Seamlessly switch between OpenAI, Anthropic, Gemini, Mistral, and Hugging Face
- **📊 Built-in Evaluation** — Comprehensive testing suite with synthetic data generation and A/B testing
- **🛡️ Security Focused** — Secure by design with input validation, rate limiting, and authentication
- **⚡ High Performance** — Optimized for scale with intelligent caching and auto-scaling

---

## ✨ Key Features

### 🧠 Multi-Agent Orchestration

- **Declarative DAGs** — Define complex workflows with simple JSON configuration
- **State Management** — Built-in state persistence and recovery
- **Error Handling** — Automatic retries, fallbacks, and graceful degradation
- **Parallel Execution** — Optimize performance with concurrent task execution

### 🧩 Composable Architecture

- **Modular Workers** — RAG, Database, Browser, and Custom workers
- **Extensible Tools** — 50+ pre-built tools with easy custom tool creation
- **Plugin System** — Add new capabilities without modifying core code
- **API Integrations** — Shopify, HubSpot, Google Calendar, and more

### 🔌 Model Agnostic

```python
# Switch between providers seamlessly
orchestrator = Orchestrator(
    llm_provider="openai",      # or "anthropic", "gemini", "mistral"
    model="gpt-4o",            # any model from your chosen provider
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### 📊 Enterprise Features

- **Structured Logging** — JSON logs with trace IDs and correlation
- **Monitoring & Metrics** — Real-time performance tracking and alerts
- **API Documentation** — Auto-generated OpenAPI/Swagger docs
- **Rate Limiting** — Built-in protection against abuse
- **Authentication** — JWT-based auth with role-based access control

---

## 🏗️ Architecture

Arklex AI is built around four core architectural components that work together seamlessly:

```mermaid
graph TB
    A[Task Graph] --> B[Orchestrator]
    B --> C[Workers]
    B --> D[Tools]
    C --> E[RAG Worker]
    C --> F[Database Worker]
    C --> G[Browser Worker]
    C --> H[Custom Workers]
    D --> I[API Tools]
    D --> J[Data Tools]
    D --> K[External Tools]
```

### 🎯 Task Graph

The **Task Graph** is a declarative DAG (Directed Acyclic Graph) that defines your agent workflow. Each node represents a task, and edges define dependencies and data flow.

```json
{
  "nodes": [
    {
      "id": "user_input",
      "type": "input",
      "description": "Process user query"
    },
    {
      "id": "search_docs",
      "type": "rag_worker",
      "description": "Search knowledge base",
      "dependencies": ["user_input"]
    },
    {
      "id": "generate_response",
      "type": "llm_worker",
      "description": "Generate final response",
      "dependencies": ["search_docs"]
    }
  ]
}
```

### ⚙️ Orchestrator

The **Orchestrator** is the core runtime that manages:

- **State Management** — Persistent workflow state across executions
- **Task Execution** — Parallel and sequential task scheduling
- **Error Recovery** — Automatic retries with exponential backoff
- **Monitoring** — Real-time metrics and health checks

### 🔧 Workers

**Workers** are modular building blocks for specific tasks:

| Worker Type | Purpose | Use Cases |
|-------------|---------|-----------|
| **RAG Worker** | Document retrieval and Q&A | Knowledge bases, FAQs, research |
| **Database Worker** | SQL operations and persistence | User data, analytics, logging |
| **Browser Worker** | Web automation and scraping | Data collection, form filling |
| **Custom Workers** | Domain-specific logic | Business rules, integrations |

### 🛠️ Tools

**Tools** are atomic utilities that extend agent capabilities:

```python
# Pre-built tools for common tasks
tools = [
    ShopifyTool(),           # E-commerce operations
    HubSpotTool(),           # CRM management
    GoogleCalendarTool(),    # Scheduling
    CalculatorTool(),        # Mathematical operations
    WebSearchTool(),         # Real-time information
    CustomTool()             # Your domain-specific logic
]
```

---

## 💡 Use Cases

Arklex AI excels in these intelligent automation scenarios:

### 🎯 Customer Service & Support

- **RAG-powered chatbots** with instant access to knowledge bases
- **Multi-step support workflows** with human-in-the-loop escalation
- **Automated ticket routing** and intelligent resolution
- **24/7 customer assistance** with context-aware responses

### 🛒 E-commerce & Retail

- **Order management** with real-time inventory tracking
- **Customer onboarding** and account management automation
- **Product recommendations** based on user behavior
- **Returns processing** with intelligent decision making

### 🏢 Business Process Automation

- **Appointment scheduling** with calendar integration
- **CRM operations** including lead scoring and follow-ups
- **Document processing** with intelligent data extraction
- **Workflow automation** across multiple systems

### 🔬 Research & Development

- **Multi-agent research workflows** for complex analysis
- **Data analysis** and automated reporting
- **A/B testing** with statistical significance tracking
- **Model evaluation** and performance monitoring

---

## ⚡ Quick Start

Get your first intelligent agent running in **under 5 minutes**:

### 🚀 Step 1: Install Arklex AI

```bash
# Install the core framework
pip install arklex

# Install with optional dependencies for enhanced functionality
pip install arklex[all]
```

### 🔑 Step 2: Set Up Your Environment

Create a `.env` file with your API keys:

```env
# Choose your preferred LLM provider
OPENAI_API_KEY=your_openai_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_key_here
# OR
GEMINI_API_KEY=your_gemini_key_here

# Optional: Vector database for RAG
MILVUS_URI=your_milvus_uri_here

# Optional: SQL database for persistence
MYSQL_USERNAME=your_username
MYSQL_PASSWORD=your_password
MYSQL_HOSTNAME=localhost
MYSQL_DB_NAME=arklex_db
```

### 🎯 Step 3: Create Your First Agent

```bash
# Create a customer service agent with RAG capabilities
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o-mini
```

### 🏃‍♂️ Step 4: Run Your Agent

```bash
# Start your agent with a test query
python run.py \
  --input-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o \
  --query "How do I reset my password?"
```

### 🌐 Step 5: Deploy as API (Optional)

```bash
# Start FastAPI server for programmatic access
python model_api.py --input-dir ./examples/customer_service
```

Your agent is now available at `http://localhost:8000` with auto-generated OpenAPI documentation!

▶️ **[Watch: Build a Customer Service Agent in 20 Minutes](https://youtu.be/y1P2Ethvy0I)**

---

## 📦 Installation

### System Requirements

- **Python 3.10+** (required)
- **8GB RAM** (recommended for production)
- **API Keys** for your chosen LLM providers
- **Optional**: Vector database (Milvus), SQL database (MySQL), web search API (Tavily)

### Installation Options

```bash
# Basic installation (core functionality)
pip install arklex

# With vector database support
pip install arklex[milvus]

# With e-commerce integrations
pip install arklex[shopify]

# With CRM integrations
pip install arklex[hubspot]

# Complete installation with all features
pip install arklex[all]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/arklexai/Agent-First-Organization.git
cd Agent-First-Organization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

---

## 🔧 Configuration

### Environment Variables

Arklex AI uses environment variables for configuration. Create a `.env` file in your project root:

```env
# =============================================================================
# REQUIRED: Choose at least one LLM provider
# =============================================================================

# OpenAI (recommended for production)
OPENAI_API_KEY=your_openai_key_here
OPENAI_ORG_ID=your_org_id_here  # Optional

# Anthropic (alternative)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google Gemini (alternative)
GEMINI_API_KEY=your_gemini_key_here

# Mistral (alternative)
MISTRAL_API_KEY=your_mistral_key_here

# Hugging Face (for open-source models)
HUGGINGFACE_API_KEY=your_huggingface_key_here

# =============================================================================
# OPTIONAL: Enhanced functionality
# =============================================================================

# LangChain integration
LANGCHAIN_API_KEY=your_langchain_key_here
LANGCHAIN_PROJECT=AgentOrg
LANGCHAIN_TRACING_V2=false

# Vector Database (Milvus)
MILVUS_URI=your_milvus_uri_here
MILVUS_USERNAME=your_milvus_username
MILVUS_PASSWORD=your_milvus_password

# SQL Database (MySQL)
MYSQL_USERNAME=your_mysql_username
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOSTNAME=localhost
MYSQL_PORT=3306
MYSQL_DB_NAME=arklex_db
MYSQL_CONNECTION_TIMEOUT=10

# Web Search (Tavily)
TAVILY_API_KEY=your_tavily_key_here

# =============================================================================
# OPTIONAL: Production settings
# =============================================================================

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
JWT_SECRET=your_jwt_secret_here
RATE_LIMIT_PER_MINUTE=100
```

### Configuration Files

Create agent-specific configuration files in JSON format:

```json
{
  "name": "Customer Service Agent",
  "description": "RAG-powered support agent with database memory",
  "version": "1.0.0",
  "orchestrator": {
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "workers": {
    "rag_worker": {
      "enabled": true,
      "vector_db": "milvus",
      "collection_name": "customer_support_docs"
    },
    "database_worker": {
      "enabled": true,
      "connection_string": "mysql://user:pass@localhost/arklex_db"
    }
  },
  "tools": [
    "shopify_tool",
    "hubspot_tool",
    "calculator_tool"
  ]
}
```

---

## 📚 Examples

Explore our comprehensive examples to get started quickly:

| Example | Description | Complexity | Use Case |
|---------|-------------|------------|----------|
| [Customer Service Agent](./examples/customer_service/) | RAG-powered support with database memory | ⭐⭐ | Customer support automation |
| [Shopify Integration](./examples/shopify/) | E-commerce order management | ⭐⭐⭐ | E-commerce operations |
| [HubSpot CRM](./examples/hubspot/) | Contact and deal management | ⭐⭐⭐ | CRM automation |
| [Calendar Booking](./examples/calendar/) | Multi-step scheduling system | ⭐⭐ | Appointment booking |
| [Syllabus Assistant](./examples/syllabus_assistant/) | Document processing and Q&A | ⭐⭐ | Content management |
| [Human-in-the-Loop](./examples/hitl_server/) | Interactive agent workflows | ⭐⭐⭐⭐ | Complex decision making |

### 🎯 Example: Customer Service Agent

```python
from arklex import Orchestrator, TaskGraph
from arklex.workers import RAGWorker, DatabaseWorker
from arklex.tools import ShopifyTool, HubSpotTool

# Create orchestrator
orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define task graph
task_graph = TaskGraph([
    {
        "id": "process_query",
        "type": "input",
        "description": "Process user query"
    },
    {
        "id": "search_knowledge",
        "type": "rag_worker",
        "description": "Search knowledge base",
        "dependencies": ["process_query"]
    },
    {
        "id": "check_user_data",
        "type": "database_worker",
        "description": "Retrieve user information",
        "dependencies": ["process_query"]
    },
    {
        "id": "generate_response",
        "type": "llm_worker",
        "description": "Generate final response",
        "dependencies": ["search_knowledge", "check_user_data"]
    }
])

# Add workers and tools
orchestrator.add_worker(RAGWorker(vector_db="milvus"))
orchestrator.add_worker(DatabaseWorker())
orchestrator.add_tool(ShopifyTool())
orchestrator.add_tool(HubSpotTool())

# Execute workflow
result = orchestrator.run(task_graph, query="How do I reset my password?")
print(result.response)
```

Each example includes:

- ✅ **Complete configuration files**
- ✅ **Ready-to-run code**
- ✅ **Step-by-step tutorials**
- ✅ **Best practices and patterns**
- ✅ **Testing and evaluation scripts**

---

## 🔍 API Reference

### Core Components

| Component | Purpose | Usage |
|-----------|---------|-------|
| **`create.py`** | Generate agent workflows from configuration | `python create.py --config config.json` |
| **`run.py`** | Execute agent workflows with input data | `python run.py --input-dir ./agent` |
| **`model_api.py`** | Start FastAPI server for agent interactions | `python model_api.py --input-dir ./agent` |
| **`eval.py`** | Run evaluation and testing suites | `python eval.py --config config.json` |

### Key Classes

#### Orchestrator

The main runtime for agent execution:

```python
from arklex import Orchestrator

orchestrator = Orchestrator(
    llm_provider="openai",
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=1000
)
```

#### TaskGraph

DAG representation of agent workflows:

```python
from arklex import TaskGraph

task_graph = TaskGraph([
    {
        "id": "task_1",
        "type": "input",
        "description": "Process input"
    },
    {
        "id": "task_2",
        "type": "llm_worker",
        "description": "Generate response",
        "dependencies": ["task_1"]
    }
])
```

#### Workers

Modular components for specific tasks:

```python
from arklex.workers import RAGWorker, DatabaseWorker

# RAG Worker for document retrieval
rag_worker = RAGWorker(
    vector_db="milvus",
    collection_name="documents"
)

# Database Worker for persistence
db_worker = DatabaseWorker(
    connection_string="mysql://user:pass@localhost/db"
)
```

#### Tools

Atomic utilities for agent operations:

```python
from arklex.tools import ShopifyTool, HubSpotTool

# Add tools to orchestrator
orchestrator.add_tool(ShopifyTool())
orchestrator.add_tool(HubSpotTool())
```

For detailed API documentation, visit our [API Reference](https://www.arklex.ai/qa/open-source).

---

## 🧪 Evaluation & Testing

Arklex AI includes comprehensive evaluation tools for robust validation and debugging:

### 🔁 Synthetic Testing

Generate realistic test scenarios automatically:

```bash
python eval.py \
  --model_api http://127.0.0.1:8000/eval/chat \
  --config ./examples/customer_service/customer_service_config.json \
  --documents_dir ./examples/customer_service \
  --output-dir ./examples/customer_service \
  --synthetic_tests 100
```

### 🧪 A/B Comparison

Compare different models, configurations, and prompts:

```python
from arklex.evaluation import ABComparison

comparison = ABComparison(
    model_a="gpt-4o",
    model_b="claude-3-5-sonnet",
    test_cases=test_cases
)

results = comparison.run()
comparison.generate_report("ab_comparison_results.html")
```

### 📊 Metrics Dashboard

Track performance metrics in real-time:

- **Latency** — Response time analysis
- **Success Rate** — Task completion percentage
- **Quality Score** — Response relevance and accuracy
- **Cost Analysis** — Token usage and API costs
- **Error Tracking** — Failure patterns and debugging

### 🐛 Debug Suite

Comprehensive debugging tools:

```python
# Enable debug mode
orchestrator.debug = True

# Access detailed logs
logs = orchestrator.get_logs()

# Trace execution flow
trace = orchestrator.get_execution_trace()

# Monitor resource usage
metrics = orchestrator.get_metrics()
```

---

## 🚀 Production Deployment

Arklex AI includes enterprise-grade features for production deployments:

### 📊 Monitoring & Observability

#### Structured Logging

```python
import logging
from arklex.middleware import LoggingMiddleware

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add logging middleware
orchestrator.add_middleware(LoggingMiddleware())
```

#### Health Checks

```python
# Built-in health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }
```

#### Metrics Collection

```python
from arklex.monitoring import MetricsCollector

# Collect performance metrics
metrics = MetricsCollector()
orchestrator.add_middleware(metrics)

# Export metrics to Prometheus
@app.get("/metrics")
async def metrics_endpoint():
    return metrics.export_prometheus()
```

### 🔐 Security & Reliability

#### Authentication & Authorization

```python
from arklex.security import JWTAuthMiddleware

# Add JWT authentication
auth_middleware = JWTAuthMiddleware(
    secret_key=os.getenv("JWT_SECRET"),
    algorithm="HS256"
)
orchestrator.add_middleware(auth_middleware)
```

#### Rate Limiting

```python
from arklex.security import RateLimitMiddleware

# Add rate limiting
rate_limit = RateLimitMiddleware(
    requests_per_minute=100,
    burst_size=20
)
orchestrator.add_middleware(rate_limit)
```

#### Input Validation

```python
from arklex.validation import InputValidator

# Validate all inputs
validator = InputValidator()
orchestrator.add_middleware(validator)
```

### ⚡ Scalability

#### Auto-scaling Configuration

```python
# Configure auto-scaling
orchestrator.configure_scaling(
    min_workers=2,
    max_workers=10,
    scale_up_threshold=0.8,
    scale_down_threshold=0.2
)
```

#### Load Balancing

```python
# Multiple orchestrator instances
orchestrators = [
    Orchestrator(port=8001),
    Orchestrator(port=8002),
    Orchestrator(port=8003)
]

# Load balancer configuration
load_balancer = LoadBalancer(orchestrators)
```

#### Caching

```python
from arklex.caching import RedisCache

# Add Redis caching
cache = RedisCache(
    host="localhost",
    port=6379,
    db=0
)
orchestrator.add_cache(cache)
```

### 🐳 Docker Deployment

```dockerfile
# Dockerfile for production deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "model_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml for full stack deployment
version: '3.8'

services:
  arklex:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MILVUS_URI=${MILVUS_URI}
    depends_on:
      - milvus
      - mysql

  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus

  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=${MYSQL_PASSWORD}
      - MYSQL_DATABASE=arklex_db
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  milvus_data:
  mysql_data:
```

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### 🚀 How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with clear commit messages
4. **Test** your changes thoroughly
5. **Push** to your branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request with detailed description

### 📋 Contribution Guidelines

#### Code Style

- Follow PEP 8 for Python code
- Use type hints for all functions
- Write comprehensive docstrings
- Keep functions small and focused

#### Testing

- Add tests for new features
- Ensure all tests pass
- Maintain good test coverage
- Include integration tests for complex features

#### Documentation

- Update README for new features
- Add docstrings to new functions
- Include usage examples
- Update API documentation

### 🎯 Areas for Contribution

- **New Workers** — Domain-specific worker implementations
- **Additional Tools** — Integration with new services and APIs
- **Performance Optimization** — Improve speed and efficiency
- **Documentation** — Tutorials, guides, and examples
- **Testing** — Test cases and evaluation scenarios
- **Bug Fixes** — Identify and resolve issues

### 📚 Resources

- 📘 [Contributing Guide](CONTRIBUTING.md)
- 🐛 [Report Issues](https://github.com/arklexai/Agent-First-Organization/issues)
- 💬 [Start a Discussion](https://github.com/arklexai/Agent-First-Organization/discussions)
- 📋 [Code of Conduct](CODE_OF_CONDUCT.md)
- 🧪 [Development Setup](https://arklex.ai/docs/development)

---

## 📞 Support

### 🆘 Getting Help

We're here to help you succeed with Arklex AI!

#### 📖 Documentation

- 🌐 [Full Documentation](https://arklex.ai/docs)
- 📖 [Getting Started Guide](https://arklexai.github.io/Agent-First-Organization/docs/intro)
- 🧪 [Tutorials](https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro)
- 🛠️ [Tools Documentation](https://arklexai.github.io/Agent-First-Organization/docs/Tools)
- 🔍 [API Reference](https://www.arklex.ai/qa/open-source)

#### 💬 Community Support

- 💬 [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)
- 🐦 [Twitter](https://twitter.com/arklexai)
- 💼 [LinkedIn](https://www.linkedin.com/company/arklex)
- 📺 [YouTube](https://youtube.com/@arklexai)

#### 🐛 Bug Reports & Issues

- 🐛 [Bug Reports](https://github.com/arklexai/Agent-First-Organization/issues)
- 📧 [Email Support](mailto:support@arklex.ai)

### 🔧 Troubleshooting

#### Common Issues

**API Key Errors**

```bash
# Verify your API key is set correctly
export OPENAI_API_KEY="your-actual-key-here"
echo $OPENAI_API_KEY  # Should display your key

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

**Import Errors**

```bash
# Reinstall with all dependencies
pip uninstall arklex
pip install arklex[all]

# Check Python version
python --version  # Should be 3.10+
```

**Database Connection Issues**

```bash
# Test MySQL connection
mysql -u username -p -h hostname -P port database_name

# Check Milvus connection
curl http://localhost:19530/health
```

**Memory Issues**

```bash
# Monitor memory usage
htop  # or top on macOS

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Performance Optimization

**Slow Response Times**

```python
# Enable caching
orchestrator.enable_caching()

# Use smaller models for development
orchestrator.model = "gpt-4o-mini"  # Faster than gpt-4o

# Optimize worker configuration
rag_worker.configure(
    batch_size=10,
    max_concurrent_requests=5
)
```

**High Memory Usage**

```python
# Limit concurrent workers
orchestrator.max_concurrent_workers = 4

# Enable garbage collection
import gc
gc.enable()

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

---

## 📄 License

Arklex AI is released under the **MIT License**. See [LICENSE](LICENSE) for details.

This means you can:

- ✅ Use Arklex AI for commercial projects
- ✅ Modify and distribute the code
- ✅ Use it in proprietary applications
- ✅ Sell applications built with Arklex AI

The only requirement is that you include the original license and copyright notice.

---

## 🙏 Acknowledgments

Thanks to all our contributors and the open-source community for making this project possible!

### 🌟 Contributors

<a href="https://github.com/arklexai/Agent-First-Organization/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=arklexai/Agent-First-Organization" />
</a>

### 🤝 Open Source Dependencies

Arklex AI builds on the shoulders of giants:

- **LangChain** — LLM framework and tooling
- **FastAPI** — Modern web framework
- **Pydantic** — Data validation
- **SQLAlchemy** — Database ORM
- **Milvus** — Vector database
- **And many more...**

---

<div align="center">

**Made with ❤️ by the Arklex AI Team**

[Website](https://arklex.ai) • [Documentation](https://arklex.ai/docs) • [GitHub](https://github.com/arklexai/Agent-First-Organization) • [Discord](https://discord.gg/arklex)

</div>
