# Arklex Agent First Organization

![Arklex AI](Arklex_AI__logo.jpeg)

![Release](https://img.shields.io/github/release/arklexai/Agent-First-Organization?logo=github)
![PyPI version](https://img.shields.io/pypi/v/arklex.svg)
![Python version](https://img.shields.io/pypi/pyversions/arklex)

A modular framework for building AI agents that complete complex tasks through structured multi-agent workflows. Arklex enables developers to create customizable workers and tools that collaborate seamlessly under intelligent orchestration.

## Key Features

* Multi-Agent Orchestration: Coordinate multiple specialized agents through graph-based task management
* Modular Architecture: Extensible workers and tools for diverse use cases
* Multi-LLM Support: OpenAI, Anthropic, Google Gemini, Mistral, and Hugging Face integration
* Built-in RAG & Database Workers: Vector search and database operations out of the box
* Comprehensive Evaluation: Automated testing with synthetic conversation generation
* Production-Ready API: Robust logging, error handling, and monitoring

## Quick Start

### Installation

```bash
pip install arklex
```

### Build Your First Agent in 3 Steps

1. Set up your environment

Create a `.env` file in the root directory with the following information:

```env
OPENAI_API_KEY=<your-openai-api-key>
# Add other API keys as needed
```

2. Create your agent configuration

```bash
python create.py --config ./examples/customer_service/customer_service_config.json --output-dir ./examples/customer_service
```

3. Start chatting

```bash
python run.py --input-dir ./examples/customer_service
```

That's it! Your AI agent is ready to handle customer service interactions.

## Documentation

* [Getting Started Guide](https://arklexai.github.io/Agent-First-Organization/docs/intro): Framework overview and core concepts
* [Tutorials](https://arklexai.github.io/Agent-First-Organization/docs/tutorials/intro): Step-by-step guides for common use cases
* [API Reference](https://www.arklex.ai/qa/open-source): Complete documentation and examples

## Use Cases

### Customer Service Agent

Build an intelligent customer service bot with contextual understanding and database integration.

Watch the tutorial: [Build a Customer Service Agent in 20 minutes](https://youtu.be/y1P2Ethvy0I)

### Supported Scenarios

* Customer Support: Automated help desk with knowledge base integration
* Booking Systems: Appointment scheduling with calendar management
* Data Analysis: Multi-step analytical workflows with visualization
* Content Generation: Collaborative writing and editing workflows

## Configuration

### Environment Setup

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

### Supported Models

| Provider      | Models                                      |
|---------------|---------------------------------------------|
| OpenAI        | gpt-4o, gpt-4o-mini                         |
| Anthropic     | claude-3-5-haiku, claude-3-5-sonnet         |
| Google        | gemini-2.0-flash                            |
| Mistral       | All Mistral models                          |
| Hugging Face  | Open source models                          |

## Core Commands

### Create Agent Workflow

```bash
python create.py \
  --config ./examples/customer_service/customer_service_config.json \
  --output-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o-mini
```

### Run Agent

```bash
python run.py \
  --input-dir ./examples/customer_service \
  --llm_provider openai \
  --model gpt-4o
```

### Start API Service

```bash
python model_api.py \
  --input-dir ./examples/customer_service
```

### Run Evaluation

```bash
python eval.py \
--model_api http://127.0.0.1:8000/eval/chat \
--config ./examples/customer_service/customer_service_config.json \
--documents_dir ./examples/customer_service \
--output-dir ./examples/customer_service
```

## Architecture

Arklex follows a centralized orchestration with distributed execution model:

* Task Graph: Defines workflow structure and dependencies
* Orchestrator: Manages agent coordination and task routing
* Workers: Specialized components (RAG, Database, API calls)
* Tools: Atomic functions for specific operations

## Evaluation & Testing

Arklex includes comprehensive evaluation tools:

* Synthetic Conversations: Generate realistic test scenarios
* Performance Metrics: Track response quality and task completion
* A/B Testing: Compare different agent configurations
* Error Analysis: Detailed logging and debugging support

## Production Features

### Logging & Monitoring

* Structured logging with request tracking
* Configurable log levels and rotation
* Request ID correlation across services
* Performance timing and metrics

### Error Handling

* Custom exception hierarchy
* Automatic retry mechanisms
* Graceful degradation
* Circuit breaker patterns

### API Service

* FastAPI-based REST endpoints
* Automatic OpenAPI documentation
* CORS support and security headers
* Health checks and status monitoring

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

* [Documentation](arklex.ai/docs)
* [GitHub Issues](https://github.com/arklexai/Agent-First-Organization/issues)
* [GitHub Discussions](https://github.com/arklexai/Agent-First-Organization/discussions)

### API Documentation

Once the application is running, you can access:

* API documentation at `/docs`
* Alternative API documentation at `/redoc`
