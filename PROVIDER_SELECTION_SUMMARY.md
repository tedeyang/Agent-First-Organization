# Model Provider Selection Refactoring Summary

## Overview

Successfully refactored the Arklex codebase to enable model provider selection via command line flags. The functionality was previously partially implemented but disabled or incomplete. This refactoring fully enables users to select different language model providers (OpenAI, Anthropic, Google Gemini, HuggingFace) when running the framework.

**Important:** OpenAI remains the default provider, ensuring backward compatibility. Users can override this by explicitly specifying a different provider via command line flags.

## Changes Made

### 1. Created Centralized Provider Utilities (`arklex/utils/provider_utils.py`)

- **`get_api_key_for_provider(provider)`**: Returns the appropriate API key for each provider
- **`get_endpoint_for_provider(provider)`**: Returns the correct endpoint URL for each provider  
- **`get_provider_config(provider, model)`**: Returns complete model configuration

**Supported Providers:**

- `openai`: Uses `OPENAI_API_KEY` environment variable
- `anthropic`: Uses `ANTHROPIC_API_KEY` environment variable
- `gemini`: Uses `GOOGLE_API_KEY` environment variable
- `huggingface`: Uses `HUGGINGFACE_API_KEY` environment variable

### 2. Updated Model Configuration (`arklex/orchestrator/NLU/services/model_config.py`)

- Enhanced `get_model_kwargs()` to properly handle API keys and endpoints
- Updated `get_model_instance()` to support provider-specific initialization
- Added proper handling for HuggingFace models (function vs class initialization)

### 3. Fixed Command Line Interface Files

#### `create.py`

- ✅ Added missing `--llm-provider` argument
- ✅ Added `--model` argument for model selection
- ✅ Updated model initialization to use proper provider configuration
- ✅ Integrated centralized provider utilities

#### `run.py`

- ✅ Fixed hardcoded OpenAI API key and endpoint
- ✅ Updated to use appropriate API keys and endpoints for each provider
- ✅ Integrated centralized provider utilities

#### `model_api.py`

- ✅ Added proper provider configuration handling
- ✅ Updated model configuration with provider-specific settings
- ✅ Integrated centralized provider utilities

#### `eval.py`

- ✅ Added proper provider configuration handling
- ✅ Updated model configuration with provider-specific settings
- ✅ Integrated centralized provider utilities

### 4. Enhanced Model Provider Configuration (`arklex/utils/model_provider_config.py`)

- Already had comprehensive provider mappings
- Supports OpenAI, Google Gemini, Anthropic, and HuggingFace
- Includes proper model classes and embedding configurations

## Usage Examples

### Task Graph Generation

```bash
# OpenAI
python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service --model gpt-4o --llm-provider openai

# Anthropic
python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service --model claude-3-5-haiku-20241022 --llm-provider anthropic

# Google Gemini
python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service --model gemini-2.0-flash --llm-provider gemini

# HuggingFace
python create.py --config ./examples/customer_service_config.json --output-dir ./examples/customer_service --model microsoft/Phi-3-mini-4k-instruct --llm-provider huggingface
```

### Running the Bot

```bash
# OpenAI
python run.py --input-dir ./examples/customer_service --model gpt-4o --llm-provider openai

# Anthropic
python run.py --input-dir ./examples/customer_service --model claude-3-5-haiku-20241022 --llm-provider anthropic

# Google Gemini
python run.py --input-dir ./examples/customer_service --model gemini-2.0-flash-lite --llm-provider gemini

# HuggingFace
python run.py --input-dir ./examples/customer_service --model microsoft/Phi-3-mini-4k-instruct --llm-provider huggingface
```

### API Server

```bash
# OpenAI
python model_api.py --input-dir ./examples/test --model gpt-4o --llm-provider openai --port 8000

# Anthropic
python model_api.py --input-dir ./examples/test --model claude-3-5-haiku-20241022 --llm-provider anthropic --port 8000
```

### Evaluation

```bash
# OpenAI
python eval.py --model_api openai --model gpt-4o --llm-provider openai --config config.json

# Anthropic
python eval.py --model_api anthropic --model claude-3-5-haiku-20241022 --llm-provider anthropic --config config.json
```

## Environment Variables Required

Set the appropriate API keys in your `.env` file:

```bash
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# For HuggingFace
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## Supported Models

### OpenAI

- `gpt-4o` (default)
- `gpt-4o-mini`
- `gpt-4.5-preview`

### Google Gemini

- `gemini-1.5-flash`
- `gemini-2.0-flash`
- `gemini-2.0-flash-lite`
- **Note:** Tool calling is only supported with `gemini-2.0-flash`

### Anthropic

- `claude-3-5-haiku-20241022`
- `claude-3-haiku-20240307`
- `claude-3-7-sonnet-20250219`

### HuggingFace

- `microsoft/Phi-3-mini-4k-instruct`
- **Note:** Tool calling is NOT supported for HuggingFace

## Key Benefits

1. **Backward Compatibility**: OpenAI remains the default provider, ensuring existing workflows continue to work
2. **Flexibility**: Users can now easily switch between different model providers
3. **Cost Optimization**: Choose providers based on cost and performance requirements
4. **Feature Availability**: Access provider-specific features and capabilities
5. **Consistency**: Unified interface across all providers
6. **Maintainability**: Centralized provider configuration management

## Testing

The refactoring has been tested to ensure:

- ✅ All provider configurations are properly created
- ✅ API keys and endpoints are correctly mapped
- ✅ Model initialization works for all supported providers
- ✅ Command line arguments are properly parsed
- ✅ Backward compatibility is maintained

## Future Enhancements

1. **Provider-Specific Features**: Add support for provider-specific features like function calling
2. **Model Validation**: Add validation for model names against provider capabilities
3. **Cost Tracking**: Implement cost tracking for different providers
4. **Performance Metrics**: Add provider-specific performance monitoring
5. **Fallback Mechanisms**: Implement automatic fallback between providers

## Conclusion

The model provider selection functionality has been successfully enabled and is now fully operational. Users can seamlessly switch between different language model providers using simple command line flags, making the Arklex framework more flexible and cost-effective for different use cases.
