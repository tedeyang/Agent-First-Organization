"""Provider utility functions for the Arklex framework.

This module provides utility functions for handling different model providers,
including API key management and endpoint configuration. It centralizes
provider-specific logic to ensure consistency across the framework.

Key Components:
1. get_api_key_for_provider: Get the appropriate API key for a provider
2. get_endpoint_for_provider: Get the appropriate endpoint for a provider
3. Provider-specific configuration mappings

Usage:
    from arklex.utils.provider_utils import get_api_key_for_provider, get_endpoint_for_provider

    # Get API key for a specific provider
    api_key = get_api_key_for_provider("openai")

    # Get endpoint for a specific provider
    endpoint = get_endpoint_for_provider("anthropic")
"""

import os
from typing import Any


def get_api_key_for_provider(provider: str) -> str:
    """Get the appropriate API key for the specified provider.

    Args:
        provider (str): The model provider name

    Returns:
        str: The API key for the provider
    """
    if provider is None:
        raise TypeError("Provider cannot be None")

    provider_api_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",  # Google API key is used for Gemini
        "huggingface": "HUGGINGFACE_API_KEY",
    }

    env_key = provider_api_keys.get(provider, "OPENAI_API_KEY")
    return os.getenv(env_key, "")


def get_endpoint_for_provider(provider: str) -> str:
    """Get the appropriate endpoint for the specified provider.

    Args:
        provider (str): The model provider name

    Returns:
        str: The endpoint URL for the provider
    """
    if provider is None:
        raise TypeError("Provider cannot be None")

    provider_endpoints = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "gemini": "https://generativelanguage.googleapis.com",
        "huggingface": "https://api-inference.huggingface.co",
    }

    return provider_endpoints.get(provider, "https://api.openai.com/v1")


def get_provider_config(provider: str, model: str) -> dict[str, Any]:
    """Get complete configuration for a specific provider and model.

    Args:
        provider (str): The model provider name
        model (str): The model name/identifier

    Returns:
        Dict[str, Any]: Complete model configuration including API key and endpoint
    """
    if provider is None:
        raise TypeError("Provider cannot be None")
    if model is None:
        raise TypeError("Model cannot be None")

    return {
        "model_name": model,
        "model_type_or_path": model,
        "llm_provider": provider,
        "api_key": get_api_key_for_provider(provider),
        "endpoint": get_endpoint_for_provider(provider),
    }
