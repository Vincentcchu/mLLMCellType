"""Main annotation module for LLMCellType."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .config import get_default_model
from .functions import PROVIDER_FUNCTIONS
from .logger import setup_logging, write_log
from .prompts import create_prompt
from .url_utils import resolve_provider_base_url
from .utils import (
    create_cache_key,
    format_results,
    load_api_key,
    load_from_cache,
    parse_marker_genes,
    save_to_cache,
)


def _extract_content_and_usage(raw_result):
    """Normalize provider response into content list/str plus usage metadata."""

    usage = None
    content = raw_result

    if isinstance(raw_result, dict):
        usage = raw_result.get("usage")

        for key in ("lines", "content", "results", "data", "choices"):
            if key in raw_result and raw_result[key] is not None:
                content = raw_result[key]
                break

    return content, usage


def annotate_clusters(
    marker_genes: dict[str, list[str]] | pd.DataFrame,
    species: str,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    tissue: str | None = None,
    additional_context: str | None = None,
    prompt_template: str | None = None,
    use_cache: bool = True,
    cache_dir: str | None = None,
    log_dir: str | None = None,
    log_level: str = "INFO",
    base_urls: str | dict[str, str] | None = None,
    track_usage: bool = False,
) -> dict[str, str] | tuple[dict[str, str], dict[str, Any]]:
    """Annotate cell clusters using LLM.

    Args:
        marker_genes: Dictionary mapping cluster names to lists of marker genes,
                     or DataFrame with 'cluster' and 'gene' columns
        species: Species name (e.g., 'human', 'mouse')
        provider: LLM provider (e.g., 'openai', 'anthropic')
        model: Model name (e.g., 'gpt-5', 'claude-sonnet-4-5-20250929')
        api_key: API key for the provider
        tissue: Tissue name (e.g., 'brain', 'liver')
        additional_context: Additional context to include in the prompt
        prompt_template: Custom prompt template
        use_cache: Whether to use cache
        cache_dir: Directory to store cache files
        log_dir: Directory to store log files
        log_level: Logging level
        base_urls: Custom base URLs for API endpoints. Can be:
                  - str: Single URL applied to all providers
                  - dict: Provider-specific URLs (e.g., {'openai': 'https://proxy.com/v1'})
        track_usage: When True, also return timing and usage metadata from the provider

    Returns:
        Dict[str, str]: Dictionary mapping cluster names to annotations. If ``track_usage``
        is True, returns a tuple of (annotations, metadata).

    """
    # Setup logging
    setup_logging(log_dir=log_dir, log_level=log_level)
    write_log(f"Starting annotation with provider: {provider}")

    # Parse marker genes if DataFrame
    if isinstance(marker_genes, pd.DataFrame):
        marker_genes = parse_marker_genes(marker_genes)

    # Get clusters
    clusters = list(marker_genes.keys())
    write_log(f"Found {len(clusters)} clusters")

    # Set default model based on provider
    if not model:
        model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {model}")

    # Get API key if not provided
    if not api_key:
        api_key = load_api_key(provider)
        if not api_key:
            error_msg = f"API key not found for provider: {provider}"
            write_log(error_msg, level="error")
            raise ValueError(error_msg)

    # Create prompt
    prompt = create_prompt(
        marker_genes=marker_genes,
        species=species,
        tissue=tissue,
        additional_context=additional_context,
        prompt_template=prompt_template,
    )

    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider)
        cached_results = load_from_cache(cache_key, cache_dir)
        if cached_results:
            write_log("Using cached results")
            return format_results(cached_results, clusters)

    # Resolve base URL
    base_url = resolve_provider_base_url(provider, base_urls)

    # Get provider function
    provider_func = PROVIDER_FUNCTIONS.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Process request
    try:
        write_log(f"Processing request with {provider} using model {model}")
        start_time = time.time()

        # Call provider function with base_url
        raw_results = provider_func(prompt, model, api_key, base_url)
        content, usage = _extract_content_and_usage(raw_results)

        # Normalize content to a list of lines
        if isinstance(content, str):
            normalized_results = content.splitlines()
        elif isinstance(content, list):
            normalized_results = content
        else:
            normalized_results = [str(content)] if content is not None else []

        end_time = time.time()
        duration = end_time - start_time
        write_log(f"Request processed in {duration:.2f} seconds")

        # Save to cache (only store normalized content to keep cache format stable)
        if use_cache:
            save_to_cache(cache_key, normalized_results, cache_dir)

        # Format results
        formatted = format_results(normalized_results, clusters)

        if track_usage:
            metadata = {
                "provider": provider,
                "model": model,
                "duration_seconds": duration,
                "usage": usage,
            }
            return formatted, metadata

        return formatted

    except Exception as e:
        error_msg = f"Error during annotation: {e!s}"
        write_log(error_msg, level="error")
        raise


def get_model_response(
    prompt: str,
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    cache_dir: str | None = None,
    base_url: str | None = None,
) -> str:
    """Get response from a model for a given prompt.

    Args:
        prompt: The prompt to send to the model
        provider: The provider name (e.g., 'openai', 'anthropic')
        model: The model name. If None, uses the default model for the provider.
        api_key: The API key for the provider. If None, loads from environment.
        use_cache: Whether to use cache
        cache_dir: The cache directory
        base_url: Optional custom base URL

    Returns:
        str: The model response

    """

    # Check if provider is valid
    if not provider:
        raise ValueError("Provider name is required")

    # Set default model if not provided
    if not model:
        model = get_default_model(provider)
        write_log(f"Using default model for {provider}: {model}")

    # Get API key if not provided
    if not api_key:
        api_key = load_api_key(provider)
        if not api_key:
            error_msg = f"API key not found for provider: {provider}"
            write_log(error_msg, level="error")
            raise ValueError(error_msg)

    # Check cache
    if use_cache:
        cache_key = create_cache_key(prompt, model, provider)
        cached_result = load_from_cache(cache_key, cache_dir)
        if cached_result:
            write_log(f"Using cached result for {model}")
            if isinstance(cached_result, list):
                return "\n".join(cached_result)
            return cached_result

    # Get provider function
    provider_func = PROVIDER_FUNCTIONS.get(provider.lower())
    if not provider_func:
        error_msg = f"Unknown provider: {provider}"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Call provider function
    try:
        write_log(f"Requesting response from {provider} ({model})")
        raw_result = provider_func(prompt, model, api_key, base_url)
        content, _usage = _extract_content_and_usage(raw_result)

        if isinstance(content, list):
            normalized = content
        elif isinstance(content, str):
            normalized = content.splitlines()
        else:
            normalized = [str(content)] if content is not None else []

        # Save to cache
        if use_cache:
            save_to_cache(cache_key, normalized, cache_dir)

        return "\n".join(normalized)
    except Exception as e:
        error_msg = f"Error getting model response: {e!s}"
        write_log(error_msg, level="error")
        raise
