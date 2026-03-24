"""OpenRouter provider module for LLMCellType."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from ..logger import write_log
from ..url_utils import get_default_api_url, validate_base_url


def get_openrouter_balance(api_key: str) -> float | None:
    """Get the current credit balance from OpenRouter API.

    Args:
        api_key: OpenRouter API key

    Returns:
        float | None: Current credit balance in USD, or None if request fails
    """
    if not api_key:
        write_log("OpenRouter API key is missing", level="error")
        return None

    try:
        url = "https://openrouter.ai/api/v1/auth/key"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        
        response = requests.get(url=url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            write_log(f"Failed to get OpenRouter balance: {response.status_code}", level="warning")
            return None
            
        data = response.json()
        # OpenRouter returns credit limit and usage in the response
        # Balance = limit - usage
        
        # Check if the response structure is as expected
        if "data" not in data:
            write_log(f"Unexpected OpenRouter API response structure: {data.keys()}", level="warning")
            return None
            
        limit = data.get("data", {}).get("limit")
        usage = data.get("data", {}).get("usage")
        
        # Handle None values - if either is None, we can't calculate balance
        if limit is None or usage is None:
            write_log(f"OpenRouter balance check incomplete (limit: {limit}, usage: {usage}). Response data keys: {data.get('data', {}).keys()}", level="warning")
            return None
        
        balance = limit - usage
        
        write_log(f"OpenRouter balance check: ${balance:.6f} (limit: ${limit:.2f}, usage: ${usage:.6f})")
        return balance
        
    except Exception as e:
        write_log(f"Error checking OpenRouter balance: {e!s}", level="warning")
        return None


def process_openrouter(
    prompt: str, model: str, api_key: str, base_url: str | None = None
) -> dict[str, Any]:
    """Process request using OpenRouter API, which provides access to various LLM models.

    Args:
        prompt: The prompt to send to the API
        model: The model name (e.g., 'openai/gpt-5', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-opus-4.1')
        api_key: OpenRouter API key
        base_url: Optional custom base URL

    Returns:
        dict[str, Any]: Dict containing cleaned lines plus usage metadata

    """
    write_log(f"Starting OpenRouter API request with model: {model}")

    # Check if API key is provided and not empty
    if not api_key:
        error_msg = "OpenRouter API key is missing or empty"
        write_log(error_msg, level="error")
        raise ValueError(error_msg)

    # Use custom URL or default URL
    if base_url:
        if not validate_base_url(base_url):
            raise ValueError(f"Invalid base URL: {base_url}")
        url = base_url
        write_log(f"Using custom base URL: {url}")
    else:
        url = get_default_api_url("openrouter")
        write_log(f"Using default URL: {url}")

    write_log(f"Using model: {model}")

    # Ensure model ID is in the correct format for OpenRouter (provider/model)
    if "/" not in model:
        write_log(
            f"Model ID '{model}' may not be in the correct format for OpenRouter. "
            "Expected format: 'provider/model'",
            level="warning",
        )

    # Prepare the request body
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    write_log("Sending API request...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/cafferychen777/mLLMCelltype",  # Optional for rankings
        "X-Title": "mLLMCelltype",  # Optional for rankings
    }

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(body), timeout=30)

            # Check for errors
            if response.status_code != 200:
                error_message = response.json()
                write_log(
                    f"OpenRouter API request failed: {error_message.get('error', {}).get('message', 'Unknown error')}",
                    level="error",
                )

                # If rate limited, wait and retry
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    write_log(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

            # Parse the response
            content = response.json()
            res = content["choices"][0]["message"]["content"].strip().split("\n")
            write_log(f"Got response with {len(res)} lines")
            write_log(f"Raw response from OpenRouter:\n{res}", level="debug")

            usage = content.get("usage") or {}

            # Some deployments may return cost information in headers
            header_usage = response.headers.get("x-usage")
            if header_usage and not usage:
                try:
                    usage = json.loads(header_usage)
                except (TypeError, json.JSONDecodeError):
                    write_log("Unable to parse x-usage header", level="debug")

            # Clean up results (remove commas at the end of lines)
            cleaned_lines = [line.rstrip(",") for line in res]

            return {
                "lines": cleaned_lines,
                "usage": usage if usage else None,
                "raw": content,
            }

        except Exception as e:
            write_log(f"Error during API call (attempt {attempt + 1}/{max_retries}): {e!s}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                write_log(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise
