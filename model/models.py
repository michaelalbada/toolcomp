import os
import json
import logging
import litellm
from utils.keystore import auth_litellm, auth_azure_openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tools.helper import get_all_tools_mapping
from openai import AzureOpenAI
import openai
import OpenSSL
import requests
import time
import random

logger = logging.getLogger(__name__)


class GenerationWrapper:
    """Base wrapper for LLM generation."""

    def __init__(self, model, sampling_params):
        self.model = model
        self.sampling_params = sampling_params
        self.tool_mapping = get_all_tools_mapping()
    
    def generate(self, prompt, tool_list=[], historical_date=None):
        """Generate text with the model."""
        pass

class AzureOpenAIWrapper(GenerationWrapper):
    """
    Azure OpenAI implementation with tool use support.

    This wrapper provides:
    - Automatic retry logic with exponential backoff for rate limits
    - Support for both API key and Azure AD token-based authentication
    - Tool/function calling with automatic tool execution loop
    - Comprehensive error handling and logging

    Environment Variables:
        AZURE_OPENAI_API_KEY: API key (if using key auth)
        AZURE_OPENAI_ENDPOINT: Azure endpoint URL
        AZURE_OPENAI_DEPLOYMENT: Deployment name
        AZURE_OPENAI_API_VERSION: API version (default: 2025-01-01-preview)
        AZURE_OPENAI_AUTH_MODE: "key" or "token" (auto-detected if not set)

    Example:
        >>> wrapper = AzureOpenAIWrapper(
        ...     model="gpt-4",
        ...     sampling_params={"temperature": 0.7, "max_tokens": 1000}
        ... )
        >>> text, history = wrapper.generate(
        ...     prompt=[{"role": "user", "content": "What's the weather?"}],
        ...     tool_list=["weather"]
        ... )
    """

    def __init__(self, model, sampling_params):
        super().__init__(model, sampling_params)

        api_key, api_base, deployment_from_env = auth_azure_openai()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        # Deployment can come from env or be extracted from model name
        self.deployment = deployment_from_env or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise ValueError("AzureOpenAIWrapper requires a deployment name (use model='azure/<deployment>' or model='<deployment>').")

        auth_mode = os.getenv("AZURE_OPENAI_AUTH_MODE", "key" if api_key else "token")

        if auth_mode == "token":
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAI(
                azure_endpoint=api_base,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
        else:
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_base,
                api_version=api_version,
            )

    def _parse_functions(self, response_message):
        """Return a list of tool call dicts normalized to {'id','type','function':{'name','arguments'}}."""
        # Our _hit_azure returns a plain dict with tool_calls already normalized.
        if isinstance(response_message, dict):
            return response_message.get("tool_calls") or []
        # Fallback if an object ever slips through
        tcalls = getattr(response_message, "tool_calls", None)
        if not tcalls:
            return []
        out = []
        for tc in tcalls:
            func = getattr(tc, "function", None)
            out.append({
                "id": getattr(tc, "id", None),
                "type": "function",
                "function": {
                    "name": getattr(func, "name", None) if func else None,
                    "arguments": getattr(func, "arguments", None) if func else None,
                },
            })
        return out

    def _call_tools(self, messages, tool_calls, tool_list, historical_date=None):
        """Execute tool calls and append tool results as tool-role messages."""
        available_functions = {tool: self.tool_mapping[tool].parse_and_hit_tool for tool in tool_list}
        for tc in tool_calls:
            # Support both dict and object forms
            if isinstance(tc, dict):
                function_name = tc.get("function", {}).get("name")
                function_args = tc.get("function", {}).get("arguments")
                call_id = tc.get("id")
            else:
                function_name = getattr(getattr(tc, "function", None), "name", None)
                function_args = getattr(getattr(tc, "function", None), "arguments", None)
                call_id = getattr(tc, "id", None)

            if function_name not in available_functions:
                messages.append({
                    "tool_call_id": call_id,
                    "role": "tool",
                    "name": function_name or "unknown_function",
                    "content": f"The tool {function_name} is not available.",
                })
                continue

            function_to_call = available_functions[function_name]
            function_response = function_to_call(function_args, historical_date)

            messages.append({
                "tool_call_id": call_id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        return messages

    def _hit_azure(self, messages, tools=None, tool_choice="auto"):
        """Single call to Azure Chat Completions with retries/backoff."""
        # Only pass supported sampling params to the SDK call
        allowed = {
            "temperature", "top_p", "max_tokens",
            "presence_penalty", "frequency_penalty",
            "stop", "seed", "logit_bias", "n", "response_format", "user",
        }
        kwargs = {k: v for k, v in self.sampling_params.items() if k in allowed and v is not None}

        max_retries_rate_limit = 15
        max_retries_other = 3
        retry_count_rate_limit = 0
        retry_count_other = 0
        base_delay = 15.0

        while retry_count_rate_limit < max_retries_rate_limit or retry_count_other < max_retries_other:
            try:
                resp = self.client.chat.completions.create(
                    model=self.deployment,  # Azure expects deployment name here
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice if tools else None,
                    **kwargs,
                )
                msg = resp.choices[0].message
                # Normalize assistant message to a plain dict
                tool_calls_norm = []
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        func = getattr(tc, "function", None)
                        tool_calls_norm.append({
                            "id": getattr(tc, "id", None),
                            "type": "function",
                            "function": {
                                "name": getattr(func, "name", None) if func else None,
                                "arguments": getattr(func, "arguments", None) if func else None,
                            },
                        })

                return {
                    "role": getattr(msg, "role", "assistant"),
                    "content": getattr(msg, "content", None),
                    "tool_calls": tool_calls_norm or None,
                }
            except Exception as e:
                error_type = type(e).__name__

                # Handle rate limit errors with exponential backoff
                if "rate" in str(e).lower() or "429" in str(e):
                    retry_count_rate_limit += 1
                    if retry_count_rate_limit >= max_retries_rate_limit:
                        logger.error(f"Azure OpenAI {error_type}: Rate limit max retries exceeded")
                        raise
                    delay = base_delay * (2 ** (retry_count_rate_limit - 1))
                    logger.warning(
                        f"Azure OpenAI rate limit hit, retrying in {delay:.1f}s "
                        f"({retry_count_rate_limit}/{max_retries_rate_limit})"
                    )
                    time.sleep(delay)
                else:
                    retry_count_other += 1
                    if retry_count_other >= max_retries_other:
                        logger.error(f"Azure OpenAI {error_type}: {str(e)}")
                        raise
                    logger.warning(f"Azure OpenAI request failed ({error_type}), retrying in 2s...")
                    time.sleep(2)

        raise Exception("Max retries exceeded for Azure OpenAI chat completion")

    def _generate(self, messages, tool_list=None, historical_date=None):
        """Generate a response with (optional) tool use."""
        tool_list = tool_list or []
        tools = [self.tool_mapping[t].get_gpt_spec() for t in tool_list if t in self.tool_mapping]

        response_message = self._hit_azure(messages, tools, tool_choice="auto")
        if not tools:
            return response_message.get("content"), messages

        tool_calls = self._parse_functions(response_message)
        messages.append(response_message)

        max_steps = 100
        steps = 0
        while tool_calls and steps < max_steps:
            messages = self._call_tools(messages, tool_calls, tool_list, historical_date)
            response_message = self._hit_azure(messages, tools, tool_choice="auto")
            tool_calls = self._parse_functions(response_message)
            messages.append(response_message)
            steps += 1

        return messages[-1].get("content"), messages

    def generate(self, prompt, tool_list=None, historical_date=None):
        """Generate a response with tool use, with retries."""
        tool_list = tool_list or []
        max_retries = 5
        last_err = None
        while max_retries > 0:
            try:
                messages = prompt.copy()
                final_text, full_history = self._generate(messages, tool_list, historical_date)
                return final_text, full_history
            except Exception as e:
                last_err = e
                max_retries -= 1
                if max_retries == 0:
                    logger.error(f"Error in generation (Azure): {e}")
                    return str(e), prompt
                logger.warning(f"Generation attempt failed, retrying ({max_retries} attempts left)...")
                time.sleep(2)

class LiteLLMWrapper(GenerationWrapper):
    """LiteLLM implementation for tool use."""
    
    def __init__(self, model, sampling_params):
        super().__init__(model, sampling_params)

        api_key, api_base = auth_litellm()
        litellm.api_key = api_key
        litellm.api_base = api_base
    
    def _parse_functions(self, response_message):
        """Parse function/tool calls from response."""
        tool_calls = (
            response_message.tool_calls if hasattr(response_message, "tool_calls") else None
        )
        return tool_calls
    
    def _hit_litellm(self, messages, tools=None, tool_choice='auto'):
        """Make a request to LiteLLM API."""
        max_retries_rate_limit = 15
        max_retries_other = 3
        base_delay = 15  # starting delay in seconds
        retry_count_rate_limit = 0
        retry_count_other = 0
        
        while retry_count_rate_limit < max_retries_rate_limit or retry_count_other < max_retries_other:
            try:
                litellm.drop_params = True
                response = litellm.completion(
                    messages=messages,
                    tools=tools if tools else None,
                    **self.sampling_params
                )
                return response.choices[0].message
            except Exception as e:
                error = e
                
                # Only apply exponential backoff for rate limit errors
                if "litellm.RateLimitError" in str(e):
                    retry_count_rate_limit += 1
                    if retry_count_rate_limit >= max_retries_rate_limit:
                        break
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** (retry_count_rate_limit - 1))  # exponential increase
                    delay = delay * (0.5 + random.random())  # add jitter (50-150% of delay)
                    if "litellm.RateLimitError" in str(e):
                        logger.warning(f"LiteLLM rate limit error, retrying with backoff in {delay:.2f}s (attempt {retry_count_rate_limit}/{max_retries_rate_limit})")

                else:
                    logger.warning(f"LiteLLM error: {e}")
                    retry_count_other += 1
                    if retry_count_other >= max_retries_other:
                        break
                    # For other errors, use a simple fixed delay
                    delay = 15
                    logger.warning(f"LiteLLM request failed, retrying in {delay}s (attempt {retry_count_other}/{max_retries_other})")
                
                time.sleep(delay)
        
        raise Exception(f"Max retries ({max_retries_rate_limit}) exceeded: {error}")
    
    def _call_tools(self, messages, tool_calls, tool_list, historical_date=None):
        """Call the tools and add responses to messages."""
        available_functions = {tool: self.tool_mapping[tool].parse_and_hit_tool for tool in tool_list}
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name not in available_functions:
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"The tool you are trying to call {function_name} is not available.",
                    }
                )
                continue
            
            function_to_call = available_functions[function_name]
            function_args = tool_call.function.arguments
            function_response = function_to_call(function_args, historical_date)
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        return messages

    def _generate(self, messages, tool_list=[], historical_date=None):
        """Generate a response with tool use."""
        tools = [self.tool_mapping[tool].get_gpt_spec() for tool in tool_list if tool in tool_list]
        response_message = self._hit_litellm(messages, tools, tool_choice='auto')
        
        if not tools:
            return response_message['content'], messages
        
        tool_calls = self._parse_functions(response_message)
        
        messages.append(response_message)  # extend conversation with assistant's reply
        max_steps = 100  # limit the number of tool call iterations
        steps = 0
        
        while tool_calls and steps < max_steps:
            messages = self._call_tools(messages, tool_calls, tool_list, historical_date)
            response_message = self._hit_litellm(messages, tools, tool_choice='auto')
            tool_calls = self._parse_functions(response_message)
            
            messages.append(response_message)
            steps += 1

        messages = [
            message.dict() if not isinstance(message, dict) else message for message in messages
        ]
        
        return messages[-1]["content"], messages

    def generate(self, prompt, tool_list=[], historical_date=None):
        """Generate a response with tool use, with retries."""
        max_retries = 5
        while True:
            try:
                messages = prompt.copy()
                final_output_text, full_message_history = self._generate(messages, tool_list, historical_date)
                break
            except Exception as e:
                max_retries -= 1
                if max_retries == 0:
                    logger.error(f"Error in generation (LiteLLM): {e}")
                    return str(e), prompt
                logger.warning(f"Generation attempt failed, retrying ({max_retries} attempts left)...")
        
        return final_output_text, full_message_history
