"""LLM client abstraction for structured extraction."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None


@dataclass
class LLMMessage:
    """Represents a message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    raw_response: Any
    model: str
    usage: Dict[str, int]
    finish_reason: str


class LLMClient(ABC):
    """Abstract base class for LLM clients supporting structured extraction."""

    @abstractmethod
    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            response_format: Optional JSON schema for structured output

        Returns:
            Standardized LLMResponse
        """
        pass

    def extract_json(self, response: LLMResponse) -> Dict[str, Any]:
        """Extract and parse JSON from response."""
        content = response.content.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}\nContent: {content}")


class AnthropicClient(LLMClient):
    """Anthropic Claude client with structured output support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY or pass api_key)")

        self.model = model
        self._client = None

    def _get_client(self) -> Any:
        """Lazy load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                ) from e
        return self._client

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Generate completion using Claude."""
        client = self._get_client()

        # Separate system message from conversation
        system_msg = None
        conv_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conv_messages.append({"role": msg.role, "content": msg.content})

        # If structured output requested, add JSON instruction to system prompt
        if response_format is not None:
            schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(response_format, indent=2)}"
            system_msg = (system_msg or "") + schema_instruction

        kwargs = {
            "model": self.model,
            "messages": conv_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if system_msg:
            kwargs["system"] = system_msg

        response = client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason or "stop",
        )


class OpenAIClient(LLMClient):
    """OpenAI client with structured output support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY or pass api_key)")

        self.model = model
        self._client = None

    def _get_client(self) -> Any:
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError("openai package required. Install with: pip install openai") from e
        return self._client

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Generate completion using GPT-4."""
        client = self._get_client()

        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Use native structured output if schema provided
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": response_format}

        response = client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            raw_response=response,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            finish_reason=response.choices[0].finish_reason or "stop",
        )


class MockLLMClient(LLMClient):
    """Mock client for testing without API calls."""

    def __init__(self, mock_responses: Optional[List[str]] = None) -> None:
        """Initialize mock client with canned responses."""
        self.mock_responses = mock_responses or []
        self.call_count = 0

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Return mock response."""
        if self.call_count < len(self.mock_responses):
            content = self.mock_responses[self.call_count]
        else:
            content = '{"status": "mock", "data": []}'

        self.call_count += 1

        return LLMResponse(
            content=content,
            raw_response=None,
            model="mock-model",
            usage={"input_tokens": 100, "output_tokens": 50},
            finish_reason="stop",
        )


class OllamaClient(LLMClient):
    """Ollama client targeting a locally hosted model."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> None:
        if requests is None:  # pragma: no cover - optional dependency
            raise ImportError("requests package required for Ollama client. Install with: pip install requests")
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.headers = headers or {}
        # Allow overriding via env; default to generous 10 minute read timeout for heavier prompts.
        env_timeout = os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS")
        self.timeout = timeout or (int(env_timeout) if env_timeout else 600)

    def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        else:
            payload["options"]["num_predict"] = -1

        if response_format is not None:
            payload["response_format"] = response_format

        response = requests.post(
            url,
            json=payload,
            headers=self.headers,
            timeout=(30, self.timeout),
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", data.get("response", ""))

        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }

        return LLMResponse(
            content=content,
            raw_response=data,
            model=data.get("model", self.model),
            usage=usage,
            finish_reason="stop" if data.get("done", True) else "unknown",
        )


def create_llm_client(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """Factory function to create LLM clients.

    Args:
        provider: "anthropic", "openai", or "mock"
        api_key: Optional API key override
        model: Optional model name override

    Returns:
        Configured LLM client
    """
    provider_lower = provider.lower()

    if provider_lower == "anthropic":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return AnthropicClient(**kwargs)

    elif provider_lower == "openai":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return OpenAIClient(**kwargs)

    elif provider_lower == "mock":
        return MockLLMClient()

    elif provider_lower == "ollama":
        if not model:
            raise ValueError("Model name required for Ollama provider")
        base_url = os.getenv("OLLAMA_BASE_URL")
        return OllamaClient(model=model, base_url=base_url)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'mock'")


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "MockLLMClient",
    "OllamaClient",
    "create_llm_client",
]
