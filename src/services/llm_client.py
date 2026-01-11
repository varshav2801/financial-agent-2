"""Centralized LLM client service with built-in retry and async support"""
from typing import Optional, TYPE_CHECKING
from contextlib import asynccontextmanager
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from src.config import Config
from src.logger import get_logger

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = get_logger(__name__)


class LLMClientService:
    """
    Centralized LLM client service using OpenAI's built-in retry mechanisms.
    
    Uses OpenAI's native retry configuration which handles:
    - Exponential backoff for rate limits
    - Automatic retry on transient errors
    - Proper connection pooling via httpx
    """

    _instance: Optional["LLMClientService"] = None
    _client: Optional[AsyncOpenAI] = None

    def __init__(self) -> None:
        """Initialize LLM client service"""
        Config.validate()
        
        # Configure httpx client with SSL settings and timeout
        # Set timeout: 30s connect, 60s read (should be enough for planning)
        timeout = httpx.Timeout(30.0, read=60.0)
        
        if not Config.OPENAI_VERIFY_SSL:
            logger.warning("SSL verification disabled - use only in trusted networks")
            http_client = httpx.AsyncClient(verify=False, timeout=timeout)
        else:
            http_client = httpx.AsyncClient(timeout=timeout)
        
        self._client = AsyncOpenAI(
            api_key=Config.OPENAI_API_KEY,
            max_retries=Config.OPENAI_MAX_RETRIES,
            http_client=http_client,
        )
        self.model = Config.OPENAI_MODEL

    @classmethod
    def get_instance(cls) -> "LLMClientService":
        """Get singleton instance of LLM client service"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (mainly for testing)"""
        if cls._instance and cls._instance._client:
            # Close the client if it exists
            # Note: AsyncOpenAI doesn't have explicit close, but we can reset
            cls._instance._client = None
        cls._instance = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client instance"""
        if self._client is None:
            raise RuntimeError("LLM client not initialized. Call get_instance() first.")
        return self._client

    async def parse_completion(
        self,
        messages: list[dict[str, str]],
        response_format: type["BaseModel"],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatCompletion:
        """
        Call OpenAI API with structured output parsing.
        
        Uses OpenAI's built-in retry mechanism configured at client initialization.
        No manual retry logic needed - handled by the SDK.
        
        Args:
            messages: Chat messages for the completion
            response_format: Pydantic model for structured output
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            
        Returns:
            ChatCompletion response with parsed structured output
        """
        model_name = model or self.model

        logger.debug(
            "Calling OpenAI API with model=%s, max_retries=%s, temperature=%s",
            model_name,
            Config.OPENAI_MAX_RETRIES,
            temperature,
        )

        # Build kwargs for API call
        api_kwargs = {
            "model": model_name,
            "messages": messages,
            "response_format": response_format,
        }

        # Only add temperature if specified (let OpenAI use default otherwise)
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        response = await self.client.beta.chat.completions.parse(**api_kwargs)

        return response

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatCompletion:
        """
        Call OpenAI API for standard chat completion.
        
        Args:
            messages: Chat messages for the completion
            model: Model to use (defaults to configured model)
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            
        Returns:
            ChatCompletion response
        """
        model_name = model or self.model

        logger.debug(
            "Calling OpenAI API (standard) with model=%s, max_retries=%s, temperature=%s",
            model_name,
            Config.OPENAI_MAX_RETRIES,
            temperature,
        )

        # Build kwargs for API call
        api_kwargs = {
            "model": model_name,
            "messages": messages,
        }

        # Only add temperature if specified
        if temperature is not None:
            api_kwargs["temperature"] = temperature

        response = await self.client.chat.completions.create(**api_kwargs)

        return response

    @asynccontextmanager
    async def session(self):
        """
        Async context manager for explicit session management.
        
        Usage:
            async with llm_client.session() as client:
                response = await client.beta.chat.completions.parse(...)
        """
        try:
            yield self.client
        finally:
            # OpenAI client handles cleanup automatically,
            # but this provides explicit session boundary
            pass


def get_llm_client() -> LLMClientService:
    """
    Get the global LLM client service instance.
    
    Returns:
        LLMClientService singleton instance
    """
    return LLMClientService.get_instance()
