"""Centralized configuration management"""
import os
from typing import Optional


class Config:
    """Centralized configuration for the application"""

    # OpenAI API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    OPENAI_MODEL_SELECTOR: str = os.getenv("OPENAI_MODEL_SELECTOR", os.getenv("OPENAI_MODEL", "gpt-5-mini"))

    # Retry and Rate Limiting Configuration
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    OPENAI_RETRY_BASE_DELAY: float = float(os.getenv("OPENAI_RETRY_BASE_DELAY", "2.0"))
    OPENAI_RATE_LIMIT_DELAY: float = float(os.getenv("OPENAI_RATE_LIMIT_DELAY", "1.0"))
    
    # SSL Configuration (for corporate proxies)
    OPENAI_VERIFY_SSL: bool = os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false"
    
    # Workflow Validation Configuration
    ENABLE_PLAN_VALIDATION: bool = os.getenv("ENABLE_PLAN_VALIDATION", "true").lower() == "true"
    MAX_PLAN_REFINEMENT_RETRIES: int = int(os.getenv("MAX_PLAN_REFINEMENT_RETRIES", "3"))
    
    # Workflow Judge Configuration
    ENABLE_EXECUTION_JUDGE: bool = os.getenv("ENABLE_EXECUTION_JUDGE", "true").lower() == "false"
    MAX_JUDGE_REFINEMENT_RETRIES: int = int(os.getenv("MAX_JUDGE_REFINEMENT_RETRIES", "3"))
    JUDGE_CONFIDENCE_THRESHOLD: int = int(os.getenv("JUDGE_CONFIDENCE_THRESHOLD", "80"))

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key with validation"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return cls.OPENAI_API_KEY
