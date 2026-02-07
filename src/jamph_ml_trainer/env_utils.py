"""Environment utilities inspired by HuggingFace Accelerate best practices."""

import os
from functools import lru_cache
from typing import Optional


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable value.
    Uses os.environ.get instead of os.getenv (Accelerate best practice).
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def get_int_from_env(env_keys: list[str], default: int) -> int:
    """
    Returns the first positive int value found in the env_keys list or the default.
    
    Args:
        env_keys: List of environment variable names to check
        default: Default value if none found
        
    Returns:
        First positive integer found or default
        
    Example:
        >>> get_int_from_env(["MAX_WORKERS", "NUM_THREADS"], 4)
        4  # if neither is set
    """
    for key in env_keys:
        val = int(os.environ.get(key, -1))
        if val >= 0:
            return val
    return default


def parse_flag_from_env(key: str, default: bool = False) -> bool:
    """
    Parse boolean flag from environment variable.
    
    Args:
        key: Environment variable name
        default: Default boolean value
        
    Returns:
        Boolean value parsed from environment
        
    Example:
        >>> parse_flag_from_env("ENABLE_ASYNC_UPLOAD", False)
        True  # if ENABLE_ASYNC_UPLOAD=1 or true or yes
    """
    value = os.environ.get(key, str(default)).lower()
    return value in ("1", "true", "yes", "on")


@lru_cache(maxsize=None)
def get_model_prefix() -> str:
    """
    Get model prefix from environment (cached for performance).
    
    Returns:
        Model prefix (default: 'jamph')
    """
    return os.environ.get("MODEL_PREFIX", "jamph")


@lru_cache(maxsize=None)
def get_hf_username() -> str:
    """
    Get HuggingFace username from environment (cached for performance).
    Checks multiple environment variables in order.
    
    Returns:
        HuggingFace username or 'unknown'
    """
    return (
        os.environ.get("HF_USERNAME")
        or os.environ.get("OLLAMA_USERNAME")
        or "unknown"
    )


@lru_cache(maxsize=None)
def get_git_username() -> str:
    """
    Get git username from environment (cached for performance).
    Checks multiple environment variables in order.
    
    Returns:
        Git username or 'unknown'
    """
    return (
        os.environ.get("GITHUB_HANDLE")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "unknown"
    )


@lru_cache(maxsize=None)
def get_team_name() -> str:
    """
    Get team name from environment (cached for performance).
    
    Returns:
        Team name (default: 'Unknown')
    """
    return os.environ.get("TEAM", "Unknown")


@lru_cache(maxsize=None)
def get_organization() -> str:
    """
    Get organization name from environment (cached for performance).
    
    Returns:
        Organization name (default: 'Unknown')
    """
    return os.environ.get("ORGANIZATION", "Unknown")


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment.
    Does NOT cache for security reasons.
    
    Returns:
        HuggingFace token or None
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def get_ollama_token() -> Optional[str]:
    """
    Get Ollama token from environment.
    Does NOT cache for security reasons.
    
    Returns:
        Ollama token or None
    """
    return os.environ.get("OLLAMA_TOKEN")


class EnvironmentConfig:
    """
    Centralized environment configuration.
    Provides clean access to all environment variables with caching.
    """
    
    @property
    def model_prefix(self) -> str:
        return get_model_prefix()
    
    @property
    def hf_username(self) -> str:
        return get_hf_username()
    
    @property
    def git_username(self) -> str:
        return get_git_username()
    
    @property
    def team_name(self) -> str:
        return get_team_name()
    
    @property
    def organization(self) -> str:
        return get_organization()
    
    @property
    def hf_token(self) -> Optional[str]:
        return get_hf_token()
    
    @property
    def ollama_token(self) -> Optional[str]:
        return get_ollama_token()
    
    @property
    def async_upload_enabled(self) -> bool:
        return parse_flag_from_env("ENABLE_ASYNC_UPLOAD", False)


# Singleton instance
env_config = EnvironmentConfig()
