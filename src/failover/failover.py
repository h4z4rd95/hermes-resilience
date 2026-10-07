"""Hermes Resilience — Model/Provider Failover Layer (Lane R2).

Audits existing fallback_providers config, implements ordered fallback chain
with bounded retries, classification of error types, cooldowns, and durable
BLOCKED/FAILED state when all candidates exhausted.

Does NOT duplicate the existing Hermes cron fallback_providers; it wraps the
existing model selection with extra resilience and observability.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("resilience.failover")


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #

class FailoverCategory(Enum):
    RATE_LIMIT = "rate_limit"          # HTTP 429 / quota exhausted
    NETWORK = "network"                # Connection refused / timeout
    AUTH = "auth"                      # 401 / invalid key
    PROVIDER_OVERLOAD = "provider_overload"  # 503 / service unavailable
    MODEL_UNAVAILABLE = "model_unavailable"  # model_id not found
    TIMEOUT = "timeout"                # request exceeded budget
    CONFIG_ERROR = "config_error"      # bad config / missing key
    UNKNOWN = "unknown"


class ProviderState(Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    EXHAUSTED = "exhausted"


@dataclass
class ProviderEntry:
    """One entry in the fallback chain."""
    name: str
    base_url: str
    model: str
    priority: int            # lower = higher priority
    cooldown_s: int = 300    # 5 min cooldown after failure
    max_retries: int = 2     # per-provider retry budget
    retries_remaining: int = 0
    last_failure: float = 0
    state: ProviderState = ProviderState.ACTIVE
    error_messages: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize retries_remaining and state on construction."""
        if self.retries_remaining == 0 and self.max_retries == 0:
            self.retries_remaining = 0
            self.state = ProviderState.EXHAUSTED
        elif self.retries_remaining == 0:
            self.retries_remaining = self.max_retries


@dataclass
class FailoverResult:
    model_used: str = ""
    provider_used: str = ""
    success: bool = False
    category: FailoverCategory = FailoverCategory.UNKNOWN
    error: str = ""
    retries: int = 0
    chain: list[dict] = field(default_factory=list)  # attempt history
    fallback_occurred: bool = False
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# Config reader — reads from existing Hermes config.yaml fallback_providers
# --------------------------------------------------------------------------- #

class FallbackChain:
    """Ordered model/provider fallback chain with bounded retries.
    
    Reads the existing config.yaml fallback_providers if present,
    or falls back to the providers declared in config.yaml model section.
    """

    def __init__(self, hermes_home: Path):
        self.config_path = hermes_home / "config.yaml"
        self.providers: list[ProviderEntry] = []
        self.cooldown_state: dict[str, float] = {}  # provider → cooldown_expiry
        self.load_config()

    def load_config(self):
        """Parse config.yaml and build fallback chain from providers section."""
        if not self.config_path.exists():
            logger.warning("No config.yaml at %s", self.config_path)
            return

        content = self.config_path.read_text()

        # Check for fallback_providers (structured list)
        if "fallback_providers:" in content:
            self._parse_fallback_providers(content)
        else:
            # Build from model + providers sections
            self._build_from_model_config(content)

        # Sort by priority
        self.providers.sort(key=lambda p: p.priority)
        logger.info("Fallback chain loaded: %d providers", len(self.providers))
        for i, p in enumerate(self.providers):
            logger.info("  [%d] %s → %s (%s)", i, p.name, p.model, p.base_url)

    def _parse_fallback_providers(self, content: str):
        """Parse structured fallback_providers list from config.yaml."""
        import re
        # Match the fallback_providers block
        block_match = re.search(
            r'fallback_providers:\s*\n((?:\s+-\s+.*\n)*)', content
        )
        if not block_match:
            return

        block = block_match.group(1)
        lines = block.strip().split('\n')
        priority = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Extract provider name from "- name: xxx" or similar
            name_match = re.search(r'(?:name|provider):\s*(\S+)', line)
            model_match = re.search(r'model:\s*(\S+)', line)
            url_match = re.search(r'(?:base_url|url):\s*(\S+)', line)

            if name_match:
                name = name_match.group(1).strip("'\"")
                model = model_match.group(1).strip("'\"") if model_match else ""
                url = url_match.group(1).strip("'\"") if url_match else ""

                self.providers.append(ProviderEntry(
                    name=name, base_url=url, model=model,
                    priority=priority
                ))
                priority += 1

    def _build_from_model_config(self, content: str):
        """Build fallback chain from model + providers sections."""
        import re

        # Primary model
        model_match = re.search(r'model:\s*(\S+)', content)
        provider_match = re.search(r'provider:\s*(\S+)', content)
        base_url_match = re.search(r'base_url:\s*(\S+)', content)

        if provider_match and model_match:
            primary = ProviderEntry(
                name=provider_match.group(1),
                model=model_match.group(1),
                base_url=(base_url_match.group(1) if base_url_match else ""),
                priority=0
            )
            self.providers.append(primary)

        # Secondary providers
        for name in ["omniroute", "9router", "ollama", "local"]:
            name_block = re.search(
                rf'{re.escape(name)}:\s*\n((?:\s+\S+:.*\n)*)', content
            )
            if name_block:
                block = name_block.group(1)
                url_m = re.search(r'base_url:\s*(\S+)', block)
                model_m = re.search(r'model:\s*(\S+)', block)
                if url_m:
                    self.providers.append(ProviderEntry(
                        name=name,
                        base_url=url_m.group(1).strip("'\""),
                        model=model_m.group(1).strip("'\"") if model_m else "",
                        priority=len(self.providers) + 1
                    ))

    def get_next_candidate(self, exclude: Optional[str] = None) -> Optional[ProviderEntry]:
        """Get the next available provider in priority order."""
        now = time.time()
        for p in self.providers:
            if p.name == exclude:
                continue
            if p.state == ProviderState.EXHAUSTED:
                continue
            if p.state == ProviderState.COOLDOWN:
                cooldown_end = self.cooldown_state.get(p.name, 0)
                if now < cooldown_end:
                    continue
                else:
                    p.state = ProviderState.ACTIVE
                    p.retries_remaining = p.max_retries
            if p.retries_remaining <= 0:
                p.retries_remaining = p.max_retries
            return p
        return None

    def mark_failure(self, provider_name: str, error: str, category: FailoverCategory):
        """Record a provider failure and apply cooldown + retry budget."""
        for p in self.providers:
            if p.name == provider_name:
                p.retries_remaining -= 1
                p.error_messages.append(f"{category.value}: {error}")
                if p.retries_remaining <= 0:
                    p.state = ProviderState.COOLDOWN
                    self.cooldown_state[provider_name] = time.time() + p.cooldown_s
                if provider_name not in self.cooldown_state:
                    self.cooldown_state[provider_name] = time.time() + p.cooldown_s
                break

    def mark_success(self, provider_name: str):
        """Reset cooldown and retry budget on success."""
        for p in self.providers:
            if p.name == provider_name:
                p.state = ProviderState.ACTIVE
                p.retries_remaining = p.max_retries
                self.cooldown_state.pop(provider_name, None)
                break

    def is_exhausted(self) -> bool:
        """True when all providers are in COOLDOWN or EXHAUSTED state."""
        for p in self.providers:
            if p.state in (ProviderState.ACTIVE,):
                return False
        return True

    def reset(self):
        """Reset all providers to active state."""
        for p in self.providers:
            p.state = ProviderState.ACTIVE
            p.retries_remaining = p.max_retries
        self.cooldown_state.clear()

    def dump(self) -> dict:
        """Current chain state for logging."""
        return {
            "providers": [
                {
                    "name": p.name,
                    "model": p.model,
                    "base_url": p.base_url,
                    "priority": p.priority,
                    "state": p.state.value,
                    "retries_left": p.retries_remaining,
                    "cooldown_until": self.cooldown_state.get(p.name),
                    "recent_errors": p.error_messages[-3:],
                }
                for p in self.providers
            ]
        }


# --------------------------------------------------------------------------- #
# Failover executor — attempts providers in order with error classification
# --------------------------------------------------------------------------- #

def classify_error(error: str, http_code: int | None = None) -> FailoverCategory:
    """Classify an error into a failover category."""
    if http_code == 429 or "429" in error or "rate" in error.lower() or "throttl" in error.lower():
        return FailoverCategory.RATE_LIMIT
    if http_code == 401 or "401" in error or "auth" in error.lower() or "unauthorized" in error.lower():
        return FailoverCategory.AUTH
    if http_code == 503 or "503" in error or "unavailable" in error.lower():
        return FailoverCategory.PROVIDER_OVERLOAD
    if http_code == 404 or "not found" in error.lower() or "not supported" in error.lower():
        return FailoverCategory.MODEL_UNAVAILABLE
    if "timeout" in error.lower() or "timed out" in error.lower() or http_code in (504,):
        return FailoverCategory.TIMEOUT
    if "connection" in error.lower() or "refused" in error.lower() or "network" in error.lower():
        return FailoverCategory.NETWORK
    return FailoverCategory.UNKNOWN


def attempt_provider(
    provider: ProviderEntry,
    prompt: str,
    model_override: str | None = None,
    budget_s: int = 120,
) -> tuple[bool, str, FailoverCategory]:
    """Attempt one provider and return (success, error_text, category).
    
    Uses the existing Hermes CLI with model/provider overrides.
    """
    model = model_override or provider.model
    env = dict(os.environ)
    # Set provider-specific env vars if available
    key_env = f"HERMES_CUSTOM_{provider.name.upper()}_API_KEY"
    if key_env not in env:
        key_env = f"HERMES_CUSTOM_{provider.name.upper().replace('-', '_')}_API_KEY"
    
    try:
        result = subprocess.run(
            [
                "hermes", "chat",
                "-q", prompt[:1000],  # brief test prompt
                "--model", model,
                "--provider", provider.name,
                "--max-turns", "1",
                "-Q",
            ],
            capture_output=True, text=True, timeout=budget_s,
            env=env, cwd="/root/.hermes"
        )
        if result.returncode == 0:
            return True, "", FailoverCategory.UNKNOWN
        else:
            error_text = (result.stderr or result.stdout or "unknown error")[:500]
            # Try to extract HTTP code from error
            code = None
            import re
            code_match = re.search(r'(\d{3})', error_text)
            if code_match:
                code = int(code_match.group(1))
            category = classify_error(error_text, code)
            return False, error_text, category
    except subprocess.TimeoutExpired:
        return False, f"timeout after {budget_s}s", FailoverCategory.TIMEOUT
    except Exception as e:
        return False, str(e), FailoverCategory.NETWORK


def execute_failover(
    prompt: str,
    chain: FallbackChain,
    budget_s: int = 120,
    max_attempts: int = 10,
) -> FailoverResult:
    """Execute failover chain: try providers in priority order.
    
    Returns FailoverResult with the provider/model that succeeded,
    or a BLOCKED state when all are exhausted.
    """
    result = FailoverResult()
    attempt = 0
    current_exclude: str | None = None

    while attempt < max_attempts:
        candidate = chain.get_next_candidate(exclude=current_exclude)
        if candidate is None:
            result.category = FailoverCategory.UNKNOWN
            result.error = "all providers exhausted or in cooldown"
            if chain.is_exhausted():
                result.category = FailoverCategory.MODEL_UNAVAILABLE
                result.error = "all fallback providers exhausted — BLOCKED state"
            break

        result.chain.append({
            "provider": candidate.name,
            "model": candidate.model,
            "attempt": attempt + 1,
        })

        success, error, category = attempt_provider(
            candidate, prompt, budget_s=budget_s
        )
        attempt += 1

        if success:
            result.success = True
            result.model_used = candidate.model
            result.provider_used = candidate.name
            result.retries = attempt - 1
            if attempt > 1:
                result.fallback_occurred = True
            chain.mark_success(candidate.name)
            break
        else:
            result.error = error
            result.category = category
            result.retries = attempt
            chain.mark_failure(candidate.name, error, category)
            current_exclude = candidate.name

    return result