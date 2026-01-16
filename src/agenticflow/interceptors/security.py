"""
Security interceptors.

Interceptors for security concerns like PII detection and masking,
content filtering, and audit logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticflow.interceptors.base import (
    InterceptContext,
    Interceptor,
    InterceptResult,
    StopExecution,
)


class PIIAction(Enum):
    """Action to take when PII is detected."""
    MASK = "mask"      # Replace PII with [REDACTED]
    BLOCK = "block"    # Stop execution
    WARN = "warn"      # Log warning but continue
    LOG = "log"        # Log detection, no modification


# Common PII patterns
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        re.IGNORECASE,
    ),
    "phone_us": re.compile(
        r"\b(?:\+1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    ),
    "ssn": re.compile(
        r"\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b",
    ),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
        r"6(?:011|5[0-9]{2})[0-9]{12})\b",
    ),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    ),
    "date_of_birth": re.compile(
        r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-]"
        r"(?:19|20)[0-9]{2}\b",
    ),
    "passport": re.compile(
        r"\b[A-Z]{1,2}[0-9]{6,9}\b",
    ),
    "api_key": re.compile(
        r"\b(?:sk|pk|api|key|token|secret|password)[-_]?"
        r"[A-Za-z0-9]{20,}\b",
        re.IGNORECASE,
    ),
}


@dataclass
class PIIShield(Interceptor):
    """Detects and handles personally identifiable information (PII).

    Scans messages and tool results for PII patterns and takes
    configured action (mask, block, warn, or log).

    Attributes:
        patterns: List of PII types to detect. Options:
            - "email", "phone_us", "ssn", "credit_card"
            - "ip_address", "date_of_birth", "passport", "api_key"
            - Or "all" for all patterns
        action: What to do when PII is found.
        mask_char: Character to use for masking (default "*").
        custom_patterns: Dict of name -> regex pattern to add.
        block_message: Message when blocking.

    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.interceptors import PIIShield, PIIAction

        # Mask emails and SSNs
        agent = Agent(
            name="assistant",
            model=model,
            intercept=[
                PIIShield(
                    patterns=["email", "ssn"],
                    action=PIIAction.MASK,
                ),
            ],
        )

        # Block if credit card detected
        agent = Agent(
            name="secure_agent",
            model=model,
            intercept=[
                PIIShield(
                    patterns=["credit_card", "ssn"],
                    action=PIIAction.BLOCK,
                ),
            ],
        )
        ```
    """

    patterns: list[str] = field(default_factory=lambda: ["all"])
    action: PIIAction = PIIAction.MASK
    mask_char: str = "*"
    mask_length: int = 8
    custom_patterns: dict[str, str] = field(default_factory=dict)
    block_message: str = "PII detected. Request blocked for security."

    def __post_init__(self) -> None:
        """Build pattern set."""
        self._patterns: dict[str, re.Pattern] = {}

        if "all" in self.patterns:
            self._patterns = dict(PII_PATTERNS)
        else:
            for name in self.patterns:
                if name in PII_PATTERNS:
                    self._patterns[name] = PII_PATTERNS[name]

        # Add custom patterns
        for name, pattern in self.custom_patterns.items():
            self._patterns[name] = re.compile(pattern, re.IGNORECASE)

    def _scan_text(self, text: str) -> list[tuple[str, str, int, int]]:
        """Scan text for PII matches.

        Returns:
            List of (pii_type, matched_text, start, end) tuples.
        """
        matches = []
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                matches.append((
                    pii_type,
                    match.group(),
                    match.start(),
                    match.end(),
                ))
        return matches

    def _mask_text(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Mask PII in text.

        Returns:
            Tuple of (masked_text, list of (pii_type, original) pairs).
        """
        detections = []
        result = text

        # Process matches in reverse order to preserve positions
        matches = sorted(self._scan_text(text), key=lambda m: m[2], reverse=True)

        for pii_type, matched, start, end in matches:
            mask = f"[{pii_type.upper()}_REDACTED]"
            result = result[:start] + mask + result[end:]
            detections.append((pii_type, matched))

        return result, detections

    def _scan_message(self, msg: dict[str, Any]) -> list[tuple[str, str]]:
        """Scan a single message for PII."""
        detections = []
        content = msg.get("content", "")

        if isinstance(content, str):
            detections.extend(
                (pii_type, matched)
                for pii_type, matched, _, _ in self._scan_text(content)
            )
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    detections.extend(
                        (pii_type, matched)
                        for pii_type, matched, _, _ in self._scan_text(text)
                    )

        return detections

    def _mask_message(self, msg: dict[str, Any]) -> tuple[dict, list[tuple[str, str]]]:
        """Mask PII in a message, returning new message and detections."""
        all_detections = []
        new_msg = dict(msg)

        content = msg.get("content", "")

        if isinstance(content, str):
            masked, detections = self._mask_text(content)
            new_msg["content"] = masked
            all_detections.extend(detections)
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    masked, detections = self._mask_text(item.get("text", ""))
                    new_content.append({**item, "text": masked})
                    all_detections.extend(detections)
                else:
                    new_content.append(item)
            new_msg["content"] = new_content

        return new_msg, all_detections

    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Scan messages before model call."""
        return await self._process_messages(ctx)

    async def post_act(self, ctx: InterceptContext) -> InterceptResult:
        """Scan tool results."""
        if ctx.tool_result is None:
            return InterceptResult.ok()

        result_str = str(ctx.tool_result)
        matches = self._scan_text(result_str)

        if not matches:
            return InterceptResult.ok()

        # Track in state
        ctx.state.setdefault("pii_shield", {"detections": []})
        ctx.state["pii_shield"]["detections"].extend(
            {"type": t, "value": v, "source": f"tool:{ctx.tool_name}"}
            for t, v, _, _ in matches
        )

        if self.action == PIIAction.BLOCK:
            raise StopExecution(
                self.block_message,
                f"PII detected in tool result: {[m[0] for m in matches]}",
            )

        return InterceptResult.ok()

    async def _process_messages(self, ctx: InterceptContext) -> InterceptResult:
        """Process messages based on action."""
        # Initialize tracking
        ctx.state.setdefault("pii_shield", {"detections": []})

        # Scan all messages
        all_detections = []
        for msg in ctx.messages:
            detections = self._scan_message(msg)
            all_detections.extend(detections)

        if not all_detections:
            return InterceptResult.ok()

        # Track detections
        ctx.state["pii_shield"]["detections"].extend(
            {"type": t, "value": v, "source": "message"}
            for t, v in all_detections
        )

        # Handle based on action
        if self.action == PIIAction.BLOCK:
            raise StopExecution(
                self.block_message,
                f"PII detected: {[d[0] for d in all_detections]}",
            )

        if self.action == PIIAction.MASK:
            # Mask all messages
            new_messages = []
            for msg in ctx.messages:
                masked_msg, _ = self._mask_message(msg)
                new_messages.append(masked_msg)
            return InterceptResult.modify_messages(new_messages)

        # WARN or LOG - just continue
        return InterceptResult.ok()

    def get_detections(self, ctx: InterceptContext) -> list[dict]:
        """Get all PII detections from context state."""
        return ctx.state.get("pii_shield", {}).get("detections", [])


@dataclass
class ContentFilter(Interceptor):
    """Filters messages containing blocked keywords or patterns.

    Useful for preventing certain topics or content from being processed.

    Attributes:
        blocked_words: List of words/phrases to block.
        blocked_patterns: List of regex patterns to block.
        action: "block" or "mask".
        case_sensitive: Whether matching is case-sensitive.
        message: Message when blocking.

    Example:
        ```python
        agent = Agent(
            name="safe_agent",
            model=model,
            intercept=[
                ContentFilter(
                    blocked_words=["password", "secret"],
                    action="block",
                ),
            ],
        )
        ```
    """

    blocked_words: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    action: str = "block"  # "block" or "mask"
    case_sensitive: bool = False
    message: str = "Content blocked by filter."

    def __post_init__(self) -> None:
        """Compile patterns."""
        flags = 0 if self.case_sensitive else re.IGNORECASE

        self._patterns: list[re.Pattern] = []

        # Add word patterns (with word boundaries)
        for word in self.blocked_words:
            escaped = re.escape(word)
            self._patterns.append(re.compile(rf"\b{escaped}\b", flags))

        # Add regex patterns
        for pattern in self.blocked_patterns:
            self._patterns.append(re.compile(pattern, flags))

    def _check_text(self, text: str) -> list[str]:
        """Check text for blocked content. Returns list of matches."""
        matches = []
        for pattern in self._patterns:
            found = pattern.findall(text)
            matches.extend(found)
        return matches

    async def pre_think(self, ctx: InterceptContext) -> InterceptResult:
        """Check messages for blocked content."""
        for msg in ctx.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                matches = self._check_text(content)
                if matches and self.action == "block":
                    raise StopExecution(
                        self.message,
                        f"Blocked content: {matches}",
                    )

        return InterceptResult.ok()


__all__ = [
    "PIIAction",
    "PIIShield",
    "ContentFilter",
]
