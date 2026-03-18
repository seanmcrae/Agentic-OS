"""
ConstitutionalAI - Constitutional AI Safety Layer
==================================================

Implements Constitutional AI (CAI) principles for filtering, critiquing,
and revising LLM outputs to ensure they are harmless, helpful, and honest.

Reference: https://arxiv.org/abs/2212.08073

Two-tier safety checking:
1. Fast heuristic pass: regex-based detection of PII, known toxicity
   keywords, prompt-injection patterns, and self-harm indicators.
2. LLM-assisted pass (when model_caller is provided): uses the model to
   critique the text against each active Principle and optionally revise
   violating content in-place.

Usage::

    safety = ConstitutionalAI(model_caller=my_llm, auto_revise=True)
    result = await safety.check(llm_response, context=user_query)
    if not result.is_safe:
        print("Violations:", result.violations)
    safe_text = result.revised
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Severity level of a safety principle violation."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Principle:
    """
    A single constitutional principle.

    Attributes:
        id: Machine-readable identifier (e.g. ``"harmlessness_violence"``).
        name: Short human-readable name.
        description: Full description of what the principle prohibits or requires.
        critique_prompt: Prompt fragment used to ask the LLM whether this
            principle is violated.
        revision_prompt: Prompt fragment used to ask the LLM to revise the
            text so that it no longer violates this principle.
        severity: Impact level of a violation.
    """

    id: str
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str
    severity: Severity = Severity.MEDIUM


@dataclass
class SafetyViolation:
    """
    A single detected safety violation.

    Attributes:
        principle_id: The ``id`` of the violated :class:`Principle`.
        principle_name: Human-readable principle name.
        severity: Severity of the violation.
        description: Explanation of why the text was flagged.
        suggested_revision: Optional rewrite that resolves the violation.
    """

    principle_id: str
    principle_name: str
    severity: Severity
    description: str
    suggested_revision: str = ""


@dataclass
class SafetyResult:
    """
    The complete result of a safety check.

    Attributes:
        is_safe: ``True`` if no violations were found (or all were resolved by
            revision).
        original: The text that was checked.
        revised: The (potentially revised) text.  Equal to ``original`` when no
            revision was applied.
        violations: List of detected :class:`SafetyViolation` objects.
        critique: Free-text critique summary produced by the LLM (empty when
            heuristic-only mode).
        revision_applied: ``True`` if the text was actually revised.
        latency_ms: Wall-clock duration of the full check in milliseconds.
    """

    is_safe: bool
    original: str
    revised: str
    violations: List[SafetyViolation] = field(default_factory=list)
    critique: str = ""
    revision_applied: bool = False
    latency_ms: float = 0.0

    @property
    def highest_severity(self) -> Optional[Severity]:
        """Return the highest severity among all violations, or None."""
        if not self.violations:
            return None
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        for sev in order:
            if any(v.severity == sev for v in self.violations):
                return sev
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "is_safe": self.is_safe,
            "revision_applied": self.revision_applied,
            "violation_count": len(self.violations),
            "highest_severity": self.highest_severity.value if self.highest_severity else None,
            "latency_ms": round(self.latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Preset principle collections
# ---------------------------------------------------------------------------

class PrincipleSet:
    """Factory methods that return curated lists of :class:`Principle` objects."""

    @classmethod
    def harmlessness(cls) -> List[Principle]:
        """Principles focused on preventing harm."""
        return [
            Principle(
                id="harmlessness_violence",
                name="No Violence",
                description=(
                    "The response must not encourage, glorify, or provide instructions "
                    "for acts of violence against any person or group."
                ),
                critique_prompt=(
                    "Does this response encourage or describe acts of violence? "
                    "Identify any such content."
                ),
                revision_prompt=(
                    "Rewrite the response to remove all content that encourages "
                    "or describes violence."
                ),
                severity=Severity.HIGH,
            ),
            Principle(
                id="harmlessness_weapons",
                name="No Weapons Instructions",
                description="Must not provide step-by-step instructions for creating weapons.",
                critique_prompt=(
                    "Does this response provide actionable instructions for creating "
                    "weapons or dangerous devices?"
                ),
                revision_prompt=(
                    "Remove or generalize any step-by-step instructions that could "
                    "be used to create weapons."
                ),
                severity=Severity.CRITICAL,
            ),
            Principle(
                id="harmlessness_self_harm",
                name="No Self-Harm Encouragement",
                description="Must not encourage or instruct self-harm or suicide.",
                critique_prompt=(
                    "Does this response contain content that encourages self-harm "
                    "or suicide?"
                ),
                revision_prompt=(
                    "Rewrite the response to remove any encouragement of self-harm "
                    "and, where appropriate, suggest professional help resources."
                ),
                severity=Severity.CRITICAL,
            ),
            Principle(
                id="harmlessness_pii",
                name="No PII Exposure",
                description=(
                    "Must not expose personally identifiable information (PII) such as "
                    "SSNs, credit card numbers, phone numbers, or email addresses of "
                    "real individuals without consent."
                ),
                critique_prompt=(
                    "Does this response contain personally identifiable information "
                    "that should not be shared publicly?"
                ),
                revision_prompt=(
                    "Replace or redact any personally identifiable information in the "
                    "response."
                ),
                severity=Severity.HIGH,
            ),
            Principle(
                id="harmlessness_illegal",
                name="No Illegal Activity Facilitation",
                description="Must not facilitate or provide instructions for illegal activities.",
                critique_prompt=(
                    "Does this response help facilitate or provide instructions for "
                    "clearly illegal activities?"
                ),
                revision_prompt=(
                    "Remove facilitation of illegal activities while preserving "
                    "educational or otherwise legitimate content."
                ),
                severity=Severity.HIGH,
            ),
        ]

    @classmethod
    def helpfulness(cls) -> List[Principle]:
        """Principles focused on ensuring genuinely helpful responses."""
        return [
            Principle(
                id="helpfulness_actionable",
                name="Actionable Guidance",
                description="Responses should provide useful, actionable guidance when asked.",
                critique_prompt=(
                    "Is this response unhelpful, evasive, or does it fail to address "
                    "the user's question?"
                ),
                revision_prompt=(
                    "Revise the response to be more directly helpful and actionable "
                    "while remaining safe."
                ),
                severity=Severity.LOW,
            ),
            Principle(
                id="helpfulness_clarity",
                name="Clarity",
                description="Responses should be clear and easy to understand.",
                critique_prompt=(
                    "Is this response confusing, ambiguous, or unnecessarily complex?"
                ),
                revision_prompt="Rewrite the response using clearer, simpler language.",
                severity=Severity.LOW,
            ),
        ]

    @classmethod
    def honesty(cls) -> List[Principle]:
        """Principles focused on truthfulness and transparency."""
        return [
            Principle(
                id="honesty_no_deception",
                name="No Deception",
                description="Must not make deliberately false or misleading statements.",
                critique_prompt=(
                    "Does this response contain false claims, misleading framing, or "
                    "deliberate deception?"
                ),
                revision_prompt=(
                    "Correct any false or misleading statements to be accurate and "
                    "transparent."
                ),
                severity=Severity.HIGH,
            ),
            Principle(
                id="honesty_uncertainty",
                name="Acknowledge Uncertainty",
                description=(
                    "Should acknowledge uncertainty rather than fabricate confident answers."
                ),
                critique_prompt=(
                    "Does this response present speculative or uncertain information "
                    "as established fact?"
                ),
                revision_prompt=(
                    "Add appropriate qualifiers (e.g., 'I believe', 'I'm not certain') "
                    "where the response expresses uncertainty."
                ),
                severity=Severity.MEDIUM,
            ),
            Principle(
                id="honesty_no_impersonation",
                name="No Impersonation",
                description=(
                    "Must not impersonate real people or claim to be human when asked "
                    "directly."
                ),
                critique_prompt=(
                    "Does this response impersonate a real person or falsely claim "
                    "to be human?"
                ),
                revision_prompt=(
                    "Correct the response to accurately represent the speaker's identity."
                ),
                severity=Severity.HIGH,
            ),
        ]

    @classmethod
    def full(cls) -> List[Principle]:
        """All built-in principles combined."""
        return cls.harmlessness() + cls.helpfulness() + cls.honesty()


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# PII
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_RE_PHONE = re.compile(
    r"(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?\d{4})"
)
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RE_CREDIT_CARD = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"        # Visa
    r"5[1-5][0-9]{14}|"                       # MasterCard
    r"3[47][0-9]{13}|"                        # Amex
    r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"       # Diners
    r"6(?:011|5[0-9]{2})[0-9]{12}|"          # Discover
    r"(?:2131|1800|35\d{3})\d{11})\b"        # JCB
)
_RE_PASSWORD = re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE)

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (_RE_SSN, "SSN"),
    (_RE_CREDIT_CARD, "Credit card number"),
    (_RE_EMAIL, "Email address"),
    (_RE_PHONE, "Phone number"),
    (_RE_PASSWORD, "Password"),
]

# Prompt injection (case-insensitive)
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?",
        r"disregard\s+(?:all\s+)?(?:previous|prior|your)\s+instructions?",
        r"forget\s+(?:all\s+)?previous\s+instructions?",
        r"you\s+are\s+now\s+(?:a|an|the)\s+\w+",
        r"act\s+as\s+(?:a|an|the)\s+\w+\s+(?:without|ignoring)",
        r"jailbreak",
        r"dan\s*mode",
        r"pretend\s+you\s+(?:have\s+no|don'?t\s+have)\s+(?:any\s+)?(?:restrictions?|rules?|guidelines?)",
        r"override\s+(?:all\s+)?safety",
        r"your\s+new\s+instructions?\s+are",
    ]
]

# Toxicity keywords (~20 severe terms; not exhaustive by design)
_TOXICITY_KEYWORDS: List[str] = [
    "nigger", "faggot", "kike", "spic", "chink", "wetback",
    "retard", "cunt", "whore",
    "rape", "lynching",
    "nazi", "genocide", "exterminate", "ethnic cleansing",
    "child porn", "lolicon",
    "bomb making", "how to make a bomb",
    "anthrax", "ricin synthesis",
]
_RE_TOXICITY = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _TOXICITY_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Self-harm indicators
_SELF_HARM_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:kill\s+(?:your)?self|end\s+(?:your\s+)?life|commit\s+suicide)\b",
        r"\b(?:self[- ]harm|self[- ]injur|cut\s+(?:your)?self)\b",
        r"\bhow\s+to\s+(?:kill|hang|poison)\s+(?:my|your)?self\b",
        r"(?:how\s+to|instructions?\s+for|steps?\s+to)\s+(?:kill\s+(?:yourself|myself)|suicide|self.?harm)",
    ]
]


# ---------------------------------------------------------------------------
# ConstitutionalAI
# ---------------------------------------------------------------------------

class ConstitutionalAI:
    """
    Constitutional AI safety layer for LLM outputs.

    Applies a configurable set of :class:`Principle` objects to critique
    and optionally revise text.  When a ``model_caller`` is provided, the LLM
    is used to perform nuanced critique and revision.  In all cases, a fast
    heuristic pass runs first.

    Args:
        model_caller: Async or sync callable ``(prompt: str) -> str``.
            When ``None``, only heuristic checks are performed.
        principles: List of :class:`Principle` objects to apply.  Defaults to
            :meth:`PrincipleSet.full`.
        auto_revise: When ``True`` (default) and violations are found, ask the
            LLM to revise the text automatically.
        max_revision_rounds: Maximum number of critique-then-revise cycles
            before giving up.  Defaults to ``2``.
    """

    def __init__(
        self,
        model_caller: Optional[Callable] = None,
        principles: Optional[List[Principle]] = None,
        auto_revise: bool = True,
        max_revision_rounds: int = 2,
    ) -> None:
        self._model_caller = model_caller
        self.principles: List[Principle] = (
            principles if principles is not None else PrincipleSet.full()
        )
        self.auto_revise = auto_revise
        self.max_revision_rounds = max_revision_rounds

        # Internal stats
        self._checks_run: int = 0
        self._violations_found: int = 0
        self._revisions_applied: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self, text: str, context: str = "") -> SafetyResult:
        """
        Run a full safety check on *text*.

        Steps:
        1. Heuristic fast pass (always).
        2. LLM-assisted critique (when model_caller is set).
        3. Optional LLM-assisted revision (up to max_revision_rounds).

        Args:
            text: The text to check.
            context: Optional prompt / context that produced the text.

        Returns:
            A :class:`SafetyResult` describing the outcome.
        """
        t_start = time.monotonic()
        self._checks_run += 1

        # Step 1: fast heuristic pass
        violations = self.heuristic_check(text)

        # Step 2: LLM critique (if available)
        critique = ""
        if self._model_caller is not None:
            try:
                llm_violations = await self.critique(text, self.principles)
                existing_ids = {v.principle_id for v in violations}
                for v in llm_violations:
                    if v.principle_id not in existing_ids:
                        violations.append(v)
                        existing_ids.add(v.principle_id)
                if llm_violations:
                    critique = "; ".join(v.description for v in llm_violations)
            except Exception as exc:
                logger.warning("LLM critique failed, heuristics only: %s", exc)

        # Step 3: Optional revision
        revised = text
        revision_applied = False

        if violations and self.auto_revise:
            if self._model_caller is not None:
                for _round in range(self.max_revision_rounds):
                    try:
                        candidate = await self.revise(revised, violations)
                        if candidate and candidate.strip() != revised.strip():
                            revised = candidate.strip()
                            revision_applied = True
                            remaining = self.heuristic_check(revised)
                            if not remaining:
                                violations = remaining
                                break
                            violations = remaining
                    except Exception as exc:
                        logger.warning(
                            "LLM revision round %d failed: %s", _round + 1, exc
                        )
                        break
            else:
                # Heuristic-only redaction
                revised = self._heuristic_revise(text, violations)
                if revised != text:
                    revision_applied = True

        if violations:
            self._violations_found += len(violations)
        if revision_applied:
            self._revisions_applied += 1

        is_safe = not any(
            v.severity in (Severity.HIGH, Severity.CRITICAL)
            for v in violations
        )

        return SafetyResult(
            is_safe=is_safe,
            original=text,
            revised=revised,
            violations=violations,
            critique=critique,
            revision_applied=revision_applied,
            latency_ms=(time.monotonic() - t_start) * 1000,
        )

    async def critique(
        self,
        text: str,
        principles: List[Principle],
    ) -> List[SafetyViolation]:
        """
        Use the LLM to identify which of *principles* are violated by *text*.

        The model is prompted to return a JSON array of violation objects.
        Empty list is returned if no violations are found or the model call
        fails.

        Args:
            text: Text to critique.
            principles: Principles to check.

        Returns:
            List of violations found by the LLM.
        """
        if not self._model_caller or not principles:
            return []

        principle_list = "\n".join(
            f"{i + 1}. [{p.severity.value}] {p.name}: {p.description}"
            for i, p in enumerate(principles)
        )

        prompt = (
            f"You are a safety auditor. Review the following text for violations "
            f"of these principles:\n\n"
            f"{principle_list}\n\n"
            f"Text to review:\n---\n{text}\n---\n\n"
            f"For each violated principle, respond with a JSON array:\n"
            f'[\n  {{"principle_id": "...", "severity": "LOW|MEDIUM|HIGH|CRITICAL", '
            f'"description": "...", "suggested_revision": "..."}}\n]\n\n'
            f"If no violations are found, respond with: []\n"
            f"Only report actual violations, not theoretical concerns."
        )

        try:
            response = await self._call_model(prompt)
            return self._parse_violation_json(
                response, {p.id: p for p in principles}
            )
        except Exception as exc:
            logger.warning("Critique call failed: %s", exc)
            return []

    async def revise(
        self,
        text: str,
        violations: List[SafetyViolation],
    ) -> str:
        """
        Ask the LLM to rewrite *text* so that it no longer violates the
        given *violations*.

        Args:
            text: The text to revise.
            violations: Violations that the revision should address.

        Returns:
            The revised text, or the original text if the call fails.
        """
        if not violations or self._model_caller is None:
            return text

        violation_lines = "\n".join(
            f"- [{v.severity.value}] {v.principle_name}: {v.description}"
            for v in violations
        )

        prompt = (
            f"You are a safety editor. Rewrite the following response to address "
            f"these safety violations while preserving the original meaning and "
            f"helpfulness as much as possible.\n\n"
            f"Violations to fix:\n{violation_lines}\n\n"
            f"Original text:\n---\n{text}\n---\n\n"
            f"Write ONLY the revised text with no commentary or explanation."
        )

        try:
            revised = await self._call_model(prompt)
            # Strip common LLM prefixes
            for prefix in [
                "Revised response:", "Here is the revised response:",
                "Revised text:", "Response:",
            ]:
                if revised.strip().startswith(prefix):
                    revised = revised.strip()[len(prefix):].strip()
            return revised
        except Exception as exc:
            logger.warning("Revision call failed: %s", exc)
            return text

    def heuristic_check(self, text: str) -> List[SafetyViolation]:
        """
        Perform a fast, regex-based safety check without LLM calls.

        Detects:
        - PII: email addresses, phone numbers, SSNs, credit card numbers,
          password patterns
        - Prompt injection patterns (ignore instructions, jailbreak, etc.)
        - ~20 known severe toxicity keywords
        - Self-harm / suicide indicators

        Args:
            text: Text to check.

        Returns:
            List of :class:`SafetyViolation` objects; empty when text is clean.
        """
        violations: List[SafetyViolation] = []

        # PII detection
        for pattern, pii_label in _PII_PATTERNS:
            if pattern.search(text):
                violations.append(SafetyViolation(
                    principle_id="harmlessness_pii",
                    principle_name="No PII Exposure",
                    severity=Severity.HIGH,
                    description=f"Detected {pii_label} in response.",
                    suggested_revision=f"Redact or remove the {pii_label}.",
                ))
                break  # One PII violation per pass is sufficient

        # Prompt injection
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(SafetyViolation(
                    principle_id="harmlessness_prompt_injection",
                    principle_name="No Prompt Injection",
                    severity=Severity.HIGH,
                    description=(
                        f"Potential prompt injection detected: "
                        f"'{match.group(0)}'"
                    ),
                    suggested_revision=(
                        "Remove the instruction-override language from the text."
                    ),
                ))
                break

        # Toxicity
        tox_match = _RE_TOXICITY.search(text)
        if tox_match:
            violations.append(SafetyViolation(
                principle_id="harmlessness_toxicity",
                principle_name="No Toxic Content",
                severity=Severity.CRITICAL,
                description=(
                    f"Severely offensive language detected: "
                    f"'{tox_match.group(0)}'"
                ),
                suggested_revision="Remove or replace the offensive language.",
            ))

        # Self-harm
        for pattern in _SELF_HARM_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(SafetyViolation(
                    principle_id="harmlessness_self_harm",
                    principle_name="No Self-Harm Encouragement",
                    severity=Severity.CRITICAL,
                    description=(
                        f"Self-harm or suicide indicator detected: "
                        f"'{match.group(0)}'"
                    ),
                    suggested_revision=(
                        "Replace with crisis resources and supportive language."
                    ),
                ))
                break

        return violations

    async def filter_batch(self, texts: List[str]) -> List[SafetyResult]:
        """
        Check a batch of texts in parallel.

        Args:
            texts: List of strings to check.

        Returns:
            List of :class:`SafetyResult` in the same order as *texts*.
        """
        return list(await asyncio.gather(*[self.check(t) for t in texts]))

    def get_stats(self) -> Dict[str, Any]:
        """Return accumulated statistics since instantiation."""
        return {
            "checks_run": self._checks_run,
            "violations_found": self._violations_found,
            "revisions_applied": self._revisions_applied,
            "violation_rate": (
                self._violations_found / self._checks_run
                if self._checks_run > 0
                else 0.0
            ),
            "principles_active": len(self.principles),
            "has_model": self._model_caller is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_model(self, prompt: str) -> str:
        """Invoke the model caller, supporting both async and sync callables."""
        if self._model_caller is None:
            raise RuntimeError("No model_caller configured.")
        if asyncio.iscoroutinefunction(self._model_caller):
            return await self._model_caller(prompt)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._model_caller, prompt)

    def _parse_violation_json(
        self,
        response: str,
        principles_by_id: Dict[str, Principle],
    ) -> List[SafetyViolation]:
        """Parse an LLM JSON response into :class:`SafetyViolation` objects."""
        response = response.strip()
        # Extract JSON array from response (may contain prose before/after)
        m = re.search(r"\[[\s\S]*?\]", response)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            violations: List[SafetyViolation] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                pid = item.get("principle_id", "unknown")
                principle = principles_by_id.get(pid)
                sev_str = item.get("severity", "MEDIUM").upper()
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = principle.severity if principle else Severity.MEDIUM

                violations.append(SafetyViolation(
                    principle_id=pid,
                    principle_name=principle.name if principle else pid,
                    severity=severity,
                    description=item.get("description", ""),
                    suggested_revision=item.get("suggested_revision", ""),
                ))
            return violations
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to parse critique JSON: %s", exc)
            return []

    def _heuristic_revise(self, text: str, violations: List[SafetyViolation]) -> str:
        """
        Apply simple regex-based redactions to handle PII violations
        without calling an LLM.
        """
        revised = text
        pii_ids = {v.principle_id for v in violations if "pii" in v.principle_id}
        if not pii_ids:
            return revised

        revised = _RE_SSN.sub("[SSN REDACTED]", revised)
        revised = _RE_CREDIT_CARD.sub("[CARD REDACTED]", revised)
        revised = _RE_EMAIL.sub("[EMAIL REDACTED]", revised)
        revised = _RE_PHONE.sub("[PHONE REDACTED]", revised)
        revised = _RE_PASSWORD.sub("password: [REDACTED]", revised)
        return revised

    def __repr__(self) -> str:
        return (
            f"ConstitutionalAI("
            f"principles={len(self.principles)}, "
            f"auto_revise={self.auto_revise}, "
            f"has_model={'yes' if self._model_caller else 'no'})"
        )
