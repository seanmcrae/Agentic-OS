"""
StructuredGenerator - LLM Structured Output with Schema Enforcement
====================================================================

Drives any callable LLM to produce structured (JSON/YAML) output that
conforms to a caller-supplied ``OutputSchema``.  Key capabilities:

- Multiple extraction strategies (direct parse, code-block extraction,
  brace-balanced extraction, YAML fallback)
- Automatic retry with error feedback sent back to the model
- Type coercion to match declared schema field types
- Mock data generation for offline / test usage when no model caller is
  provided
- Async-first API; single-item and list generation helpers included
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
# Enumerations
# ---------------------------------------------------------------------------

class ExtractMode(str, Enum):
    """Controls how the generator extracts structured data from LLM output."""

    JSON = "json"
    """Expect raw JSON output."""

    YAML = "yaml"
    """Expect raw YAML output."""

    MARKDOWN_CODE_BLOCK = "markdown_code_block"
    """Expect output wrapped in a fenced code block (``` json / ``` yaml)."""

    AUTO = "auto"
    """Try all strategies in order and return the first successful parse."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OutputSchema:
    """
    Declarative description of the desired output structure.

    Attributes:
        name: Human-readable schema identifier (e.g. ``"UserProfile"``).
        description: Short description used in the prompt to the LLM.
        fields: Mapping of field names to JSON-schema-compatible type
            descriptors (``{"age": {"type": "integer"}, ...}``).
        required: Field names that must be present in valid output.
        examples: Optional list of example dicts shown to the LLM.
    """

    name: str
    description: str
    fields: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    examples: List[dict] = field(default_factory=list)

    def to_json_schema(self) -> dict:
        """Return a minimal JSON Schema object for this OutputSchema."""
        return {
            "type": "object",
            "title": self.name,
            "description": self.description,
            "properties": self.fields,
            "required": self.required,
        }

    def field_names(self) -> List[str]:
        return list(self.fields.keys())


@dataclass
class GenerationResult:
    """
    The result of a single structured-generation call.

    Attributes:
        success: ``True`` if valid structured data was produced.
        data: Parsed and validated dict, or ``None`` on failure.
        raw_output: The full LLM response text (last attempt).
        error: Human-readable error message when ``success`` is ``False``.
        attempts: Number of generation + parse attempts made.
        latency_ms: Wall-clock time from first call to final result.
    """

    success: bool
    data: Optional[dict]
    raw_output: str
    error: Optional[str] = None
    attempts: int = 1
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# StructuredGenerator
# ---------------------------------------------------------------------------

class StructuredGenerator:
    """
    Generates structured (JSON/YAML) output from an LLM and validates it
    against a caller-supplied ``OutputSchema``.

    Args:
        model_caller: Async or sync callable with signature
            ``(prompt: str) -> str``.  When ``None``, mock data is returned.
        max_retries: How many times to retry after a parse failure before
            giving up.  Defaults to ``3``.
        extract_mode: Extraction strategy.  ``ExtractMode.AUTO`` tries all
            strategies in sequence.
        temperature: Passed to the model caller when possible.  Kept on the
            instance for callers that wrap ``model_caller`` with it.
    """

    def __init__(
        self,
        model_caller: Optional[Callable] = None,
        max_retries: int = 3,
        extract_mode: ExtractMode = ExtractMode.AUTO,
        temperature: float = 0.1,
    ) -> None:
        self._model_caller = model_caller
        self.max_retries = max_retries
        self.extract_mode = extract_mode
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: OutputSchema,
        context: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate a single structured object conforming to *schema*.

        On each failure the error message is fed back to the model as a
        correction request, up to ``max_retries`` times.

        Args:
            prompt: User instruction or question.
            schema: Output schema describing the expected structure.
            context: Optional additional context prepended to the built prompt.

        Returns:
            A :class:`GenerationResult` with parsed data or error details.
        """
        t_start = time.monotonic()

        if self._model_caller is None:
            mock = _mock_data(schema)
            return GenerationResult(
                success=True,
                data=mock,
                raw_output=json.dumps(mock, indent=2),
                attempts=1,
                latency_ms=(time.monotonic() - t_start) * 1000,
            )

        built_prompt = self._build_prompt(prompt, schema, context)
        last_raw = ""
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = await self._call_model(built_prompt)
                last_raw = raw
                data = self._extract_json(raw)
                if data is None:
                    raise ValueError("Could not extract a valid JSON object from the response.")

                data = self._coerce_types(data, schema)
                valid, errors = self.validate(data, schema)
                if not valid:
                    raise ValueError("Schema validation failed: " + "; ".join(errors))

                return GenerationResult(
                    success=True,
                    data=data,
                    raw_output=raw,
                    attempts=attempt,
                    latency_ms=(time.monotonic() - t_start) * 1000,
                )

            except Exception as exc:
                last_error = str(exc)
                logger.debug("Attempt %d/%d failed: %s", attempt, self.max_retries, last_error)
                if attempt < self.max_retries:
                    built_prompt = self._build_retry_prompt(built_prompt, last_raw, last_error)

        return GenerationResult(
            success=False,
            data=None,
            raw_output=last_raw,
            error=last_error,
            attempts=self.max_retries,
            latency_ms=(time.monotonic() - t_start) * 1000,
        )

    async def generate_list(
        self,
        prompt: str,
        item_schema: OutputSchema,
        min_items: int = 1,
        max_items: int = 10,
    ) -> GenerationResult:
        """
        Generate a JSON array of objects each conforming to *item_schema*.

        Args:
            prompt: User instruction describing the items to generate.
            item_schema: Schema for each individual list item.
            min_items: Minimum acceptable number of items in the response.
            max_items: Maximum number of items to request.

        Returns:
            A :class:`GenerationResult` whose ``data`` is a dict with key
            ``"items"`` containing the list, e.g. ``{"items": [...]}``.
        """
        t_start = time.monotonic()
        list_schema = OutputSchema(
            name=f"{item_schema.name}List",
            description=(
                f"A list of {min_items}–{max_items} {item_schema.name} objects. "
                f"Each item: {item_schema.description}"
            ),
            fields={
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": item_schema.fields},
                    "minItems": min_items,
                    "maxItems": max_items,
                }
            },
            required=["items"],
        )

        list_prompt = (
            f"{prompt}\n\n"
            f"Return a JSON object with a single key \"items\" containing an array of "
            f"{min_items} to {max_items} objects."
        )

        result = await self.generate(list_prompt, list_schema)
        if result.success:
            items = result.data.get("items", [])  # type: ignore[union-attr]
            if not isinstance(items, list):
                result.success = False
                result.error = '"items" key is not a list'
                result.data = None
            elif len(items) < min_items:
                result.success = False
                result.error = (
                    f"Expected at least {min_items} items, got {len(items)}"
                )
                result.data = None

        result.latency_ms = (time.monotonic() - t_start) * 1000
        return result

    def validate(self, data: dict, schema: OutputSchema) -> Tuple[bool, List[str]]:
        """
        Validate *data* against *schema*.

        Currently checks:
        - All required fields are present and non-null
        - Field types match the declared ``"type"`` when provided

        Args:
            data: The dict to validate.
            schema: Schema to validate against.

        Returns:
            ``(is_valid, error_messages)``
        """
        errors: List[str] = []

        for req in schema.required:
            if req not in data or data[req] is None:
                errors.append(f"Required field '{req}' is missing or null.")

        for field_name, field_spec in schema.fields.items():
            if field_name not in data:
                continue
            declared_type = field_spec.get("type") if isinstance(field_spec, dict) else None
            if declared_type is None:
                continue
            value = data[field_name]
            if not _matches_json_type(value, declared_type):
                errors.append(
                    f"Field '{field_name}' expected type '{declared_type}', "
                    f"got {type(value).__name__!r}."
                )

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    # Extraction strategies
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Attempt to parse a JSON object from *text* using multiple strategies.

        Strategies (tried in order):
        1. ``json.loads(text)`` directly
        2. Extract from a ```json…``` fenced code block
        3. Extract from any fenced code block (``` … ```)
        4. Find the first ``{…}`` with balanced braces
        5. Extract from a ```yaml…``` code block and parse with PyYAML
        """
        # Strategy 1: direct parse
        result = _try_json_parse(text.strip())
        if result is not None:
            return result

        # Strategy 2: ```json ... ``` code block
        result = _extract_from_code_block(text, lang="json")
        if result is not None:
            return result

        # Strategy 3: any fenced code block
        result = _extract_from_code_block(text, lang=None)
        if result is not None:
            return result

        # Strategy 4: balanced-brace extraction
        result = _extract_balanced_braces(text)
        if result is not None:
            return result

        # Strategy 5: YAML code block
        result = _extract_from_yaml_block(text)
        if result is not None:
            return result

        return None

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        user_prompt: str,
        schema: OutputSchema,
        context: Optional[str],
    ) -> str:
        """
        Construct a full prompt that instructs the LLM to emit valid JSON
        matching *schema*.
        """
        parts: List[str] = []

        if context:
            parts.append(f"Context:\n{context}\n")

        parts.append(
            "You are a precise data extraction assistant. "
            "Respond ONLY with valid JSON — no prose, no markdown, no code fences."
        )

        parts.append(f"\nTask:\n{user_prompt}\n")

        parts.append(f"\nOutput schema — {schema.name}:")
        parts.append(f"Description: {schema.description}")

        if schema.fields:
            field_lines = []
            for fname, fspec in schema.fields.items():
                req_marker = " (required)" if fname in schema.required else ""
                if isinstance(fspec, dict):
                    ftype = fspec.get("type", "any")
                    fdesc = fspec.get("description", "")
                    field_lines.append(f"  - {fname} [{ftype}]{req_marker}: {fdesc}")
                else:
                    field_lines.append(f"  - {fname}{req_marker}: {fspec}")
            parts.append("Fields:\n" + "\n".join(field_lines))

        if schema.examples:
            example_str = json.dumps(schema.examples[0], indent=2)
            parts.append(f"\nExample output:\n{example_str}")

        parts.append(
            "\nRespond with a single valid JSON object and nothing else."
        )
        return "\n".join(parts)

    def _build_retry_prompt(
        self,
        original_prompt: str,
        bad_output: str,
        error: str,
    ) -> str:
        """Build a correction prompt that includes the previous error."""
        return (
            f"{original_prompt}\n\n"
            f"---\n"
            f"Your previous response could not be parsed. "
            f"Fix this JSON error: {error}\n\n"
            f"Previous (broken) response:\n{bad_output}\n\n"
            f"Return ONLY the corrected JSON object."
        )

    # ------------------------------------------------------------------
    # Type coercion
    # ------------------------------------------------------------------

    def _coerce_types(self, data: dict, schema: OutputSchema) -> dict:
        """
        Attempt to coerce field values in *data* to the types declared in
        *schema*.  Leaves unknown or un-coercible values untouched.
        """
        coerced = dict(data)
        for field_name, field_spec in schema.fields.items():
            if field_name not in coerced:
                continue
            if not isinstance(field_spec, dict):
                continue
            declared_type = field_spec.get("type")
            if declared_type is None:
                continue
            coerced[field_name] = _coerce_value(coerced[field_name], declared_type)
        return coerced

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_model(self, prompt: str) -> str:
        """Invoke the model caller, supporting both async and sync callables."""
        if self._model_caller is None:
            raise RuntimeError("No model_caller configured.")
        try:
            if asyncio.iscoroutinefunction(self._model_caller):
                return await self._model_caller(prompt)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._model_caller, prompt)
        except Exception as exc:
            logger.error("Model caller raised: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Module-level parsing helpers
# ---------------------------------------------------------------------------

def _try_json_parse(text: str) -> Optional[dict]:
    """Try ``json.loads``; return None on any error."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_from_code_block(text: str, lang: Optional[str]) -> Optional[dict]:
    """
    Extract JSON from a fenced Markdown code block.

    If *lang* is given, only matches blocks with that language tag.
    If *lang* is None, matches any fenced block.
    """
    lang_pattern = re.escape(lang) if lang else r"[a-zA-Z]*"
    pattern = rf"```{lang_pattern}\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return _try_json_parse(match.group(1).strip())
    return None


def _extract_balanced_braces(text: str) -> Optional[dict]:
    """Find the first well-formed ``{...}`` substring and parse it."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _try_json_parse(text[start : i + 1])
    return None


def _extract_from_yaml_block(text: str) -> Optional[dict]:
    """Extract a ```yaml...``` block and convert it to a dict via PyYAML."""
    match = re.search(r"```yaml\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        import yaml  # type: ignore[import]
        obj = yaml.safe_load(match.group(1).strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _matches_json_type(value: Any, json_type: str) -> bool:
    """Return True if *value* is compatible with the JSON Schema *json_type*."""
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = mapping.get(json_type)
    if expected is None:
        return True  # Unknown type — accept anything
    # JSON booleans are ints in Python; be strict here
    if json_type == "integer" and isinstance(value, bool):
        return False
    if json_type == "number" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _coerce_value(value: Any, json_type: str) -> Any:
    """
    Attempt to coerce *value* to the Python equivalent of *json_type*.
    Returns the original value if coercion is not possible or not needed.
    """
    if json_type == "string":
        return str(value) if not isinstance(value, str) else value
    if json_type == "integer":
        if isinstance(value, str):
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
        if isinstance(value, float):
            return int(value)
    if json_type == "number":
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
    if json_type == "boolean":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
    if json_type == "array":
        if isinstance(value, str):
            parsed = _try_json_parse(f'[{value}]') or _try_json_parse(value)
            if isinstance(parsed, list):
                return parsed
    return value


def _mock_data(schema: OutputSchema) -> dict:
    """
    Generate plausible mock data for *schema* field names and types.

    Used when no model_caller is configured so that callers can test
    downstream code without a live LLM.
    """
    _type_defaults: Dict[str, Any] = {
        "string": "example_value",
        "integer": 42,
        "number": 3.14,
        "boolean": True,
        "array": [],
        "object": {},
        "null": None,
    }
    result: Dict[str, Any] = {}
    for fname, fspec in schema.fields.items():
        if isinstance(fspec, dict):
            ftype = fspec.get("type", "string")
            result[fname] = _type_defaults.get(ftype, f"<{fname}>")
        else:
            result[fname] = f"<{fname}>"
    return result
